"""The 100 Hz joint command stream: the only thing that publishes arm joint targets (spec §6.4-6.6).

Each tick (`step`), per side: linear interpolation from the current position to the latest 20 Hz
waypoint over `waypoint_dt` -> tracking leash (if the interpolated target is more than
`max_track_lag_rad` from the measured joints, the ramp does not advance) -> joint-limit clamp (limits
minus `q_margin_rad`) -> smoother -> hard clamp to the raw limits (a backstop that should never fire;
counted in `hard_clamps`) -> command adapter -> publish. Without waypoints it publishes the hold pose it was started
with, so there is always a command on the topic while the streamer runs.

`step(now)` does one tick for a given time, so it is testable with a fake clock; the thread only calls
it at `hz` on absolute deadlines. Supervision: if a tick raises, the thread records `fault` (the first
exception) and from then on republishes each side's last command every period until `stop()`; it
never stops publishing on its own. `healthy()` turns false, and the Runner / Session freeze and release.

Hold / resume (smoother contract): `hold()` freezes the ramp and calls the smoother's `stop()`, which
brakes to rest without coming back. A `set_waypoint` after that is applied once the smoother output has
stopped moving: `reset(last_output)` (velocity is zero there anyway) and the new ramp starts from the
smoother's output, not from the old interpolated position.
"""

from collections.abc import Callable, Sequence
import dataclasses
import logging
import threading
import time

import numpy as np

from robot_client.command import CommandAdapter
from robot_client.config import SafetyConfig
from robot_client.safety import clamp_limits
from robot_client.safety import lag_exceeded
from robot_client.smoother import Smoother

logger = logging.getLogger(__name__)

_REST_TOL_RAD = 1e-6  # a braking smoother whose output moved less than this in a period is at rest
_U_SNAP = 1e-9  # ramp parameter this close to 1 counts as the end of the segment (float accumulation)


def _joint_vector(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.array(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name}: shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: not finite: {arr.tolist()}")
    return arr


@dataclasses.dataclass(eq=False)
class _Arm:
    smoother: Smoother
    adapter: CommandAdapter
    q_min: np.ndarray
    q_max: np.ndarray
    # Ramp: target = start + u * (end - start), u advancing 0 -> 1 over waypoint_dt.
    start: np.ndarray | None = None
    end: np.ndarray | None = None
    u: float = 1.0
    holding: bool = False  # smoother.stop() called; ramp frozen
    resume: bool = False  # a waypoint arrived while holding; apply it once the smoother is at rest
    still: bool = False  # while holding: the last smoother output did not move
    out: np.ndarray | None = None  # last smoother output (joint space)
    leash: bool = False
    hard_clamped: bool = False  # the last tick's smoother output was outside [q_min, q_max]
    last_cmd: np.ndarray | None = None
    last_pub: float | None = None
    pubs: int = 0

    def target(self, u: float) -> np.ndarray:
        return self.start + u * (self.end - self.start)


class Streamer:
    def __init__(
        self,
        sides: Sequence[str],
        publish: Callable[[str, np.ndarray], None],
        measured: Callable[[str], np.ndarray],
        smoother_factory: Callable[[], Smoother],
        adapters: dict[str, CommandAdapter],
        limits: dict[str, tuple[np.ndarray, np.ndarray]],
        safety: SafetyConfig,
        hz: float = 100.0,
        waypoint_dt: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.sides = tuple(sides)
        if not self.sides or len(set(self.sides)) != len(self.sides):
            raise ValueError(f"sides must be non-empty and unique, got {self.sides}")
        if not hz > 0 or not waypoint_dt > 0:
            raise ValueError(f"hz and waypoint_dt must be > 0, got {hz}, {waypoint_dt}")
        for name, mapping in (("adapters", adapters), ("limits", limits)):
            missing = [side for side in self.sides if side not in mapping]
            if missing:
                raise ValueError(f"{name}: no entry for {missing}")
        self._arms: dict[str, _Arm] = {}
        for side in self.sides:
            q_min, q_max = (np.asarray(x, dtype=np.float64) for x in limits[side])
            if q_min.ndim != 1 or q_min.shape != q_max.shape:
                raise ValueError(f"limits[{side!r}]: q_min {q_min.shape} / q_max {q_max.shape} must be equal 1-D")
            self._arms[side] = _Arm(smoother_factory(), adapters[side], q_min, q_max)
        self._publish_fn = publish
        self._measured = measured
        self._max_lag = safety.max_track_lag_rad
        self._margin = safety.q_margin_rad
        self._max_gap = safety.stream_gap_freeze_s
        self.hz = float(hz)
        self.waypoint_dt = float(waypoint_dt)
        self._clock = clock
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._last_now: float | None = None
        self._publish_error_logged = False
        self.fault: str | None = None
        # Counters for the recorder.
        self.cycles = 0  # completed step() ticks
        self.max_gap_s = dict.fromkeys(self.sides, 0.0)  # largest interval between publishes, per side
        self.leash_events = 0  # times a side's leash engaged (transitions, not cycles)
        self.limit_clamps = 0  # (side, tick) pairs whose target was clamped to the joint limits
        self.hard_clamps = 0  # (side, tick) pairs whose smoother output had to be clamped to the raw limits

    # ---- control (any thread) ----

    def start(self, hold: dict[str, np.ndarray], *, thread: bool = True) -> None:
        """Seed every side at rest at `hold[side]` (published until the first waypoint) and start the
        thread; `thread=False` only seeds, for driving `step` by hand."""
        if self._started:
            raise RuntimeError("streamer already started")
        missing = [side for side in self.sides if side not in hold]
        if missing:
            raise ValueError(f"hold: no pose for {missing}")
        with self._lock:
            for side, arm in self._arms.items():
                q = _joint_vector(f"hold[{side!r}]", hold[side], arm.q_min.shape)
                arm.smoother.reset(q)
                arm.start, arm.end, arm.u, arm.out = q.copy(), q.copy(), 1.0, q.copy()
                arm.holding = arm.resume = arm.still = arm.leash = False
                arm.last_cmd = arm.adapter(q)
            self._started = True
        if thread:
            self._thread = threading.Thread(target=self._run, name="robot_client-streamer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the thread; the only way publishing stops."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.error("streamer thread did not stop within 1 s")

    def set_waypoint(self, side: str, q: np.ndarray) -> None:
        """Ramp linearly from the current interpolated position to `q` over `waypoint_dt`."""
        arm = self._arms[side]
        q = _joint_vector(f"waypoint[{side!r}]", q, arm.q_min.shape)
        with self._lock:
            self._require_started()
            arm.start, arm.end, arm.u = arm.target(arm.u), q, 0.0
            if arm.holding:
                arm.resume = True

    def hold(self, side: str | None = None) -> None:
        """Stop ramping and brake the smoother to rest where it is (one side, or all)."""
        for s in self.sides if side is None else (side,):
            arm = self._arms[s]
            with self._lock:
                self._require_started()
                here = arm.target(arm.u)
                arm.start, arm.end, arm.u = here, here.copy(), 1.0
                arm.holding, arm.resume, arm.still = True, False, False
                arm.smoother.stop()

    # ---- status (any thread) ----

    def healthy(self) -> bool:
        """No fault, the thread is alive and every side published within `stream_gap_freeze_s`."""
        if self.fault is not None or self._thread is None or not self._thread.is_alive():
            return False
        now = self._clock()
        return all(arm.last_pub is not None and now - arm.last_pub < self._max_gap for arm in self._arms.values())

    def leash_engaged(self, side: str) -> bool:
        return self._arms[side].leash

    def cycles_since_start(self, side: str) -> int:
        """Commands published for `side` since `start` (including republished ones after a fault)."""
        return self._arms[side].pubs

    def last_command(self, side: str) -> np.ndarray:
        """The last command published (or, before the first tick, about to be published) for `side`."""
        with self._lock:
            self._require_started()
            return self._arms[side].last_cmd.copy()

    # ---- the tick ----

    def step(self, now: float) -> dict[str, np.ndarray]:
        """One tick at time `now`: compute every side's command, then publish them. Raises on any error
        (the thread turns that into a fault); nothing is published for a tick that raised while
        computing."""
        with self._lock:
            self._require_started()
            dt = 1.0 / self.hz if self._last_now is None else max(0.0, now - self._last_now)
            self._last_now = now
            cmds = {side: self._advance(side, arm, dt) for side, arm in self._arms.items()}
            for side, cmd in cmds.items():
                self._arms[side].last_cmd = cmd
        for side, cmd in cmds.items():
            self._publish(side, cmd, now)
        self.cycles += 1
        return {side: cmd.copy() for side, cmd in cmds.items()}

    def _advance(self, side: str, arm: _Arm, dt: float) -> np.ndarray:
        if arm.holding and arm.resume and arm.still:
            arm.smoother.reset(arm.out)
            arm.start, arm.u = arm.out.copy(), 0.0
            arm.holding = arm.resume = arm.still = False

        u = arm.u if arm.holding else min(1.0, arm.u + dt / self.waypoint_dt)
        if u > 1.0 - _U_SNAP:
            u = 1.0
        target = arm.target(u)
        measured = np.asarray(self._measured(side), dtype=np.float64)
        if measured.shape != arm.q_min.shape:
            raise ValueError(f"measured[{side!r}]: shape {measured.shape} does not match {arm.q_min.shape}")
        if lag_exceeded(target, measured, self._max_lag):  # NaN measured counts as exceeded
            if not arm.leash:
                self.leash_events += 1
                logger.warning("%s: tracking lag > %.2f rad, waypoint ramp paused", side, self._max_lag)
            arm.leash = True
            u = arm.u
            target = arm.target(u)
        elif arm.leash:
            arm.leash = False
            logger.info("%s: tracking lag back within %.2f rad, ramp resumed", side, self._max_lag)
        moving = u > arm.u
        arm.u = u

        # The segment velocity is passed for the interface; the smoothers track position only (ruling R6).
        vel = (arm.end - arm.start) / self.waypoint_dt if moving else np.zeros_like(target)
        clamped_target, clamped = clamp_limits(target, arm.q_min, arm.q_max, self._margin)
        if clamped:
            self.limit_clamps += 1
            vel = np.where(clamped_target == target, vel, 0.0)

        out = np.array(arm.smoother.step(clamped_target, vel), dtype=np.float64)
        if arm.holding:
            arm.still = bool(np.max(np.abs(out - arm.out)) <= _REST_TOL_RAD)
        arm.out = out  # the smoother's own state (rest detection, resume), even if clamped below
        # Backstop: the smoothers land on the margin-clamped target without overshoot, so this should never
        # fire. If it does, publish the clamped value and leave the smoother alone (no reset).
        hard = np.clip(out, arm.q_min, arm.q_max)
        hard_clamped = not np.array_equal(hard, out, equal_nan=True)
        if hard_clamped:
            self.hard_clamps += 1
            if not arm.hard_clamped:
                logger.error("%s: smoother output %s outside the joint limits; clamped", side, out.tolist())
        arm.hard_clamped = hard_clamped
        cmd = np.asarray(arm.adapter(hard), dtype=np.float64)
        if cmd.shape != out.shape or not np.all(np.isfinite(cmd)):
            raise ValueError(f"{side}: invalid command {cmd.tolist()}")
        return cmd

    def _publish(self, side: str, cmd: np.ndarray, now: float) -> None:
        self._publish_fn(side, cmd.copy())
        arm = self._arms[side]
        if arm.last_pub is not None:
            self.max_gap_s[side] = max(self.max_gap_s[side], now - arm.last_pub)
        arm.last_pub = now
        arm.pubs += 1

    def _republish(self, now: float) -> None:
        for side in self.sides:
            try:
                self._publish(side, self.last_command(side), now)
            except Exception as e:  # keep trying every side, every period
                if self.fault is None:
                    self.fault = f"publish failed for {side}: {type(e).__name__}: {e}"
                if not self._publish_error_logged:
                    self._publish_error_logged = True
                    logger.exception("streamer: republishing the last command for %s failed", side)

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("streamer not started")

    # ---- the thread ----

    def _run(self) -> None:
        period = 1.0 / self.hz
        deadline = self._clock()
        while not self._stop_evt.is_set():
            now = self._clock()
            if self.fault is None:
                try:
                    self.step(now)
                except Exception as e:
                    self.fault = f"{type(e).__name__}: {e}"
                    logger.exception("streamer tick failed; republishing the last commands until stopped")
                    self._republish(now)
            else:
                self._republish(now)
            deadline += period
            now = self._clock()
            if deadline < now - period:  # overran by more than a period: re-anchor, do not burst
                deadline = now
            self._stop_evt.wait(max(0.0, deadline - now))
