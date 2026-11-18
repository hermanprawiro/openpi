"""The episode runner: the 20 Hz loop that turns observations into policy requests and policy actions into
streamer waypoints, evaluating safety every tick (spec §6.4-6.6).

Per tick, in order: operator key (`p` pause, Enter resume, `q` end) -> streamer health (unhealthy ->
FREEZE) -> snapshot -> `safety.evaluate` over the config's required streams -> HOLD / pause handling ->
inference results -> new request -> one action to the streamer (unless the tracking leash is engaged or
the queue is empty) -> gripper keepalive -> `recorder.step`.

HOLD and pause both: `streamer.hold()`, drop the queue, bump the generation epoch (results of requests
sent before are discarded) and, when RUN returns, replan; the first chunk after the start or after a HOLD
must start within `max_start_offset_rad` of the measured pose (else FREEZE). FREEZE ends the episode; the
caller (Session) releases the controllers. Every ending holds the streamer.

The loop runs on a deadline grid of `1 / policy.control_hz`; a tick that finishes more than one period
late re-anchors the grid (`next = now + period`, counted in `overruns`): no catch-up ticks.
"""

from collections.abc import Callable
import dataclasses
import logging
import math
import time
from typing import Literal, Protocol

from eval_client.chunking import AsyncChunkBuffer
from eval_client.chunking import ChunkExecutor
from eval_client.inference import InferenceWorker
import numpy as np

from robot_client import layout
from robot_client import safety
from robot_client.config import SiteConfig
from robot_client.keys import ENTER
from robot_client.keys import KeySource
from robot_client.preprocess import IMAGE_KEYS
from robot_client.preprocess import Snapshot
from robot_client.preprocess import build_observation
from robot_client.preprocess import to_policy_image
from robot_client.recorder import EpisodeRecorder
from robot_client.streamer import Streamer

logger = logging.getLogger(__name__)

# Epochs of one episode are generation * _EPOCH_STRIDE + k (k = number of HOLDs so far), so results of an
# earlier episode or of an earlier HOLD epoch never match the current one.
_EPOCH_STRIDE = 1_000_000
GRIPPER_KEEPALIVE_S = 1.0

EndedBy = Literal["timeout", "operator", "freeze"]


class ObsSource(Protocol):
    def snapshot(self) -> Snapshot: ...


@dataclasses.dataclass(frozen=True)
class EpisodeOutcome:
    ended_by: EndedBy
    reason: str
    steps: int  # completed ticks
    stats: dict


def required_stream_keys(cfg: SiteConfig) -> tuple[str, ...]:
    """Streams that must be fresh: every camera, every arm's measured joints, each required gripper."""
    keys = [f"cam/{name}" for name in cfg.cameras]
    keys += [f"arm/{side}" for side in cfg.arms]
    keys += [f"grip/{side}" for side, arm in cfg.arms.items() if arm.gripper.required]
    return tuple(keys)


class _End(Exception):  # noqa: N818 - control flow, not an error
    def __init__(self, ended_by: EndedBy, reason: str):
        super().__init__(reason)
        self.ended_by = ended_by
        self.reason = reason


class _SyncQueue:
    """One blocking-style request per `execute_horizon` actions: request only when empty and idle."""

    def __init__(self, cfg: SiteConfig):
        self._executor = ChunkExecutor(cfg.inference.execute_horizon, action_dim=layout.ACTION_DIM)
        self._depth = 0
        self.in_flight = False

    def __len__(self) -> int:
        return self._depth

    def wants_request(self) -> bool:
        return not self.in_flight and self._depth == 0

    def on_request_sent(self) -> None:
        self.in_flight = True

    def on_error(self) -> None:
        self.in_flight = False

    def on_chunk(self, obs_step: int, chunk: np.ndarray, step: int, *, first: bool) -> None:
        self.in_flight = False
        self._executor.load(chunk)
        self._depth += min(len(chunk), self._executor.replan_steps)

    def pop(self) -> np.ndarray | None:
        if self._depth == 0:
            return None
        self._depth -= 1
        return self._executor.next_action()

    def stats(self) -> dict:
        return {}


class _AsyncQueue:
    """`AsyncChunkBuffer` with the munich16 dims. A fresh queue (after HOLD) keeps the accumulated stats."""

    _LISTS = ("depth_at_pop", "latency_steps", "step_jumps", "switch_jumps")
    _COUNTERS = ("stale_skipped", "dry_steps", "dry_events", "chunks_dropped")

    def __init__(self, cfg: SiteConfig, carry: "_AsyncQueue | None" = None):
        a = cfg.inference.async_cfg
        self.buf = AsyncChunkBuffer(
            cfg.inference.execute_horizon,
            a.trigger_threshold,
            a.merge,
            a.blend_steps,
            action_dim=layout.ACTION_DIM,
            arm_dims=layout.ARM_DIMS,
            blend_dims=layout.ARM_DIMS,
        )
        if carry is not None:
            for name in (*self._LISTS, *self._COUNTERS):
                setattr(self.buf, name, getattr(carry.buf, name))

    def __len__(self) -> int:
        return len(self.buf)

    @property
    def in_flight(self) -> bool:
        return self.buf.in_flight

    def wants_request(self) -> bool:
        return self.buf.should_request()

    def on_request_sent(self) -> None:
        self.buf.on_request_sent()

    def on_error(self) -> None:
        self.buf.in_flight = False

    def on_chunk(self, obs_step: int, chunk: np.ndarray, step: int, *, first: bool) -> None:
        # The first chunk after a (re)start was computed while the robot stood still: no latency skip.
        self.buf.on_chunk(obs_step, chunk, next_step=obs_step if first else step)

    def pop(self) -> np.ndarray | None:
        if len(self.buf) == 0:
            self.buf.pop()  # counts the dry step; returns the held last action, which we do not resend
            return None
        return self.buf.pop()

    def stats(self) -> dict:
        return self.buf.stats()


class Runner:
    def __init__(
        self,
        cfg: SiteConfig,
        node: ObsSource,
        worker: InferenceWorker,
        streamer: Streamer,
        grippers: Callable[[str, float], None],
        recorder: EpisodeRecorder,
        keys: KeySource,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = cfg
        self.node = node
        self.worker = worker
        self.streamer = streamer
        self.grippers = grippers
        self.recorder = recorder
        self.keys = keys
        self.clock = clock
        self.sleep = sleep
        self.active = tuple(cfg.active_arms)
        self.period = 1.0 / cfg.policy.control_hz
        self.required = required_stream_keys(cfg)
        self.image_every = max(1, cfg.policy.control_hz // 2)  # same 2 Hz cadence as the recorder

    # ---- episode ----

    def run_episode(self, generation: int) -> EpisodeOutcome:
        self._begin(generation)
        try:
            ended_by, reason = self._loop()
        finally:
            try:
                self.streamer.hold()
            except Exception:
                logger.exception("runner: streamer.hold() at the end of the episode failed")
        logger.info("episode ended by %s after %d steps: %s", ended_by, self.step, reason)
        return EpisodeOutcome(ended_by, reason, self.step, self._stats())

    def _begin(self, generation: int) -> None:
        self.epoch = generation * _EPOCH_STRIDE
        self.queue = self._new_queue()
        self.step = 0
        self.paused = False
        self.holding = False  # streamer.hold() issued for the current HOLD / pause
        self.first_chunk = True  # next chunk gets the start-offset check
        self.dry_since: float | None = None
        self.prev_wp: dict[str, np.ndarray] = {}
        self.grip_last: dict[str, float] = {}
        self.grip_t: dict[str, float] = {}
        self.t_request: dict[int, float] = {}
        self.infer_ms: list[float] = []
        self.counts = dict.fromkeys(
            (
                "overruns",
                "wait_steps",
                "leash_steps",
                "hold_steps",
                "pause_steps",
                "holds",
                "requests",
                "chunks",
                "infer_errors",
                "waypoints",
                "step_clamps",
                "gripper_publishes",
            ),
            0,
        )

    def _new_queue(self, carry=None):
        if self.cfg.inference.mode == "async":
            return _AsyncQueue(self.cfg, carry)
        return _SyncQueue(self.cfg)

    def _loop(self) -> tuple[EndedBy, str]:
        timeout = self.cfg.safety.episode_timeout_s
        t0 = self.clock()
        anchor, n = t0, 0  # deadline of the next tick = anchor + n * period
        while True:
            now = self.clock()
            if now - t0 >= timeout:
                return "timeout", f"episode_timeout_s {timeout} reached"
            try:
                self._tick(now, t0)
            except _End as end:
                return end.ended_by, end.reason
            self.step += 1
            n += 1
            deadline = anchor + n * self.period
            now = self.clock()
            if now - deadline > self.period:
                self.counts["overruns"] += 1
                logger.warning("runner: tick %d finished %.3f s late; re-anchoring", self.step - 1, now - deadline)
                anchor, n = now + self.period, 0
                deadline = anchor
            if deadline > now:
                self.sleep(deadline - now)

    # ---- tick ----

    def _tick(self, now: float, t0: float) -> None:
        key = self.keys.poll()
        if key == "q":
            raise _End("operator", "operator pressed q")
        if key == "p" and not self.paused:
            self.paused = True
            logger.info("runner: paused by the operator (Enter resumes)")
        elif key == ENTER and self.paused:
            self.paused = False
            logger.info("runner: resumed by the operator")

        if not self.streamer.healthy():
            raise _End("freeze", f"command streamer unhealthy: {self.streamer.fault or 'publishing stopped'}")

        snap = self.node.snapshot()
        ages = {k: snap.ages.get(k, math.inf) for k in self.required}
        dry_s = 0.0 if self.dry_since is None else now - self.dry_since
        fatal = self.worker.error is not None or not self.worker.is_alive()
        mode, why = safety.evaluate(ages, dry_s, fatal, self.cfg.safety)
        if mode is safety.Mode.FREEZE:
            if fatal:
                why += f" ({self.worker.error or 'inference worker thread is not running'})"
            raise _End("freeze", why)

        stopped = self.paused or mode is safety.Mode.HOLD
        if stopped and not self.holding:
            self._enter_hold("operator pause" if self.paused else why)
        elif not stopped:
            self.holding = False

        self._handle_results(snap)
        obs = action = None
        if self.paused:
            status = "paused"
            self.counts["pause_steps"] += 1
        elif mode is safety.Mode.HOLD:
            status = "hold"
            self.counts["hold_steps"] += 1
        else:
            if self.queue.wants_request():
                obs = self._request(snap)
            if any(self.streamer.leash_engaged(side) for side in self.active):
                status = "leash"
                self.counts["leash_steps"] += 1
            else:
                action = self.queue.pop()
                if action is None:
                    status = "wait"
                    self.counts["wait_steps"] += 1
                    if self.dry_since is None:
                        self.dry_since = now
                else:
                    status = "run"
                    self.dry_since = None
                    self._execute(action, snap)
        self._gripper_keepalive(now)
        reason = "operator pause" if self.paused else (why if mode is not safety.Mode.RUN else None)
        self._record(now - t0, status, reason, dry_s, snap, action, obs)

    def _enter_hold(self, why: str) -> None:
        logger.warning("runner: HOLD (%s)", why)
        self.streamer.hold()
        self.holding = True
        self.counts["holds"] += 1
        self.epoch += 1
        self.queue = self._new_queue(self.queue)
        self.first_chunk = True
        self.dry_since = None

    def _handle_results(self, snap: Snapshot) -> None:
        for result in list(self.worker.drain(self.epoch)):
            if result.error is not None:
                self.counts["infer_errors"] += 1
                logger.warning("runner: inference failed (%s); requesting again", result.error)
                self.queue.on_error()
                continue
            try:
                chunk = layout.validate_chunk(result.actions)
            except ValueError as e:
                raise _End("freeze", f"unusable action chunk from the server: {e}") from e
            self.counts["chunks"] += 1
            self.infer_ms.append(result.infer_ms)
            t_request = self.t_request.pop(result.obs_step, math.nan)
            self.recorder.chunk(result.obs_step, t_request, self.clock(), self.step - result.obs_step, chunk)
            if self.first_chunk:
                self._check_start(chunk, snap)
            self.queue.on_chunk(result.obs_step, chunk, self.step, first=self.first_chunk)
            self.first_chunk = False

    def _check_start(self, chunk: np.ndarray, snap: Snapshot) -> None:
        limit = self.cfg.safety.max_start_offset_rad
        for side in self.active:
            measured = snap.arm_q.get(side)
            offset = math.inf if measured is None else safety.start_offset(chunk[0, layout.ARM_SLICES[side]], measured)
            if not offset <= limit:
                raise _End(
                    "freeze",
                    f"first action of the {side} arm is {offset:.3f} rad from measured > max_start_offset_rad {limit}",
                )

    def _request(self, snap: Snapshot) -> dict:
        try:
            obs = build_observation(snap, self.cfg.cameras, self.cfg.policy.prompt, pin=self.cfg.state.pin)
        except ValueError as e:  # StateError, missing image, bad crop
            raise _End("freeze", f"cannot build the observation: {e}") from e
        self.worker.submit(self.epoch, self.step, obs)
        self.queue.on_request_sent()
        self.t_request[self.step] = self.clock()
        self.counts["requests"] += 1
        return obs

    def _execute(self, action: np.ndarray, snap: Snapshot) -> None:
        max_step = self.cfg.safety.max_step_rad
        for side in self.active:
            prev = self.prev_wp.get(side)
            if prev is None:
                prev = np.asarray(snap.arm_q[side], dtype=np.float64)
            q, cut = safety.clamp_step(prev, action[layout.ARM_SLICES[side]], max_step)
            self.counts["step_clamps"] += int(cut)
            self.streamer.set_waypoint(side, q)
            self.prev_wp[side] = q
            self.counts["waypoints"] += 1
            width = float(np.clip(action[layout.GRIPPER_INDEX[side]], 0.0, 1.0))
            last = self.grip_last.get(side)
            if last is None or abs(width - last) > self.cfg.safety.gripper_deadband:
                self._publish_gripper(side, width)

    def _publish_gripper(self, side: str, width: float) -> None:
        self.grippers(side, width)
        self.grip_last[side] = width
        self.grip_t[side] = self.clock()
        self.counts["gripper_publishes"] += 1

    def _gripper_keepalive(self, now: float) -> None:
        for side, width in self.grip_last.items():
            if now - self.grip_t[side] >= GRIPPER_KEEPALIVE_S:
                self._publish_gripper(side, width)

    def _record(
        self, t: float, status: str, why: str | None, dry_s: float, snap: Snapshot, action, obs: dict | None
    ) -> None:
        self.recorder.step(
            {
                "step": self.step,
                "t": round(t, 4),
                "mode": status,
                "reason": why,
                "epoch": self.epoch,
                "queue": len(self.queue),
                "in_flight": self.queue.in_flight,
                "dry_s": round(dry_s, 3),
                "measured": {side: np.asarray(q).tolist() for side, q in snap.arm_q.items()},
                "knuckle": {side: float(v) for side, v in snap.knuckle.items()},
                "action": None if action is None else np.asarray(action).tolist(),
                "waypoints": {side: self.prev_wp[side].tolist() for side in self.active if side in self.prev_wp}
                if action is not None
                else None,
            }
        )
        if self.step % self.image_every == 0:
            self._record_images(snap, obs)

    def _record_images(self, snap: Snapshot, obs: dict | None) -> None:
        try:
            if obs is not None:
                images = {name: obs[key] for name, key in IMAGE_KEYS.items()}
            else:
                images = {
                    name: to_policy_image(snap.images[name], cam.train_res, cam.crop)
                    for name, cam in self.cfg.cameras.items()
                    if name in snap.images
                }
        except Exception:
            logger.exception("runner: could not preprocess images for the recorder")
            return
        self.recorder.images(self.step, images)

    def _stats(self) -> dict:
        stats = {"mode": self.cfg.inference.mode, "steps": self.step, **self.counts}
        stats["infer_ms_mean"] = round(float(np.mean(self.infer_ms)), 1) if self.infer_ms else None
        stats["infer_ms_max"] = round(float(np.max(self.infer_ms)), 1) if self.infer_ms else None
        stats.update(self.queue.stats())
        return stats
