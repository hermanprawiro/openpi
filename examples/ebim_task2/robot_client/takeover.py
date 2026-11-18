"""Controller takeover and release (spec §6.3, plan Task 12): when the joint impedance controller is switched on
and off, and in which order relative to the 100 Hz command stream.

Takeover per arm (`takeover`): measured joints present -> DDS discovery time elapsed -> no foreign messages on the
command topic -> the hold stream (= measured, seeded by `Session`) has published `warm_cycles` commands and still
equals measured within `tol_rad` -> by controller state: `inactive` -> re-check tolerance, `adapter.activate`,
`switch_controller` and confirm `active`; `active` already -> only capture the adapter reference (never deactivated
by us); anything else -> refuse. `takeover.mode: external` never calls switch_controller: the operator activates the
controller with their own tooling while we stream hold.

Release (`release`): hold the arm, wait, deactivate only what we activated. It never raises; errors are collected.
`Session` ties it together: wait for discovery and check every command topic for foreign messages before
publishing anything, start the stream with hold = measured, take over every side, and on exit (or any exception,
including KeyboardInterrupt, also during `__enter__`) release every side it took over, then stop the stream exactly
once and re-raise the original exception.

Pure logic against protocols; the ROS implementations come with the node (Task 16).
"""

from collections.abc import Callable, Mapping, Sequence
import logging
import sys
import time
from typing import Protocol

import numpy as np

from robot_client.config import ARM_NAMES
from robot_client.config import ArmConfig
from robot_client.config import SiteConfig
from robot_client.config import TakeoverConfig

logger = logging.getLogger(__name__)

OUR_FRAME_ID = "robot_client"  # header.frame_id of every command message we publish
FOREIGN_WINDOW_S = 1.0  # foreign messages on the command topic are counted over this window
EXTERNAL_WAIT_S = 60.0  # takeover.mode external: how long to wait for the operator
WARM_POLL_S = 0.01
CONFIRM_POLL_S = 0.05
EXTERNAL_POLL_S = 0.1
_EPS = 1e-9


class TakeoverError(RuntimeError):
    """A takeover pre-check or controller call failed; the message tells the operator what to do."""


class ControllerManager(Protocol):
    """One arm's controller_manager. Implementations bound every service call by their own timeout and raise on
    expiry."""

    def list_controllers(self) -> dict[str, str]:
        """Controller name -> state (`active`, `inactive`, `unconfigured`, ...)."""
        ...

    def switch(self, activate: list[str], deactivate: list[str], timeout_s: float) -> bool: ...


class GraphView(Protocol):
    def uptime_s(self) -> float:
        """Seconds since the ROS node started (DDS discovery needs a few seconds)."""
        ...

    def count_publishers(self, topic: str) -> int:
        """Publisher endpoints on `topic`, including our own command publisher."""
        ...

    def foreign_messages(self, topic: str, window_s: float) -> int:
        """Messages received on `topic` during the last `window_s` seconds whose `header.frame_id` is not
        `OUR_FRAME_ID`."""
        ...

    def measured(self, side: str) -> np.ndarray | None:
        """Latest measured joint positions of `side` (None if never received)."""
        ...


class CommandStream(Protocol):
    """What takeover/release need of `streamer.Streamer`."""

    hz: float
    fault: str | None

    def healthy(self) -> bool: ...

    def cycles_since_start(self, side: str) -> int: ...

    def last_command(self, side: str) -> np.ndarray: ...

    def hold(self, side: str | None = None) -> None: ...


class Activatable(Protocol):
    def activate(self, q_activation: np.ndarray, ref_cmd: np.ndarray) -> None: ...

    def deactivate(self) -> None: ...


class EventSink(Protocol):
    def event(self, kind: str, **fields) -> None: ...


# ---- helpers ----


def _list(side: str, arm: ArmConfig, cm: ControllerManager) -> dict[str, str]:
    try:
        return dict(cm.list_controllers())
    except Exception as e:  # timeouts and transport errors; KeyboardInterrupt propagates
        raise TakeoverError(
            f"{side}: list_controllers on {arm.controller_manager} failed: {type(e).__name__}: {e}"
        ) from e


def _state(side: str, arm: ArmConfig, cm: ControllerManager) -> str | None:
    return _list(side, arm, cm).get(arm.controller)


def _measured(side: str, arm: ArmConfig, graph: GraphView) -> np.ndarray:
    q = graph.measured(side)
    if q is None:
        raise TakeoverError(
            f"{side}: no measured joint states on {arm.measured_topic}; is the robot state broadcaster running?"
        )
    return np.asarray(q, dtype=np.float64)


def _require_healthy(side: str, streamer: CommandStream) -> None:
    if not streamer.healthy():
        raise TakeoverError(
            f"{side}: the command stream is unhealthy ({streamer.fault or 'no recent publish'}); not activating"
        )


def _check_tol(
    side: str, arm: ArmConfig, graph: GraphView, streamer: CommandStream, tol: float, when: str
) -> tuple[np.ndarray, np.ndarray]:
    """(measured, last command), both read now; raises if they differ by more than `tol` on any joint."""
    measured = _measured(side, arm, graph)
    cmd = np.asarray(streamer.last_command(side), dtype=np.float64)
    dev = float(np.max(np.abs(cmd - measured)))
    if not dev <= tol:  # NaN counts as exceeded
        raise TakeoverError(
            f"{side}: the streamed hold differs from the measured joints by {dev:.4f} rad {when} "
            f"(> takeover.tol_rad {tol}); the arm moved since the stream started. Let it settle and retry."
        )
    return measured, cmd


def _wait_discovery(graph: GraphView, tk: TakeoverConfig, sleep: Callable[[float], None]) -> None:
    """Sleep until the node has been up for the DDS discovery time and the foreign-message window."""
    wait = max(tk.discovery_wait_s, FOREIGN_WINDOW_S) - graph.uptime_s()
    if wait > 0:
        sleep(wait)


def _check_no_foreign_messages(side: str, arm: ArmConfig, graph: GraphView) -> None:
    topic = arm.command_topic
    foreign = graph.foreign_messages(topic, FOREIGN_WINDOW_S)
    if foreign:
        raise TakeoverError(
            f"{side}: {foreign} message(s) from another publisher on {topic} in the last {FOREIGN_WINDOW_S:.0f} s: "
            "is GELLO teleop still running? Stop it and retry."
        )


def _wait_warm(side: str, streamer: CommandStream, tk: TakeoverConfig, sleep: Callable[[float], None]) -> None:
    timeout = 1.0 + 2.0 * tk.warm_cycles / streamer.hz
    waited = 0.0
    while streamer.cycles_since_start(side) < tk.warm_cycles:
        if streamer.fault is not None:
            raise TakeoverError(f"{side}: the command stream faulted while warming up: {streamer.fault}")
        if waited >= timeout - _EPS:
            raise TakeoverError(
                f"{side}: only {streamer.cycles_since_start(side)}/{tk.warm_cycles} hold commands published "
                f"after {waited:.1f} s; the command stream is not running"
            )
        sleep(WARM_POLL_S)
        waited += WARM_POLL_S


# ---- takeover ----


def takeover(
    side: str,
    arm: ArmConfig,
    cm: ControllerManager,
    graph: GraphView,
    streamer: CommandStream,
    adapter: Activatable,
    tk: TakeoverConfig,
    *,
    confirm_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Take over `side` while `streamer` streams its hold (= measured). Returns True only if this call activated
    the controller (and so release must deactivate it). On any failure after switch_controller was called it
    deactivates again before re-raising."""
    topic = arm.command_topic
    # 1. measured stream present
    _measured(side, arm, graph)
    # 2. DDS discovery (and the foreign-message window) covered since the node started
    _wait_discovery(graph, tk, sleep)
    # 3. nobody else is commanding this arm (`Session` also checks this before it starts streaming at all)
    _check_no_foreign_messages(side, arm, graph)
    idle = graph.count_publishers(topic) - 1  # minus our own command publisher
    if idle > 0:
        logger.warning(
            "%s: %d other publisher(s) on %s, idle (no messages in the last %.0f s); continuing. "
            "If GELLO teleop is running, stop it.",
            side,
            idle,
            topic,
            FOREIGN_WINDOW_S,
        )
    # 4. the hold stream is warm and equal to measured
    _wait_warm(side, streamer, tk, sleep)
    _require_healthy(side, streamer)
    _check_tol(side, arm, graph, streamer, tk.tol_rad, "after warm-up")
    # 5. by controller state
    if tk.mode == "external":
        return _takeover_external(side, arm, cm, graph, streamer, adapter, tk, sleep)
    if tk.mode != "switch":
        raise ValueError(f"takeover.mode must be 'switch' or 'external', got {tk.mode!r}")
    states = _list(side, arm, cm)
    state = states.get(arm.controller)
    if state is None:
        available = ", ".join(f"{name} ({st})" for name, st in sorted(states.items())) or "none"
        raise TakeoverError(
            f"{side}: controller {arm.controller!r} is not loaded on {arm.controller_manager} (available: {available}). "
            f"Load and configure it with your tooling, fix arms.{side}.controller, or set takeover.mode: external."
        )
    if state == "active":
        logger.warning(
            "%s: %s is already active (not activated by us); streaming to it and leaving it active on release",
            side,
            arm.controller,
        )
        _require_healthy(side, streamer)
        measured, cmd = _check_tol(side, arm, graph, streamer, tk.tol_rad, "with the controller already active")
        adapter.activate(measured, cmd)
        return False
    if state != "inactive":
        raise TakeoverError(
            f"{side}: controller {arm.controller!r} is '{state}' on {arm.controller_manager}; it must be 'inactive' "
            "to be activated. Load and configure it with your tooling (e.g. ros2 control set_controller_state "
            f"{arm.controller} inactive -c {arm.controller_manager}), or set takeover.mode: external."
        )
    return _activate(side, arm, cm, graph, streamer, adapter, tk, confirm_s, sleep)


def _activate(side, arm, cm, graph, streamer, adapter, tk, confirm_s, sleep) -> bool:
    # The no-jump guarantee holds only if the stream's current output equals the arm: re-check right here, and
    # take q_activation / ref_cmd from this very reading.
    _require_healthy(side, streamer)
    measured, cmd = _check_tol(side, arm, graph, streamer, tk.tol_rad, "right before activation")
    switched = False
    try:
        adapter.activate(measured, cmd)
        switched = True  # from here on the switch may have taken effect even if the call fails
        try:
            ok = cm.switch([arm.controller], [], timeout_s=tk.service_timeout_s)
        except TakeoverError:
            raise
        except Exception as e:
            raise TakeoverError(
                f"{side}: switch_controller (activate {arm.controller}) on {arm.controller_manager} failed: "
                f"{type(e).__name__}: {e}"
            ) from e
        if not ok:
            raise TakeoverError(
                f"{side}: switch_controller refused to activate {arm.controller} on {arm.controller_manager}"
            )
        _confirm_active(side, arm, cm, graph, streamer, tk, confirm_s, sleep)
        return True
    except BaseException:
        if switched:
            logger.error("%s: activation failed after switch_controller; deactivating %s", side, arm.controller)
            for err in release(
                side, arm, cm, streamer, activated_by_us=True, sleep=sleep, timeout_s=tk.service_timeout_s
            ):
                logger.error("%s", err)
        raise


def _confirm_active(side, arm, cm, graph, streamer, tk, confirm_s, sleep) -> None:
    waited = 0.0
    while True:
        sleep(CONFIRM_POLL_S)  # also lets the controller act on the stream before the tolerance check
        waited += CONFIRM_POLL_S
        state = _state(side, arm, cm)
        if state == "active":
            _require_healthy(side, streamer)
            _check_tol(side, arm, graph, streamer, tk.tol_rad, "after activation")
            return
        if waited >= confirm_s - _EPS:
            raise TakeoverError(
                f"{side}: {arm.controller} not confirmed active within {confirm_s} s after switch_controller "
                f"(state: {state})"
            )


def _takeover_external(side, arm, cm, graph, streamer, adapter, tk, sleep) -> bool:
    state = _state(side, arm, cm)
    if state == "active":
        logger.warning("%s: %s is already active; streaming to it", side, arm.controller)
    else:
        print(
            f"[{side}] activate {arm.controller} on {arm.controller_manager} now "
            f"(the hold is streaming; waiting up to {EXTERNAL_WAIT_S:.0f} s, Ctrl-C aborts)",
            flush=True,
        )
        waited = 0.0
        while state != "active":
            if waited >= EXTERNAL_WAIT_S - _EPS:
                raise TakeoverError(
                    f"{side}: {arm.controller} not active after {EXTERNAL_WAIT_S:.0f} s (state: {state}); "
                    "activate it with your tooling and retry"
                )
            sleep(EXTERNAL_POLL_S)
            waited += EXTERNAL_POLL_S
            _require_healthy(side, streamer)
            _check_tol(side, arm, graph, streamer, tk.tol_rad, "while waiting for activation")
            state = _state(side, arm, cm)
    _require_healthy(side, streamer)
    measured, cmd = _check_tol(side, arm, graph, streamer, tk.tol_rad, "at activation")
    adapter.activate(measured, cmd)
    return False


# ---- release ----


def release(
    side: str,
    arm: ArmConfig,
    cm: ControllerManager,
    streamer: CommandStream,
    *,
    activated_by_us: bool,
    hold_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    timeout_s: float = 5.0,
) -> list[str]:
    """Hold `side`, wait `hold_s`, deactivate the controller only if we activated it. Never raises (a
    KeyboardInterrupt during the hold or the deactivation is recorded and the release continues); returns the
    errors. Publishing keeps going until the caller stops the streamer."""
    return _release(side, arm, cm, streamer, activated_by_us, hold_s, sleep, timeout_s)[0]


def _release(side, arm, cm, streamer, activated_by_us, hold_s, sleep, timeout_s) -> tuple[list[str], bool]:
    errors: list[str] = []
    interrupted = False
    try:
        streamer.hold(side)
        sleep(hold_s)
    except KeyboardInterrupt:
        interrupted = True
        errors.append(f"{side}: hold interrupted (KeyboardInterrupt); continuing the release")
    except Exception as e:
        errors.append(f"{side}: hold failed: {type(e).__name__}: {e}")
    if not activated_by_us:
        return errors, interrupted
    for attempt in range(2):  # one retry, only if the call itself was interrupted
        try:
            if not cm.switch([], [arm.controller], timeout_s=timeout_s):
                errors.append(
                    f"{side}: switch_controller refused to deactivate {arm.controller} on {arm.controller_manager}"
                )
            elif (state := cm.list_controllers().get(arm.controller)) == "active":
                errors.append(f"{side}: {arm.controller} still active after deactivation (state: {state})")
            break
        except KeyboardInterrupt:
            interrupted = True
            errors.append(
                f"{side}: deactivating {arm.controller} interrupted (KeyboardInterrupt)"
                + ("; retrying" if attempt == 0 else "")
            )
        except Exception as e:
            errors.append(
                f"{side}: deactivating {arm.controller} on {arm.controller_manager} failed: {type(e).__name__}: {e}"
            )
            break
    return errors, interrupted


# ---- session ----


def session_sides(cfg: SiteConfig, active_arms: Sequence[str] | None = None) -> tuple[str, ...]:
    """The arms a Session takes over: the active ones, plus the others when `takeover.hold_inactive_arms` (they
    only ever stream hold = their measured pose)."""
    active = tuple(cfg.active_arms if active_arms is None else active_arms)
    unknown = [side for side in active if side not in cfg.arms]
    if unknown or not active:
        raise ValueError(f"active arms must be a non-empty subset of {tuple(cfg.arms)}, got {active}")
    held = cfg.takeover.hold_inactive_arms
    return tuple(side for side in ARM_NAMES if side in active or (held and side in cfg.arms))


class Session:
    """Context manager: start the stream with hold = measured for `sides`, take every side over; on exit release
    exactly the sides it took over, stop the stream once and re-raise the original exception. Before it publishes
    anything it waits for DDS discovery and refuses if any side's command topic carries foreign messages.

    `cm` maps each side to its arm's controller manager. The adapters are deactivated (back to identity) only after
    the stream stopped, so a relative-mode command never jumps while a controller may still follow it. A
    `Session` is entered once; build a fresh streamer for each.
    """

    def __init__(
        self,
        cfg: SiteConfig,
        sides: Sequence[str],
        cm: Mapping[str, ControllerManager],
        graph: GraphView,
        streamer,
        adapters: Mapping[str, Activatable],
        *,
        recorder: EventSink | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.sides = tuple(sides)
        if not self.sides or len(set(self.sides)) != len(self.sides):
            raise ValueError(f"sides must be non-empty and unique, got {self.sides}")
        for name, mapping in (("arms", cfg.arms), ("cm", cm), ("adapters", adapters)):
            missing = [side for side in self.sides if side not in mapping]
            if missing:
                raise ValueError(f"{name}: no entry for {missing}")
        if set(self.sides) != set(streamer.sides):
            raise ValueError(f"the streamer streams {streamer.sides}, the session takes over {self.sides}")
        self.cfg = cfg
        self.cm = cm
        self.graph = graph
        self.streamer = streamer
        self.adapters = adapters
        self.recorder = recorder
        self.sleep = sleep
        self.activated_by_us: dict[str, bool] = {}  # sides taken over, in order -> whether we activated
        self._entered = False
        self._closed = False

    def __enter__(self) -> "Session":
        if self._entered:
            raise RuntimeError("a Session can be entered only once")
        self._entered = True
        tk = self.cfg.takeover
        try:
            # Never fight another publisher: check every command topic before publishing anything on any of them.
            _wait_discovery(self.graph, tk, self.sleep)
            for side in self.sides:
                _check_no_foreign_messages(side, self.cfg.arms[side], self.graph)
            hold = {side: self.graph.measured(side) for side in self.sides}
            missing = [side for side, q in hold.items() if q is None]
            if missing:
                topics = ", ".join(self.cfg.arms[side].measured_topic for side in missing)
                raise TakeoverError(
                    f"no measured joint states for {missing} ({topics}); not starting the command stream"
                )
            try:
                self.streamer.start(hold)
            except ValueError as e:
                raise TakeoverError(f"refusing to stream hold = measured: {e}") from e
            for side in self.sides:
                by_us = takeover(
                    side,
                    self.cfg.arms[side],
                    self.cm[side],
                    self.graph,
                    self.streamer,
                    self.adapters[side],
                    tk,
                    sleep=self.sleep,
                )
                self.activated_by_us[side] = by_us
                self._event("takeover", side=side, controller=self.cfg.arms[side].controller, activated_by_us=by_us)
            if not self.streamer.healthy():
                raise TakeoverError(
                    f"the command stream is unhealthy after takeover ({self.streamer.fault or 'no recent publish'})"
                )
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        interrupted = self._close()
        if exc_type is None and interrupted:
            raise KeyboardInterrupt("interrupted during release (the release was completed)")
        return False  # never swallow the original exception

    def _close(self) -> bool:
        """Release, stop the stream, reset the adapters. Never raises; returns whether a KeyboardInterrupt was
        swallowed on the way."""
        if self._closed:
            return False
        self._closed = True
        errors: list[str] = []
        interrupted = False
        timeout_s = self.cfg.takeover.service_timeout_s
        for side, by_us in self.activated_by_us.items():
            try:
                errs, intr = _release(
                    side, self.cfg.arms[side], self.cm[side], self.streamer, by_us, 0.5, self.sleep, timeout_s
                )
            except BaseException as e:  # _release does not raise; belt and braces so every side is attempted
                errs, intr = [f"{side}: release failed: {type(e).__name__}: {e}"], isinstance(e, KeyboardInterrupt)
            errors += errs
            interrupted |= intr
        try:
            self.streamer.stop()
        except BaseException as e:
            interrupted |= isinstance(e, KeyboardInterrupt)
            errors.append(f"stopping the command stream failed: {type(e).__name__}: {e}")
        for side in self.sides:
            try:
                self.adapters[side].deactivate()
            except BaseException as e:
                interrupted |= isinstance(e, KeyboardInterrupt)
                errors.append(f"{side}: resetting the command adapter failed: {type(e).__name__}: {e}")
        for err in errors:
            print(f"release: {err}", file=sys.stderr, flush=True)
        if errors:
            self._event("release_errors", errors=errors)
        self._event(
            "released",
            sides=list(self.activated_by_us),
            deactivated=[s for s, by_us in self.activated_by_us.items() if by_us],
        )
        return interrupted

    def _event(self, kind: str, **fields) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.event(kind, **fields)
        except Exception:
            logger.exception("recording event %r failed", kind)
