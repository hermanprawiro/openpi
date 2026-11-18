"""Controller takeover and release (spec §6.3, plan Task 12): when the joint impedance controller is switched on
and off, and in which order relative to the 100 Hz command stream.

Takeover per arm (`takeover`): measured joints present and fresh (`max_measured_age_s`) -> DDS discovery time
elapsed -> no foreign messages on the command topic -> the hold stream (= measured, seeded by `Session`) has published
`warm_cycles` commands and still equals measured within `tol_rad` -> by controller state: `inactive` -> re-check
tolerance, `adapter.activate`, `switch_controller` and confirm `active`; `active` already -> refused unless
`takeover.accept_active` (then only the adapter reference is captured; never switched by us); anything else ->
refuse. `takeover.mode: external` never calls switch_controller: the operator activates the controller with their
own tooling while we stream hold.

Release (`release`, used when an activation fails half-way): hold the arm, wait, deactivate. It never raises;
errors are collected. `Session` ties it together: wait for discovery, check every command topic for foreign messages
and every controller's state (an already-active one is refused) before publishing anything, start the stream with
hold = fresh measured, take over every side, and on exit (or any exception, including KeyboardInterrupt, also during
`__enter__`) release: hold every side, wait once, deactivate what we activated, and keep streaming the hold until
every controller reads not active (retrying our deactivation, or waiting for the operator's), then RELEASE_TAIL_S
more; stop the stream exactly once and re-raise the original exception. A controller that cannot be confirmed
inactive gets a loud CONTROLLER STILL ACTIVE message and the hold until the operator confirms (then
`ControllerStillActive`).

Pure logic against protocols; the ROS implementations come with the node (Task 16).
"""

from collections.abc import Callable, Mapping, Sequence
import contextlib
import logging
import sys
import time
from typing import Protocol
import uuid

import numpy as np

from robot_client.config import ARM_NAMES
from robot_client.config import ArmConfig
from robot_client.config import SiteConfig
from robot_client.config import TakeoverConfig

logger = logging.getLogger(__name__)

# header.frame_id of every command message this process publishes. Unique per process: anything else on a command
# topic, a second robot_client instance included, is foreign.
OUR_FRAME_ID = f"robot_client:{uuid.uuid4().hex[:8]}"
FOREIGN_WINDOW_S = 1.0  # foreign messages on the command topic are counted over this window
EXTERNAL_WAIT_S = 60.0  # takeover.mode external: how long to wait for the operator
WARM_POLL_S = 0.01
CONFIRM_POLL_S = 0.05
EXTERNAL_POLL_S = 0.1
# After the last deactivation the hold keeps streaming this long before the stream stops, so a controller never sees
# the stream end while it might still be active (the switch reply can race the next command otherwise).
RELEASE_TAIL_S = 0.1
RELEASE_HOLD_S = 0.5  # release: the hold streams this long before any deactivation
RELEASE_RETRY_S = 5.0  # release: a deactivation of ours that did not take is retried this long (then: loud)
RELEASE_POLL_S = 0.25  # release: period of the state polls / deactivation retries
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

    def measured_age(self, side: str) -> float:
        """Seconds since the latest measured sample of `side` arrived (inf if never)."""
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


def _say(*args, **kwargs) -> None:
    """print() for the release path: a closed terminal (SIGHUP) must never cut a release short."""
    with contextlib.suppress(OSError, ValueError):
        print(*args, **kwargs)


def _list(side: str, arm: ArmConfig, cm: ControllerManager) -> dict[str, str]:
    try:
        return dict(cm.list_controllers())
    except Exception as e:  # timeouts and transport errors; KeyboardInterrupt propagates
        raise TakeoverError(
            f"{side}: list_controllers on {arm.controller_manager} failed: {type(e).__name__}: {e}"
        ) from e


def _state(side: str, arm: ArmConfig, cm: ControllerManager) -> str | None:
    return _list(side, arm, cm).get(arm.controller)


def _measured(side: str, arm: ArmConfig, graph: GraphView, tk: TakeoverConfig) -> np.ndarray:
    """The latest measured joints, refused if missing or older than `takeover.max_measured_age_s` (a dead stream's
    last sample would make every tolerance check compare a stale pose with itself)."""
    q = graph.measured(side)
    if q is None:
        raise TakeoverError(
            f"{side}: no measured joint states on {arm.measured_topic}; is the robot state broadcaster running?"
        )
    age = float(graph.measured_age(side))
    if not age <= tk.max_measured_age_s:  # NaN counts as stale
        raise TakeoverError(
            f"{side}: the measured joint states on {arm.measured_topic} are {age:.2f} s old "
            f"(> takeover.max_measured_age_s {tk.max_measured_age_s}); is the robot state broadcaster still "
            "running? Not taking over."
        )
    return np.asarray(q, dtype=np.float64)


def _already_active_message(side: str, arm: ArmConfig) -> str:
    return (
        f"{side}: controller {arm.controller} on {arm.controller_manager} is already active; run "
        f"`release --arm {side}` (or deactivate it with your tooling), then retry. The client never streams into a "
        "controller that was active before it started (set takeover.accept_active: true only if you know it follows "
        "absolute targets)"
    )


def _not_loaded_message(side: str, arm: ArmConfig, states: Mapping[str, str]) -> str:
    available = ", ".join(f"{name} ({st})" for name, st in sorted(states.items())) or "none"
    return (
        f"{side}: controller {arm.controller!r} is not loaded on {arm.controller_manager} (available: {available}). "
        f"Load and configure it with your tooling, fix arms.{side}.controller, or set takeover.mode: external."
    )


def _wrong_state_message(side: str, arm: ArmConfig, state: str) -> str:
    return (
        f"{side}: controller {arm.controller!r} is '{state}' on {arm.controller_manager}; it must be 'inactive' "
        "to be activated. Load and configure it with your tooling (e.g. ros2 control set_controller_state "
        f"{arm.controller} inactive -c {arm.controller_manager}), or set takeover.mode: external."
    )


def _check_state_before_streaming(side: str, arm: ArmConfig, cm: ControllerManager, tk: TakeoverConfig) -> None:
    """Before anything is published: an already-active controller would act on our first message (unless
    `takeover.accept_active`); in `switch` mode the controller must also be loaded and `inactive`."""
    states = _list(side, arm, cm)
    state = states.get(arm.controller)
    if state == "active":
        if not tk.accept_active:
            raise TakeoverError(_already_active_message(side, arm))
        logger.warning("%s: %s is already active; takeover.accept_active: streaming into it", side, arm.controller)
        return
    if tk.mode == "switch":
        if state is None:
            raise TakeoverError(_not_loaded_message(side, arm, states))
        if state != "inactive":
            raise TakeoverError(_wrong_state_message(side, arm, state))


def _require_healthy(side: str, streamer: CommandStream) -> None:
    if not streamer.healthy():
        raise TakeoverError(
            f"{side}: the command stream is unhealthy ({streamer.fault or 'no recent publish'}); not activating"
        )


def _check_tol(
    side: str, arm: ArmConfig, graph: GraphView, streamer: CommandStream, tk: TakeoverConfig, when: str
) -> tuple[np.ndarray, np.ndarray]:
    """(measured, last command), both read now; raises if measured is stale or they differ by more than
    `tk.tol_rad` on any joint."""
    measured = _measured(side, arm, graph, tk)
    cmd = np.asarray(streamer.last_command(side), dtype=np.float64)
    dev = float(np.max(np.abs(cmd - measured)))
    if not dev <= tk.tol_rad:  # NaN counts as exceeded
        raise TakeoverError(
            f"{side}: the streamed hold differs from the measured joints by {dev:.4f} rad {when} "
            f"(> takeover.tol_rad {tk.tol_rad}); the arm moved since the stream started. Let it settle and retry."
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
    on_switch: Callable[[], None] | None = None,
) -> bool:
    """Take over `side` while `streamer` streams its hold (= measured). Returns True only if this call activated
    the controller (and so release must deactivate it). `on_switch` is called right before switch_controller
    (activate) is: from then on the caller must treat the controller as activated by us, even if an interrupt
    arrives before this function returns. On any failure after switch_controller was called it deactivates again
    before re-raising."""
    topic = arm.command_topic
    # 1. measured stream present and fresh
    _measured(side, arm, graph, tk)
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
    _check_tol(side, arm, graph, streamer, tk, "after warm-up")
    # 5. by controller state
    if tk.mode == "external":
        return _takeover_external(side, arm, cm, graph, streamer, adapter, tk, sleep)
    if tk.mode != "switch":
        raise ValueError(f"takeover.mode must be 'switch' or 'external', got {tk.mode!r}")
    states = _list(side, arm, cm)
    state = states.get(arm.controller)
    if state is None:
        raise TakeoverError(_not_loaded_message(side, arm, states))
    if state == "active":
        if not tk.accept_active:
            raise TakeoverError(_already_active_message(side, arm))
        logger.warning(
            "%s: %s is already active (not activated by us; takeover.accept_active); streaming to it",
            side,
            arm.controller,
        )
        _require_healthy(side, streamer)
        measured, cmd = _check_tol(side, arm, graph, streamer, tk, "with the controller already active")
        adapter.activate(measured, cmd)
        return False
    if state != "inactive":
        raise TakeoverError(_wrong_state_message(side, arm, state))
    return _activate(side, arm, cm, graph, streamer, adapter, tk, confirm_s, sleep, on_switch)


def _activate(side, arm, cm, graph, streamer, adapter, tk, confirm_s, sleep, on_switch=None) -> bool:
    # The no-jump guarantee holds only if the stream's current output equals the arm: re-check right here, and
    # take q_activation / ref_cmd from this very reading.
    _require_healthy(side, streamer)
    measured, cmd = _check_tol(side, arm, graph, streamer, tk, "right before activation")
    switched = False
    try:
        adapter.activate(measured, cmd)
        switched = True  # from here on the switch may have taken effect even if the call fails
        if on_switch is not None:
            on_switch()
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
            _check_tol(side, arm, graph, streamer, tk, "after activation")
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
            _check_tol(side, arm, graph, streamer, tk, "while waiting for activation")
            state = _state(side, arm, cm)
    _require_healthy(side, streamer)
    measured, cmd = _check_tol(side, arm, graph, streamer, tk, "at activation")
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


class ControllerStillActive(RuntimeError):  # noqa: N818 - an operator-facing outcome
    """The release could not confirm that every controller the session streamed to is inactive (the operator
    confirmed with Enter that they handled it). Raised from the Session in place of (and chained to) any original
    exception: it is the more urgent message."""


class Session:
    """Context manager: start the stream with hold = measured for `sides`, take every side over; on exit release
    and stop the stream once, then re-raise the original exception. Before it publishes anything it waits for DDS
    discovery, refuses if any side's command topic carries foreign messages, refuses an already-active controller
    (unless `takeover.accept_active`) and refuses missing or stale measured joints.

    Release: hold every side, wait once, deactivate what we activated (or may have: a switch is recorded as ours
    right before it is called) and read every other side's state. The stream keeps the hold going until every
    controller reads not active: our own deactivation is retried for RELEASE_RETRY_S; a controller we did not
    activate (external mode, accept_active) is the operator's to deactivate, for up to EXTERNAL_WAIT_S. After that a
    loud CONTROLLER STILL ACTIVE message, and the hold goes on until the state reads inactive or `ack()` (the
    operator's Enter) returns True; then `still_active` names the sides and ControllerStillActive is raised.

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
        clock: Callable[[], float] = time.monotonic,
        ack: Callable[[], bool] | None = None,
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
        self.clock = clock
        self.ack = ack
        self.activated_by_us: dict[str, bool] = {}  # sides taken over, in order -> whether we activated
        self.still_active: list[str] = []  # after the release: sides whose controller may still be active
        self._streaming = False
        self._entered = False
        self._closed = False
        self._close_done = False

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
            # Never stream into a controller that is already active: it would act on our very first message.
            for side in self.sides:
                _check_state_before_streaming(side, self.cfg.arms[side], self.cm[side], tk)
            missing = [side for side in self.sides if self.graph.measured(side) is None]
            if missing:
                topics = ", ".join(self.cfg.arms[side].measured_topic for side in missing)
                raise TakeoverError(
                    f"no measured joint states for {missing} ({topics}); not starting the command stream"
                )
            hold = {side: _measured(side, self.cfg.arms[side], self.graph, tk) for side in self.sides}
            try:
                self.streamer.start(hold)
            except ValueError as e:
                raise TakeoverError(f"refusing to stream hold = measured: {e}") from e
            self._streaming = True
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
                    on_switch=lambda side=side: self._mark_ours(side),
                )
                self.activated_by_us[side] = by_us
                self._event("takeover", side=side, controller=self.cfg.arms[side].controller, activated_by_us=by_us)
            if not self.streamer.healthy():
                raise TakeoverError(
                    f"the command stream is unhealthy after takeover ({self.streamer.fault or 'no recent publish'})"
                )
        except BaseException as e:
            self._close_guarded()
            if self.still_active:
                raise ControllerStillActive(self._still_active_message()) from e
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        interrupted = self._close_guarded()
        if self.still_active:
            raise ControllerStillActive(self._still_active_message()) from exc
        if exc_type is None and interrupted:
            raise KeyboardInterrupt("interrupted during release (the release was completed)")
        return False  # never swallow the original exception

    def _mark_ours(self, side: str) -> None:
        """Called right before switch_controller(activate): from here on the release treats `side` as ours."""
        self.activated_by_us[side] = True

    def _close_guarded(self) -> bool:
        """`_close`, re-run if an interrupt cuts it short before it really started (e.g. a Ctrl-C landing in
        `__exit__` before a subclass entered its signal deferral): the release never gets skipped."""
        interrupted = False
        while True:
            try:
                return self._close() or interrupted
            except KeyboardInterrupt:
                interrupted = True
                if self._close_done:
                    return True
                self._closed = False  # every release step is safe to repeat

    def _close(self) -> bool:
        """Release, stop the stream, reset the adapters. Never raises; returns whether a KeyboardInterrupt was
        swallowed on the way."""
        if self._closed:
            return False
        self._closed = True
        rel = _Release(self)
        if self._streaming:
            rel.run()
        try:
            self.streamer.stop()
        except BaseException as e:
            rel.interrupted |= isinstance(e, KeyboardInterrupt)
            rel.error(f"stopping the command stream failed: {type(e).__name__}: {e}")
        for side in self.sides:
            try:
                self.adapters[side].deactivate()
            except BaseException as e:
                rel.interrupted |= isinstance(e, KeyboardInterrupt)
                rel.error(f"{side}: resetting the command adapter failed: {type(e).__name__}: {e}")
        self.still_active = list(rel.still_active)
        for err in rel.errors:
            _say(f"release: {err}", file=sys.stderr, flush=True)
        if rel.errors:
            self._event("release_errors", errors=list(rel.errors))
        self._event(
            "released",
            sides=list(self.activated_by_us),
            deactivated=[s for s, by_us in self.activated_by_us.items() if by_us],
            still_active=list(self.still_active),
        )
        self._close_done = True
        return rel.interrupted

    def _still_active_message(self) -> str:
        where = ", ".join(
            f"{self.cfg.arms[side].controller} on {self.cfg.arms[side].controller_manager}"
            for side in self.still_active
        )
        return (
            f"{where} may still be active after the release (not confirmed inactive); the command stream has stopped. "
            "Deactivate it with your tooling (or `release --arm <side>`) or press the E-stop"
        )

    def _event(self, kind: str, **fields) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.event(kind, **fields)
        except Exception:
            logger.exception("recording event %r failed", kind)


class _Release:
    """One Session release while the stream runs (see `Session`). Never raises."""

    def __init__(self, session: Session):
        self.s = session
        self.errors: dict[str, None] = {}  # ordered, deduplicated (retries repeat the same error)
        self.interrupted = False
        self.still_active: list[str] = []

    def error(self, message: str) -> None:
        self.errors[message] = None

    def _guarded(self, fn, *args) -> bool:
        """Run `fn(*args)`; False if it raised (recorded; a KeyboardInterrupt is swallowed and remembered)."""
        try:
            fn(*args)
            return True
        except KeyboardInterrupt:
            self.interrupted = True
            return False
        except Exception as e:
            self.error(f"{getattr(fn, '__name__', fn)} failed: {type(e).__name__}: {e}")
            return False

    def _state(self, side: str) -> str | None:
        """The controller state; raises on failure."""
        arm = self.s.cfg.arms[side]
        return self.s.cm[side].list_controllers().get(arm.controller)

    def _settle(self, side: str, *, switch: bool) -> tuple[bool, str | None]:
        """(confirmed not active, last state read). Reads the state first (a retry after an interrupted
        deactivation must not report a failure that did not happen); `switch`: deactivate it if it is active."""
        arm = self.s.cfg.arms[side]
        where = f"{arm.controller} on {arm.controller_manager}"
        try:
            state = self._state(side)
            if state != "active":
                return True, state
            if not switch:
                return False, state
            ok = self.s.cm[side].switch([], [arm.controller], timeout_s=self.s.cfg.takeover.service_timeout_s)
            state = self._state(side)
            if state != "active":
                return True, state
            self.error(
                f"{side}: switch_controller refused to deactivate {where}"
                if not ok
                else f"{side}: {arm.controller} still active after deactivation (state: {state})"
            )
            return False, state
        except KeyboardInterrupt:
            self.interrupted = True
            self.error(f"{side}: deactivating {where} interrupted (KeyboardInterrupt); retrying")
            return False, None
        except Exception as e:
            self.error(f"{side}: deactivating {where} failed: {type(e).__name__}: {e}")
            return False, None

    def _sleep(self, dt: float) -> None:
        try:
            self.s.sleep(dt)
        except KeyboardInterrupt:
            self.interrupted = True

    def run(self) -> None:
        s = self.s
        # 1. every side holds; a single wait (skipped if the stream is not running: then deactivate at once)
        self._guarded(s.streamer.hold)
        healthy = False
        with contextlib.suppress(Exception):  # diagnostics only
            healthy = s.streamer.healthy()
        if healthy:
            self._sleep(RELEASE_HOLD_S)
        # 2. deactivate what we (may have) activated, read every other side's state
        t0 = s.clock()
        pending: dict[str, tuple[bool, float]] = {}  # side -> (we deactivate it, give-up time)
        seen_active = False
        for side in s.sides:
            ours = s.activated_by_us.get(side, False)
            done, state = self._settle(side, switch=ours)
            seen_active |= state == "active" or ours
            if done:
                continue
            if ours or state != "active":  # ours, or the state could not be read: retry for RELEASE_RETRY_S
                pending[side] = (ours, t0 + RELEASE_RETRY_S)
            else:  # active, not activated by us: the operator deactivates it
                arm = s.cfg.arms[side]
                _say(
                    f"[{side}] deactivate {arm.controller} on {arm.controller_manager} now (the client keeps "
                    f"streaming the hold until it reads inactive; waiting up to {EXTERNAL_WAIT_S:.0f} s)",
                    flush=True,
                )
                pending[side] = (False, t0 + EXTERNAL_WAIT_S)
        # 3. keep holding until every pending controller reads not active
        loud: set[str] = set()
        while pending:
            self._sleep(RELEASE_POLL_S)
            for side, (ours, give_up) in list(pending.items()):
                done, _ = self._settle(side, switch=ours and side not in loud)
                if done:
                    del pending[side]
                elif side not in loud and s.clock() >= give_up - _EPS:
                    loud.add(side)
                    self._announce_still_active(side)
            if pending and loud and self._acked():
                for side in list(pending):  # the operator says it is handled: one last look
                    if self._settle(side, switch=False)[0]:
                        del pending[side]
                self.still_active = [side for side in s.sides if side in pending]
                break
        # 4. keep streaming a little after the last deactivation (a controller never sees the stream end first)
        if seen_active:
            self._sleep(RELEASE_TAIL_S)

    def _acked(self) -> bool:
        if self.s.ack is None:
            return False
        try:
            return bool(self.s.ack())
        except KeyboardInterrupt:
            self.interrupted = True
            return False
        except Exception:
            logger.exception("reading the operator's confirmation failed")
            return False

    def _announce_still_active(self, side: str) -> None:
        arm = self.s.cfg.arms[side]
        bar = "!" * 100
        _say(
            f"{bar}\n! CONTROLLER STILL ACTIVE on {arm.controller_manager} ({arm.controller}) — deactivate it now "
            "or press the E-stop.\n! The client keeps streaming the hold until it reads inactive. Press Enter once "
            f"you have handled it.\n{bar}",
            file=sys.stderr,
            flush=True,
        )
