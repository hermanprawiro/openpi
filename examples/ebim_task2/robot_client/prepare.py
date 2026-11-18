"""Move the robot to the episode start (spec §6.3 `prepare`): impedance controllers off -> PTP left, then right, to
`start_pose` -> spine to `spine.height` -> grippers open.

Runs only while no command stream publishes and only after an operator Enter (CLI). Each step's result goes into
the report; the first failure stops the sequence. KeyboardInterrupt propagates (the motion backend cancels its
goal). Pure logic against protocols; the ROS backend comes with the node (Task 16).
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import time
from typing import Protocol

from robot_client.config import ARM_NAMES
from robot_client.config import SiteConfig
from robot_client.takeover import ControllerManager

GRIPPER_OPEN = 1.0
GRIPPER_REPUBLISH_HZ = 10.0
GRIPPER_REPUBLISH_S = 1.0  # a single message right after the publisher is created can be lost
EXTERNAL_WAIT_S = 60.0
EXTERNAL_POLL_S = 0.1
_EPS = 1e-9


class MotionBackend(Protocol):
    def ptp(
        self, side: str, joints: Sequence[float], max_vel: Sequence[float], tol: float, timeout_s: float
    ) -> bool: ...

    def spine(self, height: float, vel: float, acc: float, dec: float, timeout_s: float) -> bool: ...

    def spine_position(self, timeout_s: float) -> float | None: ...

    def gripper(self, side: str, width: float) -> None: ...


@dataclasses.dataclass(frozen=True)
class StepResult:
    step: str
    ok: bool
    detail: str = ""


@dataclasses.dataclass
class PrepareReport:
    steps: list[StepResult] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    @property
    def failed_step(self) -> str | None:
        return next((s.step for s in self.steps if not s.ok), None)

    def summary(self) -> str:
        lines = [f"  {'ok  ' if s.ok else 'FAIL'} {s.step}: {s.detail}" for s in self.steps]
        verdict = "prepare: done" if self.ok else f"prepare: stopped at '{self.failed_step}'"
        return "\n".join([verdict, *lines])


class _StepError(Exception):
    pass


def prepare(
    cfg: SiteConfig,
    cm: Mapping[str, ControllerManager],
    backend: MotionBackend,
    streamer_active: bool,  # noqa: FBT001 - the planned interface
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> PrepareReport:
    """`cm` maps each arm to its controller manager."""
    report = PrepareReport()
    if streamer_active:
        report.steps.append(
            StepResult(
                "check command stream",
                ok=False,
                detail="the joint command stream is running; prepare refuses to move the arms while it publishes",
            )
        )
        return report
    sides = [side for side in ARM_NAMES if side in cfg.arms]
    steps: list[tuple[str, Callable[[], str]]] = []
    steps += [(f"deactivate {side}", lambda side=side: _deactivate(cfg, side, cm[side], sleep)) for side in sides]
    steps += [(f"ptp {side}", lambda side=side: _ptp(cfg, side, backend)) for side in sides]
    steps += [("spine", lambda: _spine(cfg, backend)), ("grippers", lambda: _grippers(sides, backend, sleep))]
    for name, run in steps:
        try:
            result = StepResult(name, ok=True, detail=run())
        except _StepError as e:
            result = StepResult(name, ok=False, detail=str(e))
        except Exception as e:
            result = StepResult(name, ok=False, detail=f"{type(e).__name__}: {e}")
        report.steps.append(result)
        if not result.ok:
            break
    return report


def _deactivate(cfg: SiteConfig, side: str, cm: ControllerManager, sleep: Callable[[float], None]) -> str:
    arm = cfg.arms[side]
    state = cm.list_controllers().get(arm.controller)
    if state != "active":
        return f"{arm.controller} not active ({state or 'not loaded'})"
    if cfg.takeover.mode == "external":
        print(
            f"[{side}] deactivate {arm.controller} on {arm.controller_manager} now "
            f"(prepare moves the arms with PTP; waiting up to {EXTERNAL_WAIT_S:.0f} s, Ctrl-C aborts)",
            flush=True,
        )
        waited = 0.0
        while state == "active":
            if waited >= EXTERNAL_WAIT_S - _EPS:
                raise _StepError(
                    f"{arm.controller} still active after {EXTERNAL_WAIT_S:.0f} s; deactivate it and retry"
                )
            sleep(EXTERNAL_POLL_S)
            waited += EXTERNAL_POLL_S
            state = cm.list_controllers().get(arm.controller)
        return f"{arm.controller} deactivated by the operator ({state or 'not loaded'})"
    if not cm.switch([], [arm.controller], timeout_s=cfg.takeover.service_timeout_s):
        raise _StepError(f"switch_controller refused to deactivate {arm.controller} on {arm.controller_manager}")
    state = cm.list_controllers().get(arm.controller)
    if state == "active":
        raise _StepError(f"{arm.controller} still active after switch_controller")
    return f"{arm.controller} deactivated ({state or 'not loaded'})"


def _ptp(cfg: SiteConfig, side: str, backend: MotionBackend) -> str:
    arm, p = cfg.arms[side], cfg.prepare
    ok = backend.ptp(
        side, arm.start_pose, max_vel=p.ptp_max_joint_velocities, tol=p.ptp_goal_tolerance, timeout_s=p.timeout_s
    )
    if not ok:
        raise _StepError(f"PTP to start_pose failed or timed out ({arm.ptp_action}, timeout {p.timeout_s} s)")
    return f"at start_pose {list(arm.start_pose)}"


def _spine(cfg: SiteConfig, backend: MotionBackend) -> str:
    sp = cfg.spine
    if not sp.enabled:
        return "skipped (spine.enabled: false)"
    current = backend.spine_position(timeout_s=cfg.takeover.service_timeout_s)
    if current is None:
        raise _StepError(
            f"spine position unavailable ({sp.get_position_service}); not moving the spine. "
            "Check the spine node, or set spine.enabled: false"
        )
    delta = abs(sp.height - current)
    if not delta <= sp.max_delta_m:
        raise _StepError(
            f"spine.height {sp.height} m is {delta:.3f} m from the current {current:.3f} m "
            f"(> spine.max_delta_m {sp.max_delta_m}); check the height and its unit (metres)"
        )
    if not backend.spine(sp.height, sp.velocity, sp.acceleration, sp.deceleration, timeout_s=cfg.prepare.timeout_s):
        raise _StepError(
            f"spine move to {sp.height} m failed or timed out ({sp.action}, timeout {cfg.prepare.timeout_s} s)"
        )
    return f"moved from {current:.3f} m to {sp.height} m"


def _grippers(sides: Sequence[str], backend: MotionBackend, sleep: Callable[[float], None]) -> str:
    n = round(GRIPPER_REPUBLISH_S * GRIPPER_REPUBLISH_HZ)
    for _ in range(n):
        for side in sides:
            backend.gripper(side, GRIPPER_OPEN)
        sleep(1.0 / GRIPPER_REPUBLISH_HZ)
    return f"width {GRIPPER_OPEN} sent {n} times at {GRIPPER_REPUBLISH_HZ:.0f} Hz to {', '.join(sides)}"
