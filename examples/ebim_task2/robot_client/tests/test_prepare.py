from pathlib import Path

from robot_client.config import load_site_config
from robot_client.prepare import prepare

MUNICH_YAML = Path(__file__).resolve().parents[1] / "robot_site.munich.yaml"
CTRL = "joint_impedance_controller"


class FakeCM:
    def __init__(self, side, log, state="inactive"):
        self.side = side
        self.log = log
        self.states = {CTRL: state}
        self.operator_deactivates_on_list = None
        self.lists = 0

    def list_controllers(self):
        self.lists += 1
        self.log.append(("list", self.side))
        if self.operator_deactivates_on_list is not None and self.lists >= self.operator_deactivates_on_list:
            self.states[CTRL] = "inactive"
        return dict(self.states)

    def switch(self, activate, deactivate, timeout_s):
        self.log.append(("switch", self.side, tuple(activate), tuple(deactivate), timeout_s))
        for c in deactivate:
            self.states[c] = "inactive"
        return True


class FakeBackend:
    def __init__(self, log):
        self.log = log
        self.ptp_ok = {"left": True, "right": True}
        self.position = 0.40
        self.spine_ok = True

    def ptp(self, side, joints, max_vel, tol, timeout_s):
        self.log.append(("ptp", side, tuple(joints), tuple(max_vel), tol, timeout_s))
        return self.ptp_ok[side]

    def spine_position(self, timeout_s):
        self.log.append(("spine_position", timeout_s))
        return self.position

    def spine(self, height, vel, acc, dec, timeout_s):
        self.log.append(("spine", height, vel, acc, dec, timeout_s))
        return self.spine_ok

    def gripper(self, side, width):
        self.log.append(("gripper", side, width))


class Rig:
    def __init__(self, overrides=(), states=("inactive", "inactive")):
        self.cfg = load_site_config(MUNICH_YAML, list(overrides), {})
        self.log = []
        self.cms = {side: FakeCM(side, self.log, state) for side, state in zip(("left", "right"), states, strict=True)}
        self.backend = FakeBackend(self.log)
        self.sleeps = []

    def sleep(self, dt):
        self.sleeps.append(dt)
        self.log.append(("sleep", dt))

    def run(self, *, streamer_active=False):
        return prepare(self.cfg, self.cms, self.backend, streamer_active, sleep=self.sleep)

    def kinds(self):
        return [e[0] for e in self.log]


def test_prepare_sequence_and_stop_on_failure():
    rig = Rig(states=("active", "inactive"))
    report = rig.run()
    assert report.ok, report.summary()
    assert report.failed_step is None
    assert [s.step for s in report.steps] == [
        "deactivate left",
        "deactivate right",
        "ptp left",
        "ptp right",
        "spine",
        "grippers",
    ]
    cfg = rig.cfg
    motion = [e for e in rig.log if e[0] not in ("list", "sleep", "gripper")]
    assert motion == [
        ("switch", "left", (), (CTRL,), cfg.takeover.service_timeout_s),  # only the active one is deactivated
        (
            "ptp",
            "left",
            cfg.arms["left"].start_pose,
            cfg.prepare.ptp_max_joint_velocities,
            cfg.prepare.ptp_goal_tolerance,
            cfg.prepare.timeout_s,
        ),
        (
            "ptp",
            "right",
            cfg.arms["right"].start_pose,
            cfg.prepare.ptp_max_joint_velocities,
            cfg.prepare.ptp_goal_tolerance,
            cfg.prepare.timeout_s,
        ),
        ("spine_position", cfg.takeover.service_timeout_s),
        ("spine", 0.434, 0.1, 0.2, 0.2, cfg.prepare.timeout_s),
    ]
    # grippers fully open, republished at 10 Hz for 1 s on both arms
    grips = [e for e in rig.log if e[0] == "gripper"]
    assert grips == [("gripper", side, 1.0) for _ in range(10) for side in ("left", "right")]
    assert rig.sleeps == [0.1] * 10

    # PTP right fails -> spine and grippers never run; the report names the step
    rig = Rig()
    rig.backend.ptp_ok["right"] = False
    report = rig.run()
    assert not report.ok
    assert report.failed_step == "ptp right"
    assert [s.step for s in report.steps] == ["deactivate left", "deactivate right", "ptp left", "ptp right"]
    assert "spine_position" not in rig.kinds()
    assert "spine" not in rig.kinds()
    assert "gripper" not in rig.kinds()
    assert "ptp right" in report.summary()

    # an exception (e.g. an action timeout) is a failed step too
    rig = Rig()

    def boom(*args, **kwargs):
        raise TimeoutError("ptp goal not accepted within 60 s")

    rig.backend.ptp = boom
    report = rig.run()
    assert report.failed_step == "ptp left"
    assert "not accepted" in report.steps[-1].detail


def test_prepare_refuses_while_streaming():
    rig = Rig(states=("active", "active"))
    report = rig.run(streamer_active=True)
    assert not report.ok
    assert report.failed_step is not None
    assert "stream" in report.steps[0].detail
    assert rig.log == []


def test_spine_refused_on_large_delta_or_no_position():
    for position, match in ((None, "unavailable"), (0.1, "max_delta_m")):
        rig = Rig()
        rig.backend.position = position
        report = rig.run()
        assert report.failed_step == "spine"
        assert match in report.steps[-1].detail
        assert "spine" not in rig.kinds()
        assert "gripper" not in rig.kinds()

    # within max_delta_m but spine disabled: not moved (and not read), grippers still opened
    rig = Rig(["spine.enabled=false"])
    report = rig.run()
    assert report.ok
    assert "spine" not in rig.kinds()
    assert "spine_position" not in rig.kinds()
    assert "gripper" in rig.kinds()

    # the move itself failing
    rig = Rig()
    rig.backend.spine_ok = False
    assert rig.run().failed_step == "spine"


def test_external_mode_asks_operator_to_deactivate(capsys):
    rig = Rig(["takeover.mode=external"], states=("active", "inactive"))
    rig.cms["left"].operator_deactivates_on_list = 3
    report = rig.run()
    assert report.ok, report.summary()
    assert "switch" not in rig.kinds()
    assert f"deactivate {CTRL} on /left/controller_manager now" in capsys.readouterr().out

    # the operator never does: gives up after 60 s, no PTP
    rig = Rig(["takeover.mode=external"], states=("active", "inactive"))
    report = rig.run()
    assert report.failed_step == "deactivate left"
    assert sum(rig.sleeps) >= 60.0
    assert "ptp" not in rig.kinds()
    assert "switch" not in rig.kinds()


def test_deactivation_not_confirmed_stops_before_ptp():
    rig = Rig(states=("active", "inactive"))
    rig.cms["left"].switch = lambda activate, deactivate, timeout_s: True  # claims success, changes nothing
    report = rig.run()
    assert report.failed_step == "deactivate left"
    assert "ptp" not in rig.kinds()
