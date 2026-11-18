"""check_obs / check_infer / check_motion against fakes and the stub policy server (no ROS)."""

import json
from pathlib import Path

import numpy as np
import pytest
from robot_client import checks
from robot_client.config import load_site_config
from robot_client.preprocess import Snapshot
from robot_client.takeover import TakeoverError

MUNICH_YAML = Path(__file__).resolve().parents[1] / "robot_site.munich.yaml"
JOINTS = {side: [f"{side}_fr3v2_joint{i}" for i in range(1, 8)] for side in ("left", "right")}
CAMERA_RAW = {
    "head": {"height": 720, "width": 1280, "encoding": "bgra8", "step": 5120},
    "wrist_left": {"height": 480, "width": 640, "encoding": "rgb8", "step": 1920},
    "wrist_right": {"height": 480, "width": 640, "encoding": "rgb8", "step": 1920},
}
CAMERA_HZ = {"head": 20.0, "wrist_left": 30.0, "wrist_right": 30.0}


def munich(*overrides, env=None):
    return load_site_config(MUNICH_YAML, list(overrides), env or {})


def stream(topic, *, count=100, rate=100.0, age=0.01, raw=None, decode_errors=0, last_error=None):
    return {
        "topic": topic,
        "count": count,
        "rate_hz": rate if count > 1 else None,
        "max_gap_s": 0.02,
        "age_s": age,
        "raw": dict(raw or {}),
        "decode_errors": decode_errors,
        "last_error": last_error,
    }


class FakeObsNode:
    """What check_obs reads of a RobotNode: stream_stats() and snapshot()."""

    def __init__(self, cfg):
        self.stats = {}
        self.images, self.arm_q, self.knuckle, self.ages = {}, {}, {}, {}
        for name, cam in cfg.cameras.items():
            self.stats[f"cam/{name}"] = stream(cam.topic, rate=CAMERA_HZ[name], raw=CAMERA_RAW[name])
            h, w = CAMERA_RAW[name]["height"], CAMERA_RAW[name]["width"]
            self.images[name] = np.full((h, w, 3), 80, dtype=np.uint8)
            self.ages[f"cam/{name}"] = 0.01
        for side, arm in cfg.arms.items():
            self.stats[f"arm/{side}"] = stream(arm.measured_topic, rate=200.0, raw={"names": JOINTS[side]})
            self.arm_q[side] = np.array(arm.start_pose)
            self.ages[f"arm/{side}"] = 0.01
            self.stats[f"grip/{side}"] = stream(arm.gripper.state_topic, rate=50.0, raw={"names": [arm.gripper.joint]})
            self.knuckle[side] = 0.0 if side == "left" else 0.12
            self.ages[f"grip/{side}"] = 0.01
        self.stats["commands"] = {}

    def drop(self, key):
        """The stream never delivered anything."""
        topic = self.stats[key]["topic"]
        self.stats[key] = stream(topic, count=0, age=None)
        kind, name = key.split("/")
        {"cam": self.images, "arm": self.arm_q, "grip": self.knuckle}[kind].pop(name, None)
        self.ages[key] = float("inf")

    def stream_stats(self):
        return json.loads(json.dumps(self.stats))

    def snapshot(self):
        return Snapshot(
            images={k: v.copy() for k, v in self.images.items()},
            arm_q={k: v.copy() for k, v in self.arm_q.items()},
            knuckle=dict(self.knuckle),
            ages=dict(self.ages),
            t_mono=0.0,
        )


class FakeSpine:
    def __init__(self, position):
        self.position = position

    def spine_position(self, timeout_s):
        return self.position


def no_sleep(_dt):
    pass


def rows(report, status):
    return [line for line in report.lines if line.startswith(status)]


# ---- check_obs ----


def test_check_obs_passes_on_a_healthy_robot_and_saves_one_jpeg_per_camera(tmp_path):
    cfg = munich()
    report = checks.check_obs(FakeObsNode(cfg), cfg, 0.0, tmp_path, backend=FakeSpine(0.43), sleep=no_sleep)
    assert report.passed, report.format()
    assert report.name == "check_obs"
    for name in cfg.cameras:
        assert (tmp_path / f"check_obs_{name}.jpg").is_file()
    head = next(line for line in report.lines if "cam/head" in line)
    assert "1280x720 bgra8" in head
    assert "224x224x3" in head
    assert report.data["streams"]["cam/head"]["rate_hz"] == 20.0
    assert report.data["spine"]["position"] == 0.43
    saved = report.save(tmp_path)
    assert saved == tmp_path / "check_obs.json"
    assert json.loads(saved.read_text())["passed"] is True


def test_check_obs_fails_on_a_missing_joint_and_names_it(tmp_path):
    cfg = munich()
    node = FakeObsNode(cfg)
    names = JOINTS["right"][:6]  # right_fr3v2_joint7 missing: the node cannot decode the stream
    node.stats["arm/right"] = stream(
        cfg.arms["right"].measured_topic,
        raw={"names": names},
        age=None,
        decode_errors=100,
        last_error="MissingJointError: joint(s) ['right_fr3v2_joint7'] not in the message",
    )
    node.arm_q.pop("right")
    node.ages["arm/right"] = float("inf")
    report = checks.check_obs(node, cfg, 0.0, tmp_path, sleep=no_sleep)
    assert not report.passed
    fails = rows(report, "FAIL")
    assert any("right_fr3v2_joint7" in line and "arm/right" in line for line in fails), report.format()


def test_check_obs_passes_without_the_optional_left_gripper_state(tmp_path):
    cfg = munich()
    node = FakeObsNode(cfg)
    node.drop("grip/left")
    report = checks.check_obs(node, cfg, 0.0, tmp_path, sleep=no_sleep)
    assert report.passed, report.format()
    left = next(line for line in report.lines if "grip/left" in line)
    assert "optional" in left


def test_check_obs_fails_on_a_missing_required_stream(tmp_path):
    cfg = munich()
    node = FakeObsNode(cfg)
    node.drop("cam/wrist_right")
    report = checks.check_obs(node, cfg, 0.0, tmp_path, sleep=no_sleep)
    assert not report.passed
    assert any(cfg.cameras["wrist_right"].topic in line for line in rows(report, "FAIL"))


def test_check_obs_fails_on_an_unsupported_encoding(tmp_path):
    cfg = munich()
    node = FakeObsNode(cfg)
    node.stats["cam/head"] = stream(
        cfg.cameras["head"].topic,
        rate=20.0,
        age=None,
        raw={"height": 720, "width": 1280, "encoding": "mono16", "step": 2560},
        decode_errors=160,
        last_error="UnsupportedEncodingError: unsupported image encoding 'mono16' (supported: rgb8, bgr8, rgba8, bgra8)",
    )
    node.images.pop("head")
    report = checks.check_obs(node, cfg, 0.0, tmp_path, sleep=no_sleep)
    assert not report.passed
    assert any("mono16" in line for line in rows(report, "FAIL"))


def test_check_obs_warns_on_a_slow_stream_and_a_far_spine(tmp_path):
    cfg = munich()
    node = FakeObsNode(cfg)
    node.stats["cam/head"]["rate_hz"] = 12.0  # < 80 % of 20 Hz
    node.stats["arm/left"]["rate_hz"] = 60.0  # < 80 % of 100 Hz
    report = checks.check_obs(node, cfg, 0.0, tmp_path, backend=FakeSpine(0.9), sleep=no_sleep)
    assert report.passed  # warnings only
    warns = rows(report, "warn")
    assert any("cam/head" in line and "12.0 Hz" in line for line in warns)
    assert any("arm/left" in line for line in warns)
    assert any("spine" in line and "max_delta_m" in line for line in warns)


def test_check_obs_reports_state_outside_the_training_range(tmp_path):
    cfg = munich()
    node = FakeObsNode(cfg)
    stats = {"norm_stats": {"state": {"q01": [-0.1] * 16, "q99": [0.1] * 16}}}
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(stats))
    report = checks.check_obs(node, cfg, 0.0, tmp_path, norm_stats_path=str(path), sleep=no_sleep)
    assert report.passed
    assert any("outside the training range" in line for line in rows(report, "warn"))
    assert report.data["state_range"]


# ---- check_infer ----


def infer_cfg(port, *overrides):
    return munich(*overrides, env={"POLICY_SERVER_HOST": "127.0.0.1", "POLICY_SERVER_PORT": str(port)})


def test_check_infer_passes_against_a_munich16_server(stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    report = checks.check_infer(infer_cfg(server.port), n=2)
    assert report.passed, report.format()
    assert report.data["metadata"]["layout"] == "munich16"
    assert report.data["chunk_shape"] == [50, 16]
    assert len(report.data["rtt_ms"]) == 2
    assert any("zeros" in line for line in report.lines)


def test_check_infer_uses_a_live_snapshot(stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    cfg = infer_cfg(server.port)
    snap = FakeObsNode(cfg).snapshot()
    report = checks.check_infer(cfg, n=1, snapshot=snap)
    assert report.passed, report.format()
    assert any("live" in line for line in report.lines)


def test_check_infer_fails_against_a_sim20_server_with_the_metadata_message(stub_server):
    server = stub_server(layout_id="sim20", latency_s=0.01)
    report = checks.check_infer(infer_cfg(server.port), n=2)
    assert not report.passed
    fail = rows(report, "FAIL")
    assert any("munich16" in line and "stub-sim20" in line for line in fail), report.format()


def test_check_infer_fails_clearly_when_nothing_listens(free_port):
    report = checks.check_infer(infer_cfg(free_port, "server.connect_timeout_s=1"), n=1)
    assert not report.passed
    assert any(f"127.0.0.1:{free_port}" in line for line in rows(report, "FAIL"))


def test_check_infer_warns_when_the_round_trip_exceeds_the_budget(stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.2)
    # async: trigger_len = ceil(0.1 * 10) = 1 step = 50 ms < 200 ms
    cfg = infer_cfg(
        server.port, "inference.mode=async", "inference.execute_horizon=10", "inference.async.trigger_threshold=0.1"
    )
    report = checks.check_infer(cfg, n=1)
    assert report.passed
    assert any("budget" in line for line in rows(report, "warn")), report.format()


# ---- check_motion ----


@pytest.mark.parametrize(
    ("commanded", "moved", "verdict"),
    [
        (0.05, 0.05, "follows"),
        (0.05, 0.03, "follows"),
        (-0.05, -0.07, "follows"),
        (0.05, -0.05, "reversed"),
        (-0.05, 0.04, "reversed"),
        (0.05, 0.001, "no-motion"),
        (0.05, -0.004, "no-motion"),
        (0.05, 0.02, "unclear"),
        (0.05, 0.1, "unclear"),
        (0.05, -0.1, "unclear"),
    ],
)
def test_motion_verdict(commanded, moved, verdict):
    assert checks.motion_verdict(commanded, moved) == verdict


class MotionRig:
    """A fake arm: measured = base + response[j] * (commanded - base) per joint, immediately."""

    def __init__(self, cfg, side, response):
        self.cfg = cfg
        self.side = side
        self.base = np.array(cfg.arms[side].start_pose)
        self.cmd = self.base.copy()
        self.response = np.asarray(response, dtype=float)
        self.entered = self.exited = 0
        self.exit_exc = None
        self.waypoints = []
        self.jump = None  # (n-th waypoint, joint, offset): measured jumps away

    # session
    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited += 1
        self.exit_exc = exc_type
        return False

    # streamer
    def set_waypoint(self, side, q):
        assert side == self.side
        self.cmd = np.array(q, dtype=float)
        self.waypoints.append(self.cmd.copy())

    def healthy(self):
        return True

    def measured(self, side):
        q = self.base + self.response * (self.cmd - self.base)
        if self.jump is not None and len(self.waypoints) >= self.jump[0]:
            q[self.jump[1]] += self.jump[2]
        return q


def run_motion(rig, joints="all", delta=0.05):
    return checks.check_motion(
        rig.side,
        joints,
        delta,
        cfg=rig.cfg,
        session=rig,
        streamer=rig,
        measured=rig.measured,
        sleep=no_sleep,
    )


def test_check_motion_verdicts_and_the_joint_signs_line():
    cfg = munich("command_mode=relative_to_activation")
    rig = MotionRig(cfg, "right", [1, -1, 1, 1, 0, 1, 0.9])
    report = run_motion(rig)
    verdicts = [j["verdict"] for j in report.data["joints"]]
    assert verdicts == ["follows", "reversed", "follows", "follows", "no-motion", "follows", "follows"]
    assert not report.passed
    assert report.data["joint_signs"] == [1, -1, 1, 1, 1, 1, 1]
    assert any("right: [1, -1, 1, 1, 1, 1, 1]" in line for line in report.lines), report.format()
    assert rig.entered == rig.exited == 1
    np.testing.assert_allclose(rig.cmd, rig.base, atol=1e-12)  # every joint ramped back
    assert report.name == "check_motion_right"


def test_check_motion_one_joint_follows_and_steps_stay_small():
    cfg = munich()
    rig = MotionRig(cfg, "left", [1] * 7)
    report = run_motion(rig, joints=[3], delta=0.05)
    assert report.passed, report.format()
    assert [j["joint"] for j in report.data["joints"]] == [3]
    steps = np.diff(np.array([rig.base, *rig.waypoints]), axis=0)
    assert np.max(np.abs(steps)) <= 0.05 / 20 + 1e-12
    moved = np.array(rig.waypoints) - rig.base
    assert np.all(moved[:, [0, 1, 3, 4, 5, 6]] == 0)


def test_check_motion_moves_away_from_a_near_limit():
    cfg = munich()
    rig = MotionRig(cfg, "left", [1] * 7)
    q_max = cfg.safety.joint_limits["left"][1]
    rig.base[0] = q_max[0] - cfg.safety.q_margin_rad - 0.01  # no room for +delta on joint 1
    rig.cmd = rig.base.copy()
    report = run_motion(rig, joints=[1])
    assert report.passed, report.format()
    assert report.data["joints"][0]["commanded"] == pytest.approx(-0.05)


def test_check_motion_aborts_and_releases_when_the_arm_runs_away():
    cfg = munich()
    rig = MotionRig(cfg, "right", [1] * 7)
    rig.jump = (5, 2, 0.2)  # joint 3 jumps 0.2 rad > 2 * 0.05 + 0.02
    report = run_motion(rig, joints=[1])
    assert not report.passed
    assert any("aborted" in line for line in rows(report, "FAIL")), report.format()
    assert rig.exited == 1  # the session saw the abort and released
    assert rig.exit_exc is not None


def test_check_motion_reports_a_refused_takeover():
    cfg = munich()

    class Refusing(MotionRig):
        def __enter__(self):
            raise TakeoverError("right: 3 message(s) from another publisher on /right/gello/joint_states")

    rig = Refusing(cfg, "right", [1] * 7)
    report = run_motion(rig)
    assert not report.passed
    assert any("/right/gello/joint_states" in line for line in rows(report, "FAIL"))
