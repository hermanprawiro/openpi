"""ROS adapters against in-process peers: a helper node (own executor thread) publishes sensor messages, subscribes
to what we publish and serves fake controller_manager / PTP / spine endpoints. Runs inside a ROS 2 Humble
container (`ros` marker); use a private `ROS_DOMAIN_ID` (default 42 here) so nothing on the network interferes.
The RosMotionBackend tests additionally need `franka_msgs` / `franka_spine_msgs` (built in the client image)."""

import _thread
from collections.abc import Callable
import dataclasses
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.ros

rclpy = pytest.importorskip("rclpy")

from controller_manager_msgs.msg import ControllerState  # noqa: E402
from controller_manager_msgs.srv import ListControllers  # noqa: E402
from controller_manager_msgs.srv import SwitchController  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402
from robot_client.config import load_site_config  # noqa: E402
from robot_client.ros_io import PREVIEW_TOPIC  # noqa: E402
from robot_client.ros_io import RobotNode  # noqa: E402
from robot_client.ros_io import RosControllerManager  # noqa: E402
from robot_client.takeover import OUR_FRAME_ID  # noqa: E402
from robot_client.takeover import TakeoverError  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float32  # noqa: E402

MUNICH_YAML = Path(__file__).resolve().parents[1] / "robot_site.munich.yaml"
CM = "/right/controller_manager"


@pytest.fixture
def ros():
    rclpy.init(domain_id=int(os.environ.get("ROS_DOMAIN_ID", "42")), signal_handler_options=SignalHandlerOptions.NO)
    yield
    rclpy.try_shutdown()


@pytest.fixture
def cfg():
    return load_site_config(MUNICH_YAML, [], {})


class Helper:
    """A peer node spun by its own executor thread."""

    def __init__(self):
        self.node = Node("ros_io_test_helper")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.received: dict[str, list] = {}

    def subscribe(self, msg_type, topic: str) -> list:
        msgs = self.received.setdefault(topic, [])
        self.node.create_subscription(msg_type, topic, msgs.append, 10)
        return msgs

    def close(self):
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(timeout=2.0)
        self.node.destroy_node()


@pytest.fixture
def helper(ros):
    h = Helper()
    yield h
    h.close()


@pytest.fixture
def make_node(ros):
    nodes = []

    def _make(cfg, command_mode="none") -> RobotNode:
        node = RobotNode(cfg, command_mode=command_mode)
        node.start()
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.shutdown()


def wait_until(pred: Callable[[], bool], timeout: float = 5.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout} s waiting for {what}")
        time.sleep(0.01)


def bgra_image(rgb: np.ndarray, pad: int = 0) -> Image:
    """An Image message with bgra8 encoding (and `pad` bytes of row padding) carrying the RGB array."""
    h, w, _ = rgb.shape
    bgra = np.concatenate([rgb[:, :, ::-1], np.full((h, w, 1), 255, np.uint8)], axis=2).reshape(h, w * 4)
    rows = np.concatenate([bgra, np.zeros((h, pad), np.uint8)], axis=1)
    msg = Image(height=h, width=w, encoding="bgra8", step=w * 4 + pad)
    msg.data = rows.tobytes()
    return msg


def joint_state(names, positions, frame_id: str = "") -> JointState:
    msg = JointState(name=list(names), position=[float(p) for p in positions])
    msg.header.frame_id = frame_id
    return msg


# ---- RobotNode: subscriptions and snapshot ----


def test_snapshot_has_rgb_image_mapped_joints_and_fresh_ages(cfg, make_node, helper):
    node = make_node(cfg)
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    image_pub = helper.node.create_publisher(Image, cfg.cameras["head"].topic, 10)
    arm = cfg.arms["right"]
    joints_pub = helper.node.create_publisher(JointState, arm.measured_topic, 10)
    grip_pub = helper.node.create_publisher(JointState, arm.gripper.state_topic, 10)
    # reversed order plus an extra joint: mapped by name
    names = ["extra_joint", *reversed(arm.joint_names)]
    q = np.linspace(0.1, 0.7, 7)
    positions = [9.9, *reversed(q)]

    def publish_all():
        image_pub.publish(bgra_image(rgb, pad=8))
        joints_pub.publish(joint_state(names, positions))
        grip_pub.publish(joint_state(["finger", arm.gripper.joint], [0.0, 0.42]))

    def have_all():
        publish_all()
        snap = node.snapshot()
        return "head" in snap.images and "right" in snap.arm_q and "right" in snap.knuckle

    wait_until(have_all, what="head image, right joints and right gripper")
    publish_all()
    time.sleep(0.05)
    snap = node.snapshot()

    np.testing.assert_array_equal(snap.images["head"], rgb)
    assert snap.images["head"].dtype == np.uint8
    np.testing.assert_allclose(snap.arm_q["right"], q)
    assert snap.knuckle == {"right": pytest.approx(0.42)}  # left gripper never received: absent
    assert set(snap.ages) == {
        "cam/head",
        "cam/wrist_left",
        "cam/wrist_right",
        "arm/left",
        "arm/right",
        "grip/left",
        "grip/right",
    }
    for key in ("cam/head", "arm/right", "grip/right"):
        assert snap.ages[key] < 0.2, key
    for key in ("cam/wrist_left", "cam/wrist_right", "arm/left", "grip/left"):
        assert snap.ages[key] == float("inf"), key
    assert "left" not in snap.arm_q
    assert node.measured("left") is None
    np.testing.assert_allclose(node.measured("right"), q)

    stats = node.stream_stats()
    assert stats["cam/head"]["count"] >= 2
    assert stats["cam/head"]["raw"] == {"height": 4, "width": 6, "encoding": "bgra8", "step": 6 * 4 + 8}
    assert stats["cam/head"]["decode_errors"] == 0
    assert stats["arm/right"]["raw"]["names"] == names
    assert stats["cam/wrist_left"]["count"] == 0
    assert stats["cam/wrist_left"]["age_s"] is None


def test_snapshot_arrays_are_copies(cfg, make_node, helper):
    node = make_node(cfg)
    pub = helper.node.create_publisher(JointState, cfg.arms["left"].measured_topic, 10)
    names = cfg.arms["left"].joint_names

    def have():
        pub.publish(joint_state(names, np.zeros(7)))
        return node.measured("left") is not None

    wait_until(have, what="left joints")
    node.snapshot().arm_q["left"][:] = 5.0
    node.measured("left")[:] = 5.0
    np.testing.assert_array_equal(node.measured("left"), np.zeros(7))


def test_undecodable_messages_are_counted_and_not_stored(cfg, make_node, helper):
    node = make_node(cfg)
    image_pub = helper.node.create_publisher(Image, cfg.cameras["wrist_left"].topic, 10)
    joints_pub = helper.node.create_publisher(JointState, cfg.arms["left"].measured_topic, 10)
    bad_image = Image(height=2, width=2, encoding="mono16", step=4)
    bad_image.data = bytes(8)
    short_joints = joint_state(cfg.arms["left"].joint_names[:6], np.zeros(6))  # joint7 missing

    def errors_seen():
        image_pub.publish(bad_image)
        joints_pub.publish(short_joints)
        stats = node.stream_stats()
        return stats["cam/wrist_left"]["decode_errors"] > 0 and stats["arm/left"]["decode_errors"] > 0

    wait_until(errors_seen, what="decode errors")
    stats = node.stream_stats()
    assert "mono16" in stats["cam/wrist_left"]["last_error"]
    assert "left_fr3v2_joint7" in stats["arm/left"]["last_error"]
    snap = node.snapshot()
    assert "wrist_left" not in snap.images
    assert "left" not in snap.arm_q
    assert snap.ages["cam/wrist_left"] == float("inf")
    assert snap.ages["arm/left"] == float("inf")


def test_best_effort_publishers_are_received(cfg, make_node, helper):
    """Sensor streams are subscribed best-effort, so best-effort publishers (common for cameras) match too."""
    node = make_node(cfg)
    pub = helper.node.create_publisher(JointState, cfg.arms["right"].measured_topic, rclpy.qos.qos_profile_sensor_data)

    def have():
        pub.publish(joint_state(cfg.arms["right"].joint_names, np.ones(7)))
        return node.measured("right") is not None

    wait_until(have, what="right joints from a best-effort publisher")


# ---- RobotNode: graph view ----


def test_count_publishers_includes_ours_and_counts_an_extra_publisher(cfg, make_node, helper):
    node = make_node(cfg, command_mode="live")
    topic = cfg.arms["right"].command_topic
    wait_until(lambda: node.count_publishers(topic) == 1, what="our own command publisher")
    helper.node.create_publisher(JointState, topic, 10)
    wait_until(lambda: node.count_publishers(topic) == 2, what="the extra publisher")


def test_foreign_messages_counts_only_other_frame_ids_within_the_window(cfg, make_node, helper):
    node = make_node(cfg, command_mode="live")
    arm = cfg.arms["right"]
    foreign_pub = helper.node.create_publisher(JointState, arm.command_topic, 10)
    wait_until(lambda: node.count_publishers(arm.command_topic) == 2, what="discovery")
    time.sleep(0.3)  # let the subscription match too
    for _ in range(20):
        node.publish_arm("right", np.zeros(7))  # ours: never foreign
    time.sleep(0.3)
    assert node.foreign_messages(arm.command_topic, 1.0) == 0

    def foreign_seen():
        foreign_pub.publish(joint_state(arm.command_joint_names or arm.joint_names, np.zeros(7), frame_id="gello"))
        return node.foreign_messages(arm.command_topic, 1.0) > 0

    wait_until(foreign_seen, what="a foreign message")
    time.sleep(0.4)
    assert node.foreign_messages(arm.command_topic, 0.2) == 0  # all older than the window
    assert node.foreign_messages(arm.command_topic, 5.0) > 0
    with pytest.raises(ValueError, match="not a command topic"):
        node.foreign_messages("/some/other/topic", 1.0)


def test_another_robot_client_instance_is_foreign(cfg, make_node, helper):
    """A second robot_client process (its own `robot_client:<id>` frame_id) is a foreign publisher."""
    node = make_node(cfg, command_mode="none")
    topic = cfg.arms["right"].command_topic
    pub = helper.node.create_publisher(JointState, topic, 10)
    other = "robot_client:deadbeef"
    assert other != OUR_FRAME_ID

    def seen():
        pub.publish(joint_state(cfg.arms["right"].joint_names, np.zeros(7), frame_id=other))
        return node.foreign_messages(topic, 1.0) > 0

    wait_until(seen, what="a message from another robot_client instance")


def test_measured_age_tracks_the_last_sample(cfg, make_node, helper):
    node = make_node(cfg)
    assert node.measured_age("left") == float("inf")
    pub = helper.node.create_publisher(JointState, cfg.arms["left"].measured_topic, 10)

    def have():
        pub.publish(joint_state(cfg.arms["left"].joint_names, np.zeros(7)))
        return node.measured("left") is not None

    wait_until(have, what="left joints")
    assert node.measured_age("left") < 0.5
    time.sleep(0.4)  # the stream stops
    assert node.measured_age("left") >= 0.4
    assert node.measured_age("nonexistent") == float("inf")


def test_foreign_messages_from_a_best_effort_publisher_are_seen(cfg, make_node, helper):
    node = make_node(cfg, command_mode="none")
    topic = cfg.arms["left"].command_topic
    pub = helper.node.create_publisher(JointState, topic, rclpy.qos.qos_profile_sensor_data)

    def seen():
        pub.publish(joint_state(cfg.arms["left"].joint_names, np.zeros(7)))
        return node.foreign_messages(topic, 1.0) > 0

    wait_until(seen, what="a message from a best-effort foreign publisher")


def test_uptime_grows(cfg, make_node):
    node = make_node(cfg)
    t0 = node.uptime_s()
    time.sleep(0.05)
    assert 0.0 <= t0 < node.uptime_s()


# ---- RobotNode: publishing ----


def test_live_publish_arm_uses_command_names_frame_id_and_gripper_float32(cfg, make_node, helper):
    command_names = tuple(f"gello_joint{i}" for i in range(1, 8))
    arms = dict(cfg.arms)
    arms["right"] = dataclasses.replace(arms["right"], command_joint_names=command_names)
    cfg = dataclasses.replace(cfg, arms=arms)
    node = make_node(cfg, command_mode="live")
    arm = cfg.arms["right"]
    cmds = helper.subscribe(JointState, arm.command_topic)
    widths = helper.subscribe(Float32, arm.gripper.command_topic)
    q = np.linspace(-0.3, 0.3, 7)

    def got():
        node.publish_arm("right", q)
        node.publish_gripper("right", 0.25)
        return cmds and widths

    wait_until(got, what="command and gripper messages")
    msg = cmds[-1]
    assert list(msg.name) == list(command_names)
    np.testing.assert_allclose(msg.position, q)
    assert msg.header.frame_id == OUR_FRAME_ID
    assert msg.header.stamp.sec > 0
    assert widths[-1].data == pytest.approx(0.25)
    assert node.stream_stats()["commands"]["right"]["published"] >= 1
    with pytest.raises(ValueError, match="7"):
        node.publish_arm("right", np.zeros(6))


def test_preview_mode_publishes_on_the_preview_topic_only(cfg, make_node, helper):
    node = make_node(cfg, command_mode="preview")
    arm = cfg.arms["left"]
    preview_topic = PREVIEW_TOPIC.format(side="left")
    assert preview_topic == "/left/robot_client_preview/joint_states"
    previews = helper.subscribe(JointState, preview_topic)

    def got():
        node.publish_arm("left", np.full(7, 0.5))
        node.publish_gripper("left", 0.0)  # suppressed
        return previews

    wait_until(got, what="a preview message")
    assert list(previews[-1].name) == list(arm.joint_names)
    assert previews[-1].header.frame_id == OUR_FRAME_ID
    own = {topic for topic, _ in node.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace())}
    for side in ("left", "right"):
        assert cfg.arms[side].command_topic not in own
        assert cfg.arms[side].gripper.command_topic not in own
        assert PREVIEW_TOPIC.format(side=side) in own
    assert node.count_publishers(arm.command_topic) == 0
    assert node.stream_stats()["commands"]["left"]["gripper_suppressed"] >= 1


def test_none_mode_creates_no_command_publishers_and_refuses_to_publish(cfg, make_node):
    node = make_node(cfg, command_mode="none")
    own = {topic for topic, _ in node.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace())}
    for side in ("left", "right"):
        assert cfg.arms[side].command_topic not in own
        assert cfg.arms[side].gripper.command_topic not in own
        assert PREVIEW_TOPIC.format(side=side) not in own
    with pytest.raises(RuntimeError, match="none"):
        node.publish_arm("left", np.zeros(7))
    node.publish_gripper("left", 1.0)  # suppressed, no error


def test_unknown_command_mode_is_rejected(cfg, ros):
    with pytest.raises(ValueError, match="command_mode"):
        RobotNode(cfg, command_mode="sim")


# ---- RosControllerManager ----


class FakeControllerManager:
    """list_controllers / switch_controller services under `ns` on the helper node."""

    def __init__(self, helper: Helper, ns: str, *, delay_s: float = 0.0, ok: bool = True):
        self.states = {"joint_impedance_controller": "inactive", "franka_robot_state_broadcaster": "active"}
        self.switch_requests = []
        self.delay_s = delay_s
        self.ok = ok
        helper.node.create_service(ListControllers, f"{ns}/list_controllers", self._list)
        helper.node.create_service(SwitchController, f"{ns}/switch_controller", self._switch)

    def _list(self, request, response):
        time.sleep(self.delay_s)
        response.controller = [ControllerState(name=n, state=s) for n, s in self.states.items()]
        return response

    def _switch(self, request, response):
        time.sleep(self.delay_s)
        self.switch_requests.append(request)
        if self.ok:
            for name in request.activate_controllers:
                self.states[name] = "active"
            for name in request.deactivate_controllers:
                self.states[name] = "inactive"
        response.ok = self.ok
        return response


def test_controller_manager_lists_and_switches(cfg, make_node, helper):
    node = make_node(cfg)
    fake = FakeControllerManager(helper, CM)
    cm = RosControllerManager(node, CM, timeout_s=3.0)
    assert cm.list_controllers() == {
        "joint_impedance_controller": "inactive",
        "franka_robot_state_broadcaster": "active",
    }
    assert cm.switch(["joint_impedance_controller"], [], timeout_s=1.5) is True
    assert cm.list_controllers()["joint_impedance_controller"] == "active"
    request = fake.switch_requests[-1]
    assert list(request.activate_controllers) == ["joint_impedance_controller"]
    assert list(request.deactivate_controllers) == []
    assert request.strictness == SwitchController.Request.STRICT
    assert request.activate_asap is True
    assert (request.timeout.sec, request.timeout.nanosec) == (1, 500_000_000)
    fake.ok = False
    assert cm.switch([], ["joint_impedance_controller"], timeout_s=1.0) is False


def test_controller_manager_without_service_times_out_naming_it(cfg, make_node):
    node = make_node(cfg)
    cm = RosControllerManager(node, "/nowhere/controller_manager", timeout_s=0.3)
    t0 = time.monotonic()
    with pytest.raises(TakeoverError, match="/nowhere/controller_manager/list_controllers"):
        cm.list_controllers()
    assert time.monotonic() - t0 < 1.0
    with pytest.raises(TakeoverError, match="/nowhere/controller_manager/switch_controller"):
        cm.switch([], ["joint_impedance_controller"], timeout_s=0.2)


def test_controller_manager_slow_response_times_out_naming_the_service(cfg, make_node, helper):
    node = make_node(cfg)
    FakeControllerManager(helper, CM, delay_s=1.5)
    cm = RosControllerManager(node, CM, timeout_s=0.4)
    t0 = time.monotonic()
    with pytest.raises(TakeoverError, match=f"{CM}/list_controllers"):
        cm.list_controllers()
    assert time.monotonic() - t0 < 1.2


# ---- RosMotionBackend ----


def test_motion_backend_without_franka_messages_raises_a_clear_error(cfg, make_node):
    try:
        import franka_msgs  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("franka_msgs is importable here")
    from robot_client.ros_io import RosMotionBackend

    with pytest.raises(RuntimeError, match="franka_msgs"):
        RosMotionBackend(make_node(cfg), cfg)


class FakeMotionServers:
    """PTP action server for the right arm, spine move action and get_position service. `ptp_mode`:
    'reach' (succeed, TARGET_REACHED), 'abort' (ABORTED with an error message), 'hang' (run until canceled)."""

    def __init__(self, helper: Helper, cfg):
        from franka_msgs.action import PTPMotion
        from franka_msgs.msg import TargetStatus
        from franka_spine_msgs.action import MoveAbsolute
        from franka_spine_msgs.srv import GetPosition
        from rclpy.action import ActionServer
        from rclpy.action import CancelResponse
        from rclpy.callback_groups import ReentrantCallbackGroup

        self.PTPMotion, self.TargetStatus, self.MoveAbsolute = PTPMotion, TargetStatus, MoveAbsolute
        self.ptp_goals, self.spine_goals = [], []
        self.ptp_mode = "reach"
        self.spine_success = True
        self.position = 0.43
        self.canceled = threading.Event()
        group = ReentrantCallbackGroup()
        self._servers = [
            ActionServer(
                helper.node,
                PTPMotion,
                cfg.arms["right"].ptp_action,
                self._ptp,
                cancel_callback=lambda _: CancelResponse.ACCEPT,
                callback_group=group,
            ),
            ActionServer(helper.node, MoveAbsolute, cfg.spine.action, self._spine, callback_group=group),
        ]
        helper.node.create_service(GetPosition, cfg.spine.get_position_service, self._position, callback_group=group)

    def _ptp(self, goal_handle):
        self.ptp_goals.append(goal_handle.request)
        result = self.PTPMotion.Result()
        if self.ptp_mode == "hang":
            while not goal_handle.is_cancel_requested:
                time.sleep(0.01)
            self.canceled.set()
            goal_handle.canceled()
            result.target_status.status = self.TargetStatus.ABORTED
            return result
        if self.ptp_mode == "abort":
            result.target_status.status = self.TargetStatus.ABORTED
            result.error_message = "joint limit violation"
            goal_handle.abort()
            return result
        result.target_status.status = self.TargetStatus.TARGET_REACHED
        goal_handle.succeed()
        return result

    def _spine(self, goal_handle):
        self.spine_goals.append(goal_handle.request)
        result = self.MoveAbsolute.Result(success=self.spine_success, error="" if self.spine_success else "fault")
        goal_handle.succeed()
        return result

    def _position(self, request, response):
        response.position = self.position
        response.success = True
        return response


@pytest.fixture
def motion(cfg, make_node, helper):
    pytest.importorskip("franka_msgs")
    pytest.importorskip("franka_spine_msgs")
    from robot_client.ros_io import RosMotionBackend

    servers = FakeMotionServers(helper, cfg)
    node = make_node(cfg, command_mode="live")
    return RosMotionBackend(node, cfg), servers, node


def test_ptp_sends_the_goal_fields_and_reports_target_reached(motion):
    backend, servers, _ = motion
    pose = [0.1, -0.2, 0.3, -2.0, 0.0, 1.5, 0.7]
    assert backend.ptp("right", pose, max_vel=[0.3] * 7, tol=0.01, timeout_s=5.0) is True
    goal = servers.ptp_goals[-1]
    np.testing.assert_allclose(goal.goal_joint_configuration, pose)
    np.testing.assert_allclose(goal.maximum_joint_velocities, [0.3] * 7)
    assert goal.goal_tolerance == pytest.approx(0.01)


def test_ptp_aborted_is_a_failure(motion):
    backend, servers, _ = motion
    servers.ptp_mode = "abort"
    assert backend.ptp("right", [0.0] * 7, max_vel=[0.3] * 7, tol=0.01, timeout_s=5.0) is False


def test_ptp_without_server_fails_within_the_timeout(motion):
    backend, _, _ = motion
    t0 = time.monotonic()
    assert backend.ptp("left", [0.0] * 7, max_vel=[0.3] * 7, tol=0.01, timeout_s=0.3) is False
    assert time.monotonic() - t0 < 1.0


def test_ptp_timeout_cancels_the_goal(motion):
    backend, servers, _ = motion
    servers.ptp_mode = "hang"
    t0 = time.monotonic()
    assert backend.ptp("right", [0.0] * 7, max_vel=[0.3] * 7, tol=0.01, timeout_s=0.5) is False
    assert time.monotonic() - t0 < 3.0
    assert servers.canceled.wait(2.0)


def test_ptp_keyboard_interrupt_cancels_the_goal(motion):
    backend, servers, _ = motion
    servers.ptp_mode = "hang"
    timer = threading.Timer(0.5, _thread.interrupt_main)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            backend.ptp("right", [0.0] * 7, max_vel=[0.3] * 7, tol=0.01, timeout_s=10.0)
    finally:
        timer.cancel()
    assert servers.canceled.wait(2.0)


def test_spine_move_and_position(motion, cfg):
    backend, servers, _ = motion
    assert backend.spine_position(timeout_s=2.0) == pytest.approx(0.43)
    assert backend.spine(0.45, 0.1, 0.2, 0.3, timeout_s=5.0) is True
    goal = servers.spine_goals[-1]
    assert (goal.position, goal.velocity, goal.acceleration, goal.deceleration) == pytest.approx((0.45, 0.1, 0.2, 0.3))
    servers.spine_success = False
    assert backend.spine(0.45, 0.1, 0.2, 0.3, timeout_s=5.0) is False


def test_spine_position_unavailable_is_none(ros, helper, cfg):
    pytest.importorskip("franka_spine_msgs")
    from robot_client.ros_io import RosMotionBackend

    cfg = dataclasses.replace(cfg, spine=dataclasses.replace(cfg.spine, get_position_service="/nowhere/get_position"))
    node = RobotNode(cfg, command_mode="none")
    node.start()
    try:
        t0 = time.monotonic()
        assert RosMotionBackend(node, cfg).spine_position(timeout_s=0.3) is None
        assert time.monotonic() - t0 < 1.0
    finally:
        node.shutdown()


def test_gripper_publishes_width(motion, helper, cfg):
    backend, _, _ = motion
    widths = helper.subscribe(Float32, cfg.arms["left"].gripper.command_topic)

    def got():
        backend.gripper("left", 1.0)
        return widths

    wait_until(got, what="gripper width")
    assert widths[-1].data == pytest.approx(1.0)
