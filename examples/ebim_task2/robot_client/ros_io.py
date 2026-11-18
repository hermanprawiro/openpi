"""ROS 2 adapters (spec §6.2, plan Task 16): the only module that talks to ROS (Humble, Fast DDS).

`RobotNode` subscribes to the cameras, the measured arm joints and the gripper joints (best-effort sensor-data QoS,
which matches both reliable and best-effort publishers), stores the latest decoded sample of each stream with its
`time.monotonic()` arrival time and per-stream counters, and publishes the arm joint targets and gripper widths. It
implements `runner.ObsSource` (`snapshot`) and `takeover.GraphView` (`uptime_s`, `count_publishers`,
`foreign_messages`, `measured`); `publish_arm` / `measured` are the streamer's callables. `command_mode`:
`live` publishes on the configured command topics, `preview` publishes arm targets on
`/<side>/robot_client_preview/joint_states` and suppresses gripper commands, `none` creates no command publishers.

`RosControllerManager` implements `takeover.ControllerManager` and `RosMotionBackend` implements
`prepare.MotionBackend`. Every service and action call is asynchronous with a bounded wait (never a blocking
`call()`), so a dead peer costs at most the configured timeout; a timed-out or interrupted action goal is canceled.
The node must be spinning (`start()`) for any call to complete.

If rclpy is not initialized yet, `RobotNode` initializes it without rclpy's signal handlers: a Ctrl-C must reach
Python as KeyboardInterrupt (the release path) and must not shut the ROS context down, because release still needs
the controller_manager.
"""

from collections import deque
from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import logging
import math
import threading
import time
from typing import Any, Literal

from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import ListControllers
from controller_manager_msgs.srv import SwitchController
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from robot_client.config import SiteConfig
from robot_client.preprocess import Snapshot
from robot_client.preprocess import decode_compressed_image
from robot_client.preprocess import decode_raw_image
from robot_client.preprocess import map_joints
from robot_client.takeover import OUR_FRAME_ID
from robot_client.takeover import TakeoverError

logger = logging.getLogger(__name__)

CommandMode = Literal["live", "preview", "none"]
COMMAND_MODES = ("live", "preview", "none")
NODE_NAME = "robot_client"
PREVIEW_TOPIC = "/{side}/robot_client_preview/joint_states"
FOREIGN_KEEP_S = 10.0  # foreign-message arrival times older than this are dropped
POLL_S = 0.005  # granularity of the bounded waits on futures / service and action servers
CANCEL_WAIT_S = 2.0  # how long a timed-out or interrupted goal waits for its cancel to be acknowledged
EXECUTOR_STOP_S = 2.0

# Watches the command topics for foreign publishers (GELLO teleop). Best-effort so that it matches reliable and
# best-effort publishers alike: a reliable subscription would never see a best-effort publisher's messages.
_FOREIGN_WATCH_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
_COMMAND_QOS = 10  # reliable, keep last 10: matches reliable and best-effort controller subscriptions


@dataclasses.dataclass
class _Stream:
    """Latest sample and counters of one subscribed stream. Guarded by RobotNode._lock."""

    topic: str
    count: int = 0  # messages received (decodable or not)
    decode_errors: int = 0
    last_error: str | None = None
    first_rx: float | None = None
    last_rx: float | None = None
    max_gap_s: float = 0.0  # largest interval between two received messages
    raw: dict = dataclasses.field(default_factory=dict)  # shape/encoding/names of the last message
    value: Any = None  # last decoded sample
    t_sample: float | None = None  # arrival time of `value`

    def received(self, now: float, raw: dict) -> None:
        if self.last_rx is not None:
            self.max_gap_s = max(self.max_gap_s, now - self.last_rx)
        else:
            self.first_rx = now
        self.last_rx = now
        self.count += 1
        self.raw = raw

    def as_dict(self, now: float) -> dict:
        span = (self.last_rx - self.first_rx) if self.count > 1 else 0.0
        return {
            "topic": self.topic,
            "count": self.count,
            "rate_hz": (self.count - 1) / span if span > 0 else None,
            "max_gap_s": self.max_gap_s,
            "age_s": None if self.t_sample is None else now - self.t_sample,
            "raw": dict(self.raw),
            "decode_errors": self.decode_errors,
            "last_error": self.last_error,
        }


def _wait_until(is_ready: Callable[[], bool], deadline: float) -> bool:
    """Poll (interruptibly) until `is_ready()` or `deadline` (monotonic) passes; returns `is_ready()`'s outcome."""
    while not is_ready():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(POLL_S, remaining))
    return True


def _call_service(client, request, timeout_s: float):
    """Response of `client` to `request` within `timeout_s` (waiting for the service included); raises
    TimeoutError naming the service. A request that is abandoned (timeout, Ctrl-C) is dropped from the client."""
    deadline = time.monotonic() + timeout_s
    if not _wait_until(client.service_is_ready, deadline):
        raise TimeoutError(f"service {client.srv_name} not available within {timeout_s:g} s")
    future = client.call_async(request)
    try:
        done = _wait_until(future.done, deadline)
    except BaseException:
        _abandon(client, future)
        raise
    if not done:
        _abandon(client, future)
        raise TimeoutError(f"service {client.srv_name} did not answer within {timeout_s:g} s")
    return future.result()


def _abandon(client, future) -> None:
    future.cancel()
    with contextlib.suppress(Exception):  # best effort: a late response is then just ignored
        client.remove_pending_request(future)


class RobotNode(Node):
    def __init__(self, cfg: SiteConfig, *, command_mode: CommandMode):
        if command_mode not in COMMAND_MODES:
            raise ValueError(f"command_mode must be one of {COMMAND_MODES}, got {command_mode!r}")
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        super().__init__(NODE_NAME)
        self.cfg = cfg
        self.command_mode = command_mode
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._executor: MultiThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._closed = False

        self._streams: dict[str, _Stream] = {}
        for name, cam in cfg.cameras.items():
            key = f"cam/{name}"
            if cam.type == "compressed":
                self._subscribe(CompressedImage, cam.topic, key, self._on_compressed)
            else:
                self._subscribe(Image, cam.topic, key, self._on_raw_image)
        for side, arm in cfg.arms.items():
            self._subscribe(JointState, arm.measured_topic, f"arm/{side}", self._joints_handler(arm.joint_names))
            self._subscribe(
                JointState,
                arm.gripper.state_topic,
                f"grip/{side}",
                self._joints_handler((arm.gripper.joint,), scalar=True),
            )

        # Foreign messages on the command topics: arrival times of messages not published by us.
        self._foreign: dict[str, deque[float]] = {}
        for arm in cfg.arms.values():
            topic = arm.command_topic
            self._foreign[topic] = deque()
            self.create_subscription(
                JointState,
                topic,
                lambda msg, topic=topic: self._on_command_topic(topic, msg),
                _FOREIGN_WATCH_QOS,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

        self._arm_pubs = {}
        self._command_names: dict[str, tuple[str, ...]] = {}
        self._gripper_pubs = {}
        self._command_stats = {
            side: {"published": 0, "gripper_published": 0, "gripper_suppressed": 0} for side in cfg.arms
        }
        for side, arm in cfg.arms.items():
            if command_mode == "live":
                self._arm_pubs[side] = self.create_publisher(JointState, arm.command_topic, _COMMAND_QOS)
                self._command_names[side] = tuple(arm.command_joint_names or arm.joint_names)
                self._gripper_pubs[side] = self.create_publisher(Float32, arm.gripper.command_topic, _COMMAND_QOS)
            elif command_mode == "preview":
                self._arm_pubs[side] = self.create_publisher(JointState, PREVIEW_TOPIC.format(side=side), _COMMAND_QOS)
                self._command_names[side] = tuple(arm.joint_names)

    # ---- lifecycle ----

    def start(self) -> None:
        """Spin the node in a daemon thread (MultiThreadedExecutor)."""
        if self._executor is not None:
            raise RuntimeError("RobotNode.start() called twice")
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin, name="ros-executor", daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception:  # e.g. the context shut down under us; nothing to recover
            if not self._closed:
                logger.exception("ROS executor stopped")

    def shutdown(self) -> None:
        """Stop spinning, destroy the node and, if this node initialized rclpy, shut it down. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=EXECUTOR_STOP_S)
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=EXECUTOR_STOP_S)
        self.destroy_node()
        if self._owns_rclpy:
            rclpy.try_shutdown()

    # ---- subscriptions ----

    def _subscribe(self, msg_type, topic: str, key: str, handler: Callable[[Any], Any]) -> None:
        self._streams[key] = _Stream(topic)
        self.create_subscription(
            msg_type,
            topic,
            lambda msg: self._on_message(key, msg, handler),
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),  # one per stream: streams decode in parallel
        )

    def _on_message(self, key: str, msg, handler: Callable[[Any], Any]) -> None:
        now = time.monotonic()
        raw = self._raw_info(msg)
        try:
            value = handler(msg)
            error = None
        except Exception as e:  # counted and shown by check-obs; the stream just goes stale
            value, error = None, f"{type(e).__name__}: {e}"
        with self._lock:
            stream = self._streams[key]
            stream.received(now, raw)
            if error is None:
                stream.value, stream.t_sample = value, now
                return
            stream.decode_errors += 1
            stream.last_error = error
            first_error = stream.decode_errors == 1
        if first_error:
            logger.warning("%s (%s): %s", key, stream.topic, error)

    @staticmethod
    def _raw_info(msg) -> dict:
        if isinstance(msg, Image):
            return {"height": msg.height, "width": msg.width, "encoding": msg.encoding, "step": msg.step}
        if isinstance(msg, CompressedImage):
            return {"format": msg.format, "bytes": len(msg.data)}
        return {"names": list(msg.name)}

    @staticmethod
    def _on_raw_image(msg: Image) -> np.ndarray:
        return decode_raw_image(msg.data, msg.height, msg.width, msg.step, msg.encoding)

    @staticmethod
    def _on_compressed(msg: CompressedImage) -> np.ndarray:
        return decode_compressed_image(msg.data)

    @staticmethod
    def _joints_handler(wanted: Sequence[str], *, scalar: bool = False) -> Callable[[JointState], Any]:
        wanted = tuple(wanted)

        def handle(msg: JointState):
            q = map_joints(msg.name, msg.position, wanted)
            return float(q[0]) if scalar else q

        return handle

    def _on_command_topic(self, topic: str, msg: JointState) -> None:
        if msg.header.frame_id == OUR_FRAME_ID:
            return
        now = time.monotonic()
        with self._lock:
            times = self._foreign[topic]
            times.append(now)
            while times and now - times[0] > FOREIGN_KEEP_S:
                times.popleft()

    # ---- ObsSource / GraphView ----

    def snapshot(self) -> Snapshot:
        """The latest sample of every stream. `ages` has an entry for every configured stream (`inf` if it never
        delivered a decodable sample); `images` / `arm_q` / `knuckle` only hold streams that did."""
        with self._lock:
            now = time.monotonic()
            images, arm_q, knuckle, ages = {}, {}, {}, {}
            for key, stream in self._streams.items():
                ages[key] = math.inf if stream.t_sample is None else now - stream.t_sample
                if stream.value is None:
                    continue
                kind, name = key.split("/", 1)
                if kind == "cam":
                    images[name] = stream.value.copy()
                elif kind == "arm":
                    arm_q[name] = stream.value.copy()
                else:
                    knuckle[name] = stream.value
        return Snapshot(images=images, arm_q=arm_q, knuckle=knuckle, ages=ages, t_mono=now)

    def measured(self, side: str) -> np.ndarray | None:
        """Latest measured joints of `side` in policy order (a copy), or None before the first sample."""
        with self._lock:
            stream = self._streams.get(f"arm/{side}")
            value = None if stream is None else stream.value
            return None if value is None else value.copy()

    def uptime_s(self) -> float:
        return time.monotonic() - self._t0

    # count_publishers(topic) is rclpy's Node.count_publishers: every publisher on the graph, ours included.

    def foreign_messages(self, topic: str, window_s: float) -> int:
        """Messages on command topic `topic` in the last `window_s` seconds whose frame_id is not ours."""
        if topic not in self._foreign:
            raise ValueError(f"{topic} is not a command topic of this node ({sorted(self._foreign)})")
        with self._lock:
            now = time.monotonic()
            return sum(1 for t in self._foreign[topic] if now - t <= window_s)

    def stream_stats(self) -> dict:
        """Per stream (`cam/<name>`, `arm/<side>`, `grip/<side>`) counters plus `commands` per side; plain data."""
        with self._lock:
            now = time.monotonic()
            stats = {key: stream.as_dict(now) for key, stream in self._streams.items()}
            stats["commands"] = {side: dict(c) for side, c in self._command_stats.items()}
        return stats

    # ---- publishing ----

    def publish_arm(self, side: str, q: np.ndarray) -> None:
        """Joint targets of `side`: `live` on its command topic, `preview` on the preview topic; header frame_id is
        `OUR_FRAME_ID`. Raises in `none` mode (no publishers exist)."""
        pub = self._arm_pubs.get(side)
        if pub is None:
            if self.command_mode == "none":
                raise RuntimeError("RobotNode in command_mode 'none' has no command publishers")
            raise KeyError(f"no arm {side!r} in the site config")
        names = self._command_names[side]
        positions = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)]
        if len(positions) != len(names):
            raise ValueError(f"{side}: {len(positions)} joint targets for {len(names)} command joints")
        msg = JointState(name=list(names), position=positions)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = OUR_FRAME_ID
        pub.publish(msg)
        with self._lock:
            self._command_stats[side]["published"] += 1

    def publish_gripper(self, side: str, width: float) -> None:
        """Gripper width (Float32, 1.0 = open) on the gripper command topic; suppressed in `preview` / `none`."""
        pub = self._gripper_pubs.get(side)
        with self._lock:
            self._command_stats[side]["gripper_published" if pub is not None else "gripper_suppressed"] += 1
        if pub is not None:
            pub.publish(Float32(data=float(width)))


class RosControllerManager:
    """`takeover.ControllerManager` for the controller_manager at `namespace` (e.g. `/left/controller_manager`).
    Every call waits at most `timeout_s` (service discovery included) and raises TakeoverError naming the service."""

    def __init__(self, node: Node, namespace: str, timeout_s: float):
        self.namespace = namespace.rstrip("/")
        self.timeout_s = timeout_s
        self._list = node.create_client(ListControllers, f"{self.namespace}/list_controllers")
        self._switch = node.create_client(SwitchController, f"{self.namespace}/switch_controller")

    def _call(self, client, request, timeout_s: float):
        try:
            response = _call_service(client, request, timeout_s)
        except TimeoutError as e:
            raise TakeoverError(f"{e}; is the controller_manager running?") from e
        if response is None:
            raise TakeoverError(f"service {client.srv_name} returned no response")
        return response

    def list_controllers(self) -> dict[str, str]:
        response = self._call(self._list, ListControllers.Request(), self.timeout_s)
        return {c.name: c.state for c in response.controller}

    def switch(self, activate: list[str], deactivate: list[str], timeout_s: float) -> bool:
        """STRICT switch with activate_asap. The controller_manager gives up after `timeout_s`; we wait for its
        answer at most `timeout_s` + 1 s, and never less than the service timeout."""
        request = SwitchController.Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        sec = math.floor(timeout_s)
        request.timeout = Duration(sec=int(sec), nanosec=round((timeout_s - sec) * 1e9))
        response = self._call(self._switch, request, max(self.timeout_s, timeout_s + 1.0))
        return bool(response.ok)


class RosMotionBackend:
    """`prepare.MotionBackend`: franka PTP action per arm, spine MoveAbsolute action and GetPosition service,
    grippers via `RobotNode.publish_gripper`. Needs `franka_msgs` and `franka_spine_msgs` (built into the client
    image). Failures are logged and reported as False / None; KeyboardInterrupt cancels the goal and propagates."""

    def __init__(self, node: RobotNode, cfg: SiteConfig):
        try:
            from franka_msgs.action import PTPMotion
            from franka_msgs.msg import TargetStatus
            from franka_spine_msgs.action import MoveAbsolute
            from franka_spine_msgs.srv import GetPosition
        except ImportError as e:
            raise RuntimeError(
                "prepare needs the franka_msgs and franka_spine_msgs Python packages (franka_ros2 b4164f5), which "
                f"the robot_client image builds and sources; they are not importable here ({e})"
            ) from e
        from action_msgs.msg import GoalStatus

        self._PTPMotion, self._TargetStatus, self._MoveAbsolute, self._GetPosition = (
            PTPMotion,
            TargetStatus,
            MoveAbsolute,
            GetPosition,
        )
        self._succeeded = GoalStatus.STATUS_SUCCEEDED
        self.node = node
        self.cfg = cfg
        self._ptp = {side: ActionClient(node, PTPMotion, arm.ptp_action) for side, arm in cfg.arms.items()}
        self._spine = ActionClient(node, MoveAbsolute, cfg.spine.action)
        self._spine_position = node.create_client(GetPosition, cfg.spine.get_position_service)

    def ptp(self, side: str, joints: Sequence[float], max_vel: Sequence[float], tol: float, timeout_s: float) -> bool:
        goal = self._PTPMotion.Goal()
        goal.goal_joint_configuration = [float(v) for v in joints]
        goal.maximum_joint_velocities = [float(v) for v in max_vel]
        goal.goal_tolerance = float(tol)
        name = self.cfg.arms[side].ptp_action
        outcome = self._run_goal(self._ptp[side], goal, timeout_s, name)
        if outcome is None:
            return False
        status, result = outcome
        if status != self._succeeded or result.target_status.status != self._TargetStatus.TARGET_REACHED:
            logger.warning(
                "%s: PTP failed (goal status %d, target status %d): %s",
                name,
                status,
                result.target_status.status,
                result.error_message or "no error message",
            )
            return False
        return True

    def spine(self, height: float, vel: float, acc: float, dec: float, timeout_s: float) -> bool:
        goal = self._MoveAbsolute.Goal(
            position=float(height), velocity=float(vel), acceleration=float(acc), deceleration=float(dec)
        )
        outcome = self._run_goal(self._spine, goal, timeout_s, self.cfg.spine.action)
        if outcome is None:
            return False
        status, result = outcome
        if status != self._succeeded or not result.success:
            logger.warning(
                "%s: spine move failed (goal status %d, stop_by %r): %s",
                self.cfg.spine.action,
                status,
                result.stop_by,
                result.error or "no error message",
            )
            return False
        return True

    def spine_position(self, timeout_s: float) -> float | None:
        try:
            response = _call_service(self._spine_position, self._GetPosition.Request(), timeout_s)
        except TimeoutError as e:
            logger.warning("spine position unavailable: %s", e)
            return None
        if response is None or not response.success:
            logger.warning("spine position unavailable: %s reported failure", self.cfg.spine.get_position_service)
            return None
        return float(response.position)

    def gripper(self, side: str, width: float) -> None:
        self.node.publish_gripper(side, width)

    # ---- goals ----

    def _run_goal(self, client: ActionClient, goal, timeout_s: float, name: str):
        """(goal status, result) of `goal`, or None if the server is missing, rejects it, or acceptance or the
        result takes longer than `timeout_s` each. A goal still running on timeout or on any exception
        (KeyboardInterrupt included) is canceled, also one accepted only after we stopped waiting."""
        if not _wait_until(client.server_is_ready, time.monotonic() + timeout_s):
            logger.warning("%s: action server not available within %g s", name, timeout_s)
            return None
        send_future = client.send_goal_async(goal)
        goal_handle = None
        canceled = False
        try:
            if not _wait_until(send_future.done, time.monotonic() + timeout_s):
                send_future.add_done_callback(lambda f: self._cancel_late(f, name))
                logger.warning(
                    "%s: goal not accepted within %g s; canceling it if it is accepted later", name, timeout_s
                )
                return None
            goal_handle = send_future.result()
            if not goal_handle.accepted:
                logger.warning("%s: goal rejected", name)
                return None
            result_future = goal_handle.get_result_async()
            if not _wait_until(result_future.done, time.monotonic() + timeout_s):
                logger.warning("%s: no result within %g s; canceling the goal", name, timeout_s)
                canceled = True
                self._cancel(goal_handle, name)
                return None
            wrapped = result_future.result()
            return wrapped.status, wrapped.result
        except BaseException:
            if goal_handle is None:  # cancels it once (or if already) accepted
                send_future.add_done_callback(lambda f: self._cancel_late(f, name))
            elif not canceled:
                self._cancel(goal_handle, name)
            raise

    def _cancel(self, goal_handle, name: str) -> None:
        cancel_future = goal_handle.cancel_goal_async()
        if not _wait_until(cancel_future.done, time.monotonic() + CANCEL_WAIT_S):
            logger.error("%s: goal cancel not acknowledged within %g s", name, CANCEL_WAIT_S)

    @staticmethod
    def _cancel_late(send_future, name: str) -> None:
        try:
            goal_handle = send_future.result()
        except Exception:  # the goal never reached the server
            return
        if goal_handle is not None and goal_handle.accepted:
            logger.warning("%s: canceling a goal accepted after we stopped waiting", name)
            goal_handle.cancel_goal_async()
