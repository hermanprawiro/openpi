"""A fake Munich FR3 Duo, so the whole robot client runs on a machine without the robot (dev tool, plan Task 19).

    python -m robot_client.fake_robot --site PATH [--episode-dir DIR --episode N [--still]]
        [--events /out/fake_robot_events.jsonl] [--tau 0.05] [--command-mode absolute|relative --signs 1,-1,1,1,1,1,1]
        [--left-gripper-state on|off] [--controller-state unconfigured|inactive|active] [--refuse-deactivate SIDE ...]

Everything is served on the topic / service / action names of the site YAML:

- cameras: raw `bgra8` head at 1280x720 (20 Hz) and `rgb8` wrists at 640x480 (30 Hz), synthetic moving gradients.
  With `--episode-dir` (the raw Munich dataset) the frames of episode `--episode` are replayed in a loop at their
  native size instead (the dataset's head video is 336x188 = `train_res`, so the client skips its resize, as in
  training); `--still` repeats the first frame of each video. Decoding uses OpenCV (AV1 works with the image's
  `python3-opencv`); the episode's first `observation.state` row (arm joints, knuckles) needs `pyarrow`, which the
  client image does not ship: `pip3 install pyarrow` in the container first. Without `--episode-dir` the arms start
  at `arms.<side>.start_pose` with both grippers open.
- measured joints of both arms (the site's `fr3v2` joint names, 200 Hz). While the fake `joint_impedance_controller`
  of an arm is active, the arm tracks the command topic with a first-order lag (`--tau` s; 0.4 makes a slow
  controller that exercises the client's tracking leash); otherwise it stands still. `--command-mode relative`
  reads commands as GELLO-style offsets: target = q_act + signs * (msg - msg_at_activation), where q_act is the
  measured pose at activation and msg_at_activation the first command received after it (`--signs`, for both arms).
- gripper knuckle state (50 Hz) following the width commands: knuckle = 0.8 * (1 - width); `--left-gripper-state off`
  never publishes the left one (Munich has no left gripper state).
- per arm, `<controller_manager>/list_controllers` and `/switch_controller` (STRICT: activating needs `inactive`,
  deactivating needs `active`); `--controller-state` sets the impedance controllers' initial state;
  `--refuse-deactivate SIDE` (repeatable) makes that arm's switch_controller refuse every deactivation (ok false,
  state unchanged), like a controller_manager that cannot stop the controller.
- per arm, the franka PTP action (aborted while the impedance controller is active; ramps each joint at its
  `maximum_joint_velocities`), the spine MoveAbsolute action (ramps at the goal velocity) and the spine GetPosition
  service (starting at `spine.height`).

`--events PATH` appends one JSON line per command (with the sender's `frame_id`, header `stamp` and the measured pose
at arrival), gripper command, switch, PTP / spine goal and result, each with `mono` (time.monotonic(), comparable
across processes on one host) and `wall`. A `ready` line is written once everything is up.
"""

import argparse
import array
from collections.abc import Sequence
import dataclasses
import itertools
import json
import logging
import math
from pathlib import Path
import signal
import sys
import threading
import time

from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from controller_manager_msgs.srv import SwitchController
from franka_msgs.action import PTPMotion
from franka_msgs.msg import TargetStatus
from franka_spine_msgs.action import MoveAbsolute
from franka_spine_msgs.srv import GetPosition
import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from robot_client import layout
from robot_client.config import ARM_NAMES
from robot_client.config import N_JOINTS
from robot_client.config import SiteConfig
from robot_client.config import load_site_config

logger = logging.getLogger("fake_robot")

# camera -> (width, height, encoding, rate Hz) of the synthetic frames
CAMERAS = {
    "head": (1280, 720, "bgra8", 20.0),
    "wrist_left": (640, 480, "rgb8", 30.0),
    "wrist_right": (640, 480, "rgb8", 30.0),
}
JOINT_HZ = 200.0
GRIPPER_EVERY = 4  # gripper state at JOINT_HZ / 4 = 50 Hz
KNUCKLE_CLOSED = 0.8  # knuckle angle (rad) of a fully closed gripper (width 0)
SYNTHETIC_FRAMES = 10  # distinct synthetic frames per camera, cycled
BROADCASTER = "franka_robot_state_broadcaster"
CONTROLLER_STATES = ("unconfigured", "inactive", "active")
FEEDBACK_PERIOD_S = 0.1
GOAL_POLL_S = 0.005
PTP_DEFAULT_VEL = 0.3  # rad/s for a joint whose maximum velocity is missing or not positive
POSITION_EPS = 1e-4


class EventLog:
    """JSON lines with monotonic and wall time; a no-op without a path. Thread-safe."""

    def __init__(self, path: str | None):
        self._file = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", buffering=1)  # noqa: SIM115 - closed in close()
        self._lock = threading.Lock()

    def write(self, kind: str, **fields) -> None:
        if self._file is None:
            return
        line = json.dumps({"mono": time.monotonic(), "wall": time.time(), "kind": kind, **fields})
        with self._lock:
            self._file.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


@dataclasses.dataclass(eq=False)
class Arm:
    side: str
    joint_names: tuple[str, ...]
    command_names: tuple[str, ...]
    controller: str
    q: np.ndarray
    controller_state: str
    width: float = 1.0
    cmd: np.ndarray | None = None  # latest command (joint order of `joint_names`)
    q_act: np.ndarray | None = None  # measured pose at activation
    cmd_act: np.ndarray | None = None  # first command after activation (relative mode reference)
    ptp_goal: np.ndarray | None = None
    ptp_vel: np.ndarray | None = None

    @property
    def knuckle(self) -> float:
        return KNUCKLE_CLOSED * (1.0 - min(1.0, max(0.0, self.width)))


@dataclasses.dataclass
class EpisodeStart:
    q: dict[str, np.ndarray]
    width: dict[str, float]


# ----------------------------------------------------------------------------------------------------
# Camera frames
# ----------------------------------------------------------------------------------------------------


def _synthetic_frames(name: str, width: int, height: int, encoding: str) -> list[array.array]:
    """SYNTHETIC_FRAMES gradients whose blue level and a vertical bar move from frame to frame."""
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    tint = {"head": 0, "wrist_left": 85, "wrist_right": 170}[name]
    frames = []
    for k in range(SYNTHETIC_FRAMES):
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.broadcast_to(x, (height, width)).astype(np.uint8)
        rgb[:, :, 1] = np.broadcast_to(y, (height, width)).astype(np.uint8)
        rgb[:, :, 2] = (tint + 25 * k) % 256
        bar = (k * width) // SYNTHETIC_FRAMES
        rgb[:, bar : bar + max(1, width // 40)] = 255
        frames.append(_encode(rgb, encoding))
    return frames


def _encode(rgb: np.ndarray, encoding: str) -> array.array:
    if encoding == "bgra8":
        alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
        data = np.concatenate([rgb[:, :, ::-1], alpha], axis=2)
    elif encoding == "rgb8":
        data = rgb
    else:  # pragma: no cover - CAMERAS only uses the two above
        raise ValueError(f"unsupported encoding {encoding}")
    # array.array('B') is what rclpy stores for uint8[]; building it from bytes is a memcpy.
    return array.array("B", np.ascontiguousarray(data).tobytes())


class VideoFrames:
    """Frames of one episode video in a loop (or its first frame forever), RGB uint8, decoded on demand."""

    def __init__(self, path: Path, *, still: bool):
        import cv2

        self._cv2 = cv2
        self.path = path
        self.still = still
        self._cap = None
        self._first = self._read_first()
        self.height, self.width = self._first.shape[:2]

    def _open(self):
        cap = self._cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.path}")
        return cap

    def _read_first(self) -> np.ndarray:
        cap = self._open()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot decode a frame of {self.path}")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def next(self) -> np.ndarray:
        if self.still:
            return self._first
        if self._cap is None:
            self._cap = self._open()
        ok, frame = self._cap.read()
        if not ok:  # end of the video: loop
            self._cap.release()
            self._cap = self._open()
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(f"cannot decode {self.path} again")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)


def load_episode_start(episode_dir: Path, episode: int) -> EpisodeStart:
    """Arm joints and gripper widths of the episode's first `observation.state` row (munich16 layout)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "--episode-dir needs pyarrow for the episode's initial joint state (pip3 install pyarrow)"
        ) from e
    path = episode_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    row = pq.read_table(path, columns=["observation.state"]).column(0)[0].as_py()
    state = np.asarray(row, dtype=np.float64)[list(layout.DATASET_STATE_INDICES)]
    q = {side: state[layout.ARM_SLICES[side]].copy() for side in ARM_NAMES}
    width = {side: 1.0 - float(state[layout.GRIPPER_INDEX[side]]) / KNUCKLE_CLOSED for side in ARM_NAMES}
    return EpisodeStart(q, width)


# ----------------------------------------------------------------------------------------------------
# The robot
# ----------------------------------------------------------------------------------------------------


class FakeRobot(Node):
    def __init__(
        self,
        cfg: SiteConfig,
        events: EventLog,
        *,
        tau: float,
        command_mode: str,
        signs: np.ndarray,
        left_gripper_state: bool,
        controller_state: str,
        refuse_deactivate: Sequence[str] = (),
        episode_dir: Path | None = None,
        episode: int = 0,
        still: bool = False,
    ):
        super().__init__("fake_robot")
        self.cfg = cfg
        self.events = events
        self.tau = tau
        self.command_mode = command_mode
        self.signs = signs
        self.refuse_deactivate = frozenset(refuse_deactivate)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        group = ReentrantCallbackGroup()

        start = load_episode_start(episode_dir, episode) if episode_dir is not None else None
        self.arms: dict[str, Arm] = {}
        for side, arm_cfg in cfg.arms.items():
            q = np.asarray(arm_cfg.start_pose, dtype=np.float64) if start is None else start.q[side]
            self.arms[side] = Arm(
                side=side,
                joint_names=tuple(arm_cfg.joint_names),
                command_names=tuple(arm_cfg.command_joint_names or arm_cfg.joint_names),
                controller=arm_cfg.controller,
                q=q.copy(),
                controller_state=controller_state,
                width=1.0 if start is None else start.width[side],
            )
        self.spine_position = cfg.spine.height
        self._spine_goal: tuple[float, float] | None = None  # (position, velocity)

        # Measured joints and gripper state.
        self._joint_pubs = {
            side: self.create_publisher(JointState, a.measured_topic, 10) for side, a in cfg.arms.items()
        }
        self._gripper_pubs = {
            side: self.create_publisher(JointState, a.gripper.state_topic, 10)
            for side, a in cfg.arms.items()
            if side != "left" or left_gripper_state
        }

        # Commands.
        for side, arm_cfg in cfg.arms.items():
            self.create_subscription(
                JointState, arm_cfg.command_topic, lambda m, s=side: self._on_command(s, m), 10, callback_group=group
            )
            self.create_subscription(
                Float32,
                arm_cfg.gripper.command_topic,
                lambda m, s=side: self._on_gripper(s, m),
                10,
                callback_group=group,
            )

        # controller_manager services, PTP and spine actions, spine position.
        self._action_servers = []
        for side, arm_cfg in cfg.arms.items():
            ns = arm_cfg.controller_manager.rstrip("/")
            self.create_service(
                ListControllers, f"{ns}/list_controllers", lambda q, r, s=side: self._list(s, r), callback_group=group
            )
            self.create_service(
                SwitchController,
                f"{ns}/switch_controller",
                lambda q, r, s=side: self._switch(s, q, r),
                callback_group=group,
            )
            self._action_servers.append(
                ActionServer(
                    self,
                    PTPMotion,
                    arm_cfg.ptp_action,
                    lambda h, s=side: self._ptp(s, h),
                    cancel_callback=lambda _: CancelResponse.ACCEPT,
                    callback_group=group,
                )
            )
        self._action_servers.append(
            ActionServer(
                self,
                MoveAbsolute,
                cfg.spine.action,
                self._spine,
                cancel_callback=lambda _: CancelResponse.ACCEPT,
                callback_group=group,
            )
        )
        self.create_service(GetPosition, cfg.spine.get_position_service, self._get_position, callback_group=group)

        # Cameras.
        self._camera_sources = {}
        for name, cam in cfg.cameras.items():
            width, height, encoding, hz = CAMERAS[name]
            pub = self.create_publisher(Image, cam.topic, qos_profile_sensor_data)
            if episode_dir is None:
                source = itertools.cycle(_synthetic_frames(name, width, height, encoding)).__next__
                size = (width, height)
            else:
                video = VideoFrames(
                    episode_dir / "videos" / "chunk-000" / f"observation.images.{name}" / f"episode_{episode:06d}.mp4",
                    still=still,
                )
                source = lambda video=video, encoding=encoding: _encode(video.next(), encoding)  # noqa: E731
                size = (video.width, video.height)
            self._camera_sources[name] = (pub, source, size, encoding, hz)

    # ---- lifecycle ----

    def start(self) -> None:
        self._spawn(self._physics_loop, "physics")
        for name in self._camera_sources:
            self._spawn(lambda name=name: self._camera_loop(name), f"camera-{name}")

    def _spawn(self, target, name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(2.0)

    def _loop(self, hz: float, tick) -> None:
        """Call `tick(dt)` at `hz` on absolute deadlines until stopped (re-anchored after an overrun)."""
        period = 1.0 / hz
        last = deadline = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                tick(now - last)
            except Exception:
                if self._stop.is_set():
                    return
                logger.exception("%s tick failed", threading.current_thread().name)
            last = now
            deadline += period
            now = time.monotonic()
            if deadline < now - period:
                deadline = now
            self._stop.wait(max(0.0, deadline - now))

    # ---- physics: arms, PTP, spine; measured and gripper state ----

    def _physics_loop(self) -> None:
        count = [0]

        def tick(dt: float) -> None:
            with self._lock:
                for arm in self.arms.values():
                    self._advance(arm, dt)
                self._advance_spine(dt)
                joints = {side: (arm.joint_names, arm.q.tolist()) for side, arm in self.arms.items()}
                knuckles = {side: arm.knuckle for side, arm in self.arms.items()}
            stamp = self.get_clock().now().to_msg()
            for side, (names, q) in joints.items():
                msg = JointState(name=list(names), position=q, velocity=[0.0] * len(q), effort=[0.0] * len(q))
                msg.header.stamp = stamp
                self._joint_pubs[side].publish(msg)
            if count[0] % GRIPPER_EVERY == 0:
                for side, pub in self._gripper_pubs.items():
                    msg = JointState(name=[self.cfg.arms[side].gripper.joint], position=[knuckles[side]])
                    msg.header.stamp = stamp
                    pub.publish(msg)
            count[0] += 1

        self._loop(JOINT_HZ, tick)

    def _advance(self, arm: Arm, dt: float) -> None:
        if arm.ptp_goal is not None:
            step = arm.ptp_vel * dt
            arm.q += np.clip(arm.ptp_goal - arm.q, -step, step)
            return
        if arm.controller_state != "active":
            return
        target = self._controller_target(arm)
        if target is not None:
            arm.q += (1.0 - math.exp(-dt / self.tau)) * (target - arm.q)

    def _controller_target(self, arm: Arm) -> np.ndarray | None:
        if arm.cmd is None:
            return None
        if self.command_mode == "absolute":
            return arm.cmd
        if arm.cmd_act is None:
            return None
        return arm.q_act + self.signs * (arm.cmd - arm.cmd_act)

    def _advance_spine(self, dt: float) -> None:
        if self._spine_goal is None:
            return
        goal, velocity = self._spine_goal
        step = velocity * dt
        self.spine_position += min(step, max(-step, goal - self.spine_position))

    # ---- cameras ----

    def _camera_loop(self, name: str) -> None:
        pub, source, (width, height), encoding, hz = self._camera_sources[name]
        channels = 4 if encoding == "bgra8" else 3
        frame_id = f"{name}_camera"

        def tick(_dt: float) -> None:
            msg = Image(height=height, width=width, encoding=encoding, step=width * channels)
            msg.data = source()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = frame_id
            pub.publish(msg)

        self._loop(hz, tick)

    # ---- commands ----

    def _on_command(self, side: str, msg: JointState) -> None:
        arm = self.arms[side]
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        by_name = dict(zip(msg.name, msg.position, strict=False))
        missing = [n for n in arm.command_names if n not in by_name]
        if missing or len(msg.name) != len(msg.position):
            self.events.write(
                "command_rejected", side=side, frame_id=msg.header.frame_id, names=list(msg.name), missing=missing
            )
            return
        position = np.array([by_name[n] for n in arm.command_names], dtype=np.float64)
        with self._lock:
            measured = arm.q.tolist()
            active = arm.controller_state == "active"
            arm.cmd = position
            if active and arm.cmd_act is None:
                arm.cmd_act = position.copy()
        self.events.write(
            "command",
            side=side,
            frame_id=msg.header.frame_id,
            stamp=stamp,
            position=position.tolist(),
            measured=measured,
            active=active,
        )

    def _on_gripper(self, side: str, msg: Float32) -> None:
        with self._lock:
            self.arms[side].width = float(msg.data)
        self.events.write("gripper", side=side, width=float(msg.data))

    # ---- controller_manager ----

    def _list(self, side: str, response):
        arm = self.arms[side]
        with self._lock:
            state = arm.controller_state
        response.controller = [
            ControllerState(
                name=arm.controller, state=state, type="franka_example_controllers/JointImpedanceController"
            ),
            ControllerState(
                name=BROADCASTER, state="active", type="franka_robot_state_broadcaster/FrankaRobotStateBroadcaster"
            ),
        ]
        return response

    def _switch(self, side: str, request, response):
        arm = self.arms[side]
        activate, deactivate = list(request.activate_controllers), list(request.deactivate_controllers)
        strict = request.strictness == SwitchController.Request.STRICT
        with self._lock:
            problems = []
            for name in activate:
                if name != arm.controller:
                    problems.append(f"{name} is not loaded")
                elif arm.controller_state != "inactive":
                    problems.append(f"{name} is {arm.controller_state}, cannot activate")
                elif arm.ptp_goal is not None:
                    problems.append(f"a PTP motion is running, cannot activate {name}")
            for name in deactivate:
                if name != arm.controller:
                    problems.append(f"{name} is not loaded")
                elif arm.controller_state != "active":
                    problems.append(f"{name} is {arm.controller_state}, cannot deactivate")
            refused = bool(deactivate) and side in self.refuse_deactivate
            if refused:
                problems.append("deactivation refused (--refuse-deactivate)")
            ok = not (strict and problems) and not refused
            if ok:  # BEST_EFFORT: apply what is possible
                if arm.controller in deactivate and arm.controller_state == "active":
                    arm.controller_state = "inactive"
                    arm.q_act = arm.cmd_act = None
                elif arm.controller in activate and arm.controller_state == "inactive" and arm.ptp_goal is None:
                    arm.controller_state = "active"
                    arm.q_act, arm.cmd_act = arm.q.copy(), None
            state = arm.controller_state
        self.events.write(
            "switch", side=side, activate=activate, deactivate=deactivate, ok=ok, problems=problems, state=state
        )
        response.ok = ok
        return response

    # ---- PTP ----

    def _ptp(self, side: str, goal_handle):
        arm = self.arms[side]
        request = goal_handle.request
        goal = np.asarray(request.goal_joint_configuration, dtype=np.float64)
        vel = np.asarray(request.maximum_joint_velocities, dtype=np.float64)
        result = PTPMotion.Result()
        self.events.write("ptp_goal", side=side, goal=goal.tolist(), max_vel=vel.tolist())
        error = None
        if goal.shape != (N_JOINTS,) or not np.all(np.isfinite(goal)):
            error = f"goal_joint_configuration needs {N_JOINTS} finite values, got {goal.tolist()}"
        with self._lock:
            if error is None and arm.controller_state == "active":
                error = f"{arm.controller} is active; deactivate it before a PTP motion"
            if error is None:
                if vel.shape != (N_JOINTS,):
                    vel = np.full(N_JOINTS, PTP_DEFAULT_VEL)
                arm.ptp_vel = np.where(vel > 0, vel, PTP_DEFAULT_VEL)
                arm.ptp_goal = goal
        if error is not None:
            result.target_status.status = TargetStatus.ABORTED
            result.error_message = error
            goal_handle.abort()
            self.events.write("ptp_result", side=side, reached=False, error=error)
            return result

        def at_goal() -> bool:
            return float(np.max(np.abs(arm.q - goal))) <= POSITION_EPS

        canceled = self._run_goal(goal_handle, at_goal, lambda: self._ptp_feedback())
        with self._lock:
            arm.ptp_goal = arm.ptp_vel = None
            reached = at_goal()
        if canceled:
            result.target_status.status = TargetStatus.ABORTED
            result.error_message = "canceled"
            _end_canceled(goal_handle)
        else:
            result.target_status.status = TargetStatus.TARGET_REACHED
            goal_handle.succeed()
        self.events.write("ptp_result", side=side, reached=reached and not canceled, canceled=canceled)
        return result

    def _ptp_feedback(self):
        feedback = PTPMotion.Feedback()
        feedback.target_status.status = TargetStatus.EXECUTING
        return feedback

    # ---- spine ----

    def _spine(self, goal_handle):
        request = goal_handle.request
        self.events.write("spine_goal", position=request.position, velocity=request.velocity)
        result = MoveAbsolute.Result()
        if not (math.isfinite(request.position) and request.velocity > 0):
            result.success, result.error, result.stop_by = False, "position must be finite and velocity > 0", "error"
            goal_handle.abort()
            self.events.write("spine_result", success=False, error=result.error)
            return result
        with self._lock:
            self._spine_goal = (float(request.position), float(request.velocity))

        def feedback():
            return MoveAbsolute.Feedback(current_position=self.spine_position)

        canceled = self._run_goal(
            goal_handle, lambda: abs(self.spine_position - request.position) <= POSITION_EPS, feedback
        )
        with self._lock:
            self._spine_goal = None
            position = self.spine_position
        result.success = not canceled
        result.stop_by = "canceled" if canceled else "target_reached"
        if canceled:
            _end_canceled(goal_handle)
        else:
            goal_handle.succeed()
        self.events.write("spine_result", success=result.success, position=position)
        return result

    def _get_position(self, request, response):
        with self._lock:
            response.position = float(self.spine_position)
        response.success = True
        return response

    # ---- goals ----

    def _run_goal(self, goal_handle, done, make_feedback) -> bool:
        """Wait until `done()` (the physics thread moves towards the goal); True if canceled first."""
        next_feedback = 0.0
        while True:
            with self._lock:
                finished = done()
            if finished:
                return False
            if goal_handle.is_cancel_requested or self._stop.is_set():
                return True
            now = time.monotonic()
            if now >= next_feedback:
                goal_handle.publish_feedback(make_feedback())
                next_feedback = now + FEEDBACK_PERIOD_S
            time.sleep(GOAL_POLL_S)

    def describe_cameras(self) -> dict[str, str]:
        return {name: f"{w}x{h} {enc} @ {hz:g} Hz" for name, (_, _, (w, h), enc, hz) in self._camera_sources.items()}


def _end_canceled(goal_handle) -> None:
    """End a goal that stopped early: canceled if the client asked for it, else (we are shutting down) aborted."""
    if goal_handle.is_cancel_requested:
        goal_handle.canceled()
    else:
        goal_handle.abort()


# ----------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------


def _signs(text: str) -> np.ndarray:
    try:
        signs = [int(v) for v in text.split(",")]
    except ValueError:
        signs = []
    if len(signs) != N_JOINTS or any(s not in (-1, 1) for s in signs):
        raise argparse.ArgumentTypeError(f"expected {N_JOINTS} comma-separated values of 1 / -1, got {text!r}")
    return np.asarray(signs, dtype=np.float64)


def _positive(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text!r}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m robot_client.fake_robot", description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", required=True, help="site YAML (topic, service and action names, start poses)")
    parser.add_argument("--episode-dir", type=Path, default=None, help="raw Munich dataset: replay its frames")
    parser.add_argument("--episode", type=int, default=0, help="episode index in --episode-dir (default 0)")
    parser.add_argument("--still", action="store_true", help="with --episode-dir: repeat the first frame")
    parser.add_argument("--events", default=None, help="append events as JSON lines to this file")
    parser.add_argument("--tau", type=_positive, default=0.05, help="controller lag time constant in s (default 0.05)")
    parser.add_argument("--command-mode", choices=("absolute", "relative"), default="absolute")
    parser.add_argument(
        "--signs", type=_signs, default=np.ones(N_JOINTS), help="relative mode joint signs, e.g. 1,-1,1,1,1,1,1"
    )
    parser.add_argument("--left-gripper-state", choices=("on", "off"), default="on")
    parser.add_argument("--controller-state", choices=CONTROLLER_STATES, default="inactive")
    parser.add_argument(
        "--refuse-deactivate",
        action="append",
        choices=ARM_NAMES,
        default=[],
        help="this arm's switch_controller refuses every deactivation (repeatable)",
    )
    return parser.parse_args(argv)


def _interrupt(signum, _frame) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_site_config(args.site, [], {})
    signal.signal(signal.SIGTERM, _interrupt)
    events = EventLog(args.events)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    robot = None
    executor = MultiThreadedExecutor()
    try:
        robot = FakeRobot(
            cfg,
            events,
            tau=args.tau,
            command_mode=args.command_mode,
            signs=args.signs,
            left_gripper_state=args.left_gripper_state == "on",
            controller_state=args.controller_state,
            refuse_deactivate=args.refuse_deactivate,
            episode_dir=args.episode_dir,
            episode=args.episode,
            still=args.still,
        )
        executor.add_node(robot)
        robot.start()
        cameras = robot.describe_cameras()
        events.write("ready", cameras=cameras, tau=args.tau, command_mode=args.command_mode, signs=args.signs.tolist())
        logger.info("fake robot ready (cameras %s, tau %g s, %s commands)", cameras, args.tau, args.command_mode)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if robot is not None:
            robot.stop()
        executor.shutdown(timeout_sec=2.0)
        if robot is not None:
            robot.destroy_node()
        rclpy.try_shutdown()
        events.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
