"""ROS 2 node: latest-value observation cache + command publishers.

The only module (besides main) that imports ROS. All cached values are
guarded by one lock; accessors return copies. Callbacks run on the rclpy
spin thread, the episode loop reads via the accessors.
"""

import json
import math
import threading
import time

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from eval_client import actions as actions_mod
from eval_client import obs as obs_mod
from eval_client.constants import CAMERA_KEYS
from eval_client.constants import LEFT_GRIPPER_DRIVER
from eval_client.constants import LEFT_JOINTS
from eval_client.constants import OBS_STALE_S
from eval_client.constants import RIGHT_GRIPPER_DRIVER
from eval_client.constants import RIGHT_JOINTS
from eval_client.constants import SPINE_JOINT
from eval_client.constants import Topics


class EvalClientNode(Node):
    def __init__(self, topics: Topics | None = None, qos_depth: int = 10):
        super().__init__("ebim_task2_eval_client")
        topics = topics or Topics()

        self._lock = threading.Lock()
        self._sim_time = None
        self._joint_states = {}
        self._base_twist = (0.0, 0.0, 0.0)
        self._images = dict.fromkeys(CAMERA_KEYS)
        self._last_msg_wall = {}
        self._reset_events = 0
        self._last_reset_wall = -math.inf

        self.create_subscription(Clock, topics.clock, self._on_clock, qos_depth)
        self.create_subscription(JointState, topics.joint_states_full, self._on_joint_states, qos_depth)
        self.create_subscription(Odometry, topics.odom, self._on_odom, qos_depth)
        for key, topic in topics.camera_images().items():
            self.create_subscription(
                Image,
                topic,
                lambda msg, key=key: self._on_image(key, msg),
                qos_profile_sensor_data,
            )
        self.create_subscription(String, topics.scene_reset, self._on_scene_reset, qos_depth)

        self._reset_request_pub = self.create_publisher(String, topics.scene_reset_request, qos_depth)
        self._pedal_pub = self.create_publisher(String, topics.pedal_state, qos_depth)
        self._left_arm_pub = self.create_publisher(JointState, topics.left_arm_cmd, qos_depth)
        self._right_arm_pub = self.create_publisher(JointState, topics.right_arm_cmd, qos_depth)
        self._left_grip_pub = self.create_publisher(JointState, topics.left_grip_cmd, qos_depth)
        self._right_grip_pub = self.create_publisher(JointState, topics.right_grip_cmd, qos_depth)
        self._spine_pub = self.create_publisher(JointState, topics.spine_cmd, qos_depth)

    # -- subscription callbacks (rclpy spin thread) ----------------------
    def _mark(self, key: str) -> None:
        self._last_msg_wall[key] = time.monotonic()

    def _on_clock(self, msg) -> None:
        with self._lock:
            self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
            self._mark("clock")

    def _on_joint_states(self, msg) -> None:
        with self._lock:
            for idx, name in enumerate(msg.name):
                if idx < len(msg.position):
                    self._joint_states[name] = float(msg.position[idx])
            self._mark("joint_states")

    def _on_odom(self, msg) -> None:
        twist = msg.twist.twist
        with self._lock:
            self._base_twist = (
                float(twist.linear.x),
                float(twist.linear.y),
                float(twist.angular.z),
            )
            self._mark("odom")

    def _on_image(self, key: str, msg) -> None:
        array = obs_mod.decode_image(bytes(msg.data), msg.height, msg.width, msg.step, msg.encoding)
        with self._lock:
            self._images[key] = array
            self._mark(f"image_{key}")

    def _on_scene_reset(self, msg) -> None:
        with self._lock:
            self._reset_events += 1
            self._last_reset_wall = time.monotonic()

    # -- accessors (episode-loop thread) ---------------------------------
    def required_streams(self) -> list:
        return ["clock", "joint_states", "odom"] + [f"image_{key}" for key in CAMERA_KEYS]

    def sim_now(self):
        with self._lock:
            return self._sim_time

    def streams_fresh(self, max_age_s: float = OBS_STALE_S) -> bool:
        now = time.monotonic()
        with self._lock:
            return all(now - self._last_msg_wall.get(key, -math.inf) < max_age_s for key in self.required_streams())

    def streams_fresh_since(self, wall_time: float) -> bool:
        """True once every stream delivered a message after wall_time
        (used post-reset: rendering pauses during the scene reset, so
        merely-recent pre-reset frames are not enough)."""
        with self._lock:
            return all(self._last_msg_wall.get(key, -math.inf) > wall_time for key in self.required_streams())

    def missing_streams(self) -> list:
        now = time.monotonic()
        with self._lock:
            return [
                key for key in self.required_streams() if now - self._last_msg_wall.get(key, -math.inf) >= OBS_STALE_S
            ]

    def snapshot(self):
        """(sim_time, state, images dict) -- None if incomplete or the
        state is non-finite (e.g. a joint missing from joint_states)."""
        with self._lock:
            if self._sim_time is None or any(self._images[key] is None for key in CAMERA_KEYS):
                return None
            state = obs_mod.build_state(dict(self._joint_states), self._base_twist)
            images = {key: self._images[key] for key in CAMERA_KEYS}
            sim_time = self._sim_time
        if not np.all(np.isfinite(state)):
            return None
        return sim_time, state, images

    def reset_count(self) -> int:
        with self._lock:
            return self._reset_events

    def last_reset_event_wall(self) -> float:
        with self._lock:
            return self._last_reset_wall

    def clock_restarted_after(self, event_wall: float) -> bool:
        """True once a clock message arrived after event_wall. The
        scene_reset event is published after world.reset() completes, so
        post-event clock messages are post-reset by construction."""
        with self._lock:
            return self._sim_time is not None and self._last_msg_wall.get("clock", -math.inf) > event_wall

    def reset_request_has_subscriber(self) -> bool:
        """DDS discovery-race guard: a request published before the
        sim-side subscription is matched is silently dropped."""
        return self._reset_request_pub.get_subscription_count() > 0

    # -- publishers -------------------------------------------------------
    def request_reset(self) -> None:
        msg = String()
        msg.data = json.dumps({"source": "ebim_task2_eval_client"})
        self._reset_request_pub.publish(msg)

    def _joint_msg(self, names, positions) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        return msg

    def publish_action(self, action, pedal_threshold: float) -> None:
        cmd = actions_mod.split_action(action, pedal_threshold)
        self._left_arm_pub.publish(self._joint_msg(LEFT_JOINTS, cmd.left_arm))
        self._right_arm_pub.publish(self._joint_msg(RIGHT_JOINTS, cmd.right_arm))
        self._left_grip_pub.publish(self._joint_msg([LEFT_GRIPPER_DRIVER], [cmd.left_grip_rad]))
        self._right_grip_pub.publish(self._joint_msg([RIGHT_GRIPPER_DRIVER], [cmd.right_grip_rad]))
        self._spine_pub.publish(self._joint_msg([SPINE_JOINT], [cmd.spine]))
        pedal = String()
        pedal.data = cmd.pedal
        self._pedal_pub.publish(pedal)

    def publish_stop(self) -> None:
        """Zero the base; arms hold their last target (the sim bridge has
        a 1 s command watchdog)."""
        pedal = String()
        pedal.data = "NONE"
        self._pedal_pub.publish(pedal)


def spin_in_thread(node: EvalClientNode) -> threading.Thread:
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return thread
