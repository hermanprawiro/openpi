"""Sim/policy contract constants for the EBiM task2 eval client.

Topic names and joint names are transcribed from the benchmark repo's
task2_isaacsim/config/topics.yaml and services/recording/record_task2.py
(the upstream source of truth); this client is standalone from that repo,
so any change there must be mirrored here.
"""

import dataclasses

CONTROL_HZ = 30.0
STATE_DIM = 20
ACTION_DIM = 20
CHUNK_LEN = 50
# Action layout (see actions.split_action): base twist 0-2 | left arm 3-9 |
# right arm 10-16 | grippers 17-18 | spine 19. Arm joints (rad) are the
# only dims whose step-to-step jumps are meaningful to measure; spine (m)
# is also a continuous absolute target. Base twist is quantized to pedal
# tokens and the grippers are effectively binary, so chunk merges never
# interpolate them: they always take the fresh chunk's value.
ARM_ACTION_DIMS = tuple(range(3, 17))
BLEND_ACTION_DIMS = (*ARM_ACTION_DIMS, 19)
IMAGE_SIZE = 224
OBS_STALE_S = 2.0

LEFT_JOINTS = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
SPINE_JOINT = "franka_spine_vertical_joint"
LEFT_GRIPPER_DRIVER = "left_right_finger_joint"
RIGHT_GRIPPER_DRIVER = "right_right_finger_joint"
GRIPPER_CLOSED_RAD = 0.8

# Bridge pedal rates: the dataset's base twists are exactly the discrete
# pedal values (+-0.5 m/s linear, +-1.2 rad/s yaw, or zero).
PEDAL_MAX_LINEAR = 0.5
PEDAL_MAX_ANGULAR = 1.2

# Policy-observation camera keys (the eval camera and ground-truth object
# poses must never feed the policy -- competition rule).
CAMERA_KEYS = ("head", "wrist_left", "wrist_right")

# Must match the training dataset's task string (prompt_from_task=True).
DEFAULT_PROMPT = "Pick up the thermal pad and place it on the target RAM board."


@dataclasses.dataclass(frozen=True)
class Topics:
    # /isaac/clock, deliberately NOT /clock: a leaked OmniGraph publisher
    # spams time=0.0 on /clock (see topics.yaml in the benchmark repo).
    clock: str = "/isaac/clock"
    joint_states_full: str = "/isaac/joint_states_full"  # requires sim --record
    odom: str = "/isaac/odom"  # requires sim --record
    head_image: str = "/isaac/head_camera/image_raw"
    left_wrist_image: str = "/isaac/left_wrist_camera/image_raw"
    right_wrist_image: str = "/isaac/right_wrist_camera/image_raw"
    left_arm_cmd: str = "/isaac/left_joint_commands"
    right_arm_cmd: str = "/isaac/right_joint_commands"
    left_grip_cmd: str = "/isaac/left_robotiq_joint_commands"
    right_grip_cmd: str = "/isaac/right_robotiq_joint_commands"
    spine_cmd: str = "/isaac/spine_joint_commands"
    pedal_state: str = "/pedal/state"
    scene_reset: str = "/isaac/task2/scene_reset"
    scene_reset_request: str = "/isaac/task2/scene_reset_request"

    def camera_images(self) -> dict[str, str]:
        return {
            "head": self.head_image,
            "wrist_left": self.left_wrist_image,
            "wrist_right": self.right_wrist_image,
        }
