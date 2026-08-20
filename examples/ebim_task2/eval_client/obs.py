"""Observation building: sim joint states + images -> policy obs dict.

Pure numpy/stdlib except make_observation, which lazily imports
openpi_client.image_tools (the same letterbox resize the training
converter used -- matching it matters).
"""

from collections.abc import Iterator
import math

import numpy as np

from eval_client.constants import GRIPPER_CLOSED_RAD
from eval_client.constants import IMAGE_SIZE
from eval_client.constants import LEFT_GRIPPER_DRIVER
from eval_client.constants import LEFT_JOINTS
from eval_client.constants import RIGHT_GRIPPER_DRIVER
from eval_client.constants import RIGHT_JOINTS
from eval_client.constants import SPINE_JOINT
from eval_client.constants import STATE_DIM


def candidate_joint_names(name: str) -> Iterator[str]:
    """Robot-USD joint-name variants, mirroring the sim bridge's resolver."""
    yield name
    if "fr3v2_joint" in name:
        yield name.replace("fr3v2_joint", "fr3v2_1_joint")
    if name == "left_right_finger_joint":
        yield "left_fr3v2_finger_joint1"
    if name == "right_right_finger_joint":
        yield "right_fr3v2_finger_joint1"


def resolve_joint(joint_map: dict, name: str, default: float = math.nan) -> float:
    for candidate in candidate_joint_names(name):
        value = joint_map.get(candidate)
        if value is not None and math.isfinite(value):
            return float(value)
    return default


def gripper_open_fraction(driver_position_rad: float) -> float:
    if not math.isfinite(driver_position_rad):
        return math.nan
    return float(np.clip(1.0 - driver_position_rad / GRIPPER_CLOSED_RAD, 0.0, 1.0))


def build_state(joint_map: dict, base_twist) -> np.ndarray:
    """20-dim policy state: [base vel 3 | left 7 | right 7 | grips 2 | spine].

    Matches the layout produced by convert_task2_data_to_lerobot.py.
    """
    state = np.empty(STATE_DIM, dtype=np.float32)
    state[0:3] = base_twist
    for i, name in enumerate(LEFT_JOINTS):
        state[3 + i] = resolve_joint(joint_map, name)
    for i, name in enumerate(RIGHT_JOINTS):
        state[10 + i] = resolve_joint(joint_map, name)
    state[17] = gripper_open_fraction(resolve_joint(joint_map, LEFT_GRIPPER_DRIVER))
    state[18] = gripper_open_fraction(resolve_joint(joint_map, RIGHT_GRIPPER_DRIVER))
    state[19] = resolve_joint(joint_map, SPINE_JOINT)
    return state


def decode_image(data: bytes, height: int, width: int, step: int, encoding: str) -> np.ndarray:
    """sensor_msgs/Image payload -> HxWx3 uint8 RGB (step-aware reshape)."""
    channels = 3
    array = np.frombuffer(data, dtype=np.uint8)
    array = array.reshape(height, step)[:, : width * channels]
    array = array.reshape(height, width, channels)
    if encoding.lower() in ("bgr8", "bgra8"):
        array = array[:, :, ::-1]
    return np.ascontiguousarray(array)


def make_observation(state: np.ndarray, images: dict, prompt: str) -> dict:
    """Policy server obs dict (matches openpi/examples/ebim_task2)."""
    from openpi_client import image_tools

    observation = {
        "observation/state": state,
        "prompt": prompt,
    }
    for key, obs_key in (
        ("head", "observation/image"),
        ("wrist_left", "observation/left_wrist_image"),
        ("wrist_right", "observation/right_wrist_image"),
    ):
        observation[obs_key] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images[key], IMAGE_SIZE, IMAGE_SIZE)
        )
    return observation
