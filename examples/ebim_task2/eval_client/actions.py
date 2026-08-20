"""Action dispatch helpers: 20-dim policy action -> sim command values."""

import dataclasses

import numpy as np

from eval_client.constants import ACTION_DIM
from eval_client.constants import GRIPPER_CLOSED_RAD
from eval_client.constants import PEDAL_MAX_ANGULAR
from eval_client.constants import PEDAL_MAX_LINEAR


def gripper_driver_rad(open_fraction: float) -> float:
    """Inverse of obs.gripper_open_fraction, for commanding."""
    return float((1.0 - np.clip(open_fraction, 0.0, 1.0)) * GRIPPER_CLOSED_RAD)


def pedal_token(vx: float, vy: float, wz: float, threshold: float) -> str:
    """Quantize a base twist to the bridge's pedal-token vocabulary.

    Dominant-axis selection with a dead-band: the recorded dataset only
    contains the discrete pedal twists, so nearest-token quantization is
    faithful. The angular axis is compared after normalizing by its higher
    magnitude (1.2 vs 0.5) so a full-rate yaw competes fairly with a
    full-rate strafe.
    """
    wz_scaled = wz * (PEDAL_MAX_LINEAR / PEDAL_MAX_ANGULAR)
    magnitudes = (abs(vx), abs(vy), abs(wz_scaled))
    if max(magnitudes) < threshold:
        return "NONE"
    axis = int(np.argmax(magnitudes))
    if axis == 0:
        return "FWD" if vx > 0 else "BACK"
    if axis == 1:
        return "A" if vy > 0 else "B"
    return "A+C" if wz > 0 else "B+C"


@dataclasses.dataclass(frozen=True)
class ActionCommand:
    left_arm: np.ndarray  # 7 absolute joint targets (rad)
    right_arm: np.ndarray
    left_grip_rad: float  # gripper driver-joint target (rad)
    right_grip_rad: float
    spine: float  # spine height (m)
    pedal: str  # pedal token for the base


def split_action(action, pedal_threshold: float) -> ActionCommand:
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"Bad action shape: {a.shape}, expected ({ACTION_DIM},)")
    return ActionCommand(
        left_arm=a[3:10],
        right_arm=a[10:17],
        left_grip_rad=gripper_driver_rad(a[17]),
        right_grip_rad=gripper_driver_rad(a[18]),
        spine=float(a[19]),
        pedal=pedal_token(a[0], a[1], a[2], pedal_threshold),
    )
