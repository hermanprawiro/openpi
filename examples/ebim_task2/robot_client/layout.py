"""The munich16 state/action layout shared by the converter, the client and the training config.

Pure numpy; no ROS or openpi imports so it can be used anywhere.

State  (16): [L 7 measured rad, L knuckle rad, R 7 measured rad, R knuckle rad]
Action (16): [L 7 target rad,   L width %,     R 7 target rad,   R width %]
Raw dataset values, no unit conversion.
"""

import numpy as np

LAYOUT_ID = "munich16"
STATE_DIM = 16
ACTION_DIM = 16
CONTROL_HZ = 20
ACTION_HORIZON = 50
PROMPT = "Pick up the thermal pad and place it on the target RAM board"
# (W, H) the head camera is resized to (if its live size differs) before letterboxing to 224x224.
# Single source for the converter and the site YAML (`cameras.head.train_res`).
HEAD_TRAIN_RES = (336, 188)

ARM_SLICES = {"left": slice(0, 7), "right": slice(8, 15)}
GRIPPER_INDEX = {"left": 7, "right": 15}
ARM_DIMS = (*range(7), *range(8, 15))

# Indices into the Munich 42-dim observation.state / 17-dim action of the raw dataset.
DATASET_STATE_INDICES = (*range(7), 7, *range(21, 28), 28)
DATASET_ACTION_INDICES = tuple(range(16))

# What a munich16 policy server advertises (and what the client requires of it).
POLICY_METADATA: dict = {
    "layout": LAYOUT_ID,
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "control_hz": CONTROL_HZ,
    "action_horizon": ACTION_HORIZON,
    "prompt": PROMPT,
}


class LayoutMismatchError(ValueError):
    """The policy server does not speak the munich16 layout the client was built for."""


def validate_server_metadata(meta: dict, prompt: str) -> None:
    """Raise LayoutMismatchError unless `meta` describes a munich16 policy that expects `prompt`."""
    if "layout" not in meta:
        raise LayoutMismatchError(f"server is not a {LAYOUT_ID} policy (config={meta.get('config')})")
    problems = []
    for key, expected in POLICY_METADATA.items():
        if key == "prompt":
            continue
        got = meta.get(key)
        if got != expected:
            problems.append(f"{key}: expected {expected!r}, got {got!r}")
    if meta.get("prompt") != prompt:
        problems.append(f"prompt: server has {meta.get('prompt')!r}, client uses {prompt!r}")
    if problems:
        raise LayoutMismatchError(
            f"server metadata does not match the {LAYOUT_ID} layout (config={meta.get('config')}): "
            + "; ".join(problems)
        )


def validate_chunk(chunk) -> np.ndarray:
    """Return `chunk` as float64 (T, 16); raise ValueError if it is not a usable munich16 action chunk."""
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"action chunk must be 2-D (T, {ACTION_DIM}), got ndim={arr.ndim}")
    if arr.shape[1] != ACTION_DIM:
        raise ValueError(f"action chunk width {arr.shape[1]} != {ACTION_DIM}: wrong-layout server?")
    if arr.shape[0] < 1:
        raise ValueError("action chunk has no rows")
    if not np.isfinite(arr).all():
        raise ValueError("action chunk contains non-finite values")
    return arr
