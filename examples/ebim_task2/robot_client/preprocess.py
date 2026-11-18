"""Observation preprocessing for the policy. Must match what the converter bakes into the dataset.

The ROS node decodes sensor payloads (`decode_raw_image` / `decode_compressed_image`, `map_joints`), collects
the latest samples in a `Snapshot`, and `build_observation` turns that into the exact dict the policy server
expects (keys of `EbimTask2Inputs`). Pure numpy/PIL; no ROS imports.
"""

from collections.abc import Mapping, Sequence
import dataclasses

import numpy as np
from openpi_client import image_tools
from PIL import Image

from robot_client import layout
from robot_client.config import CameraConfig

POLICY_IMAGE_SIZE = 224

# Observation key per camera name (the keys EbimTask2Inputs reads).
IMAGE_KEYS = {
    "head": "observation/image",
    "wrist_left": "observation/left_wrist_image",
    "wrist_right": "observation/right_wrist_image",
}

# encoding -> (channels per pixel, indices of the R, G, B channels)
_RAW_ENCODINGS = {
    "rgb8": (3, (0, 1, 2)),
    "bgr8": (3, (2, 1, 0)),
    "rgba8": (4, (0, 1, 2)),
    "bgra8": (4, (2, 1, 0)),
}

# Slack added around the training [q01, q99] before a state dim is reported out of range.
_RANGE_MARGIN = 0.10


class UnsupportedEncodingError(ValueError):
    """A sensor_msgs/Image encoding the client does not decode."""

    def __init__(self, encoding: str):
        self.encoding = encoding
        super().__init__(f"unsupported image encoding {encoding!r} (supported: {', '.join(_RAW_ENCODINGS)})")


class MissingJointError(ValueError):
    """A joint the client needs is absent from a JointState message."""

    def __init__(self, missing: Sequence[str], seen: Sequence[str]):
        self.missing = list(missing)
        self.seen = list(seen)
        super().__init__(f"joint(s) {self.missing} not in the message; names seen: {self.seen}")


class StateError(ValueError):
    """The snapshot cannot be turned into a trustworthy 16-dim state."""


# ----------------------------------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------------------------------


def decode_raw_image(data: bytes, height: int, width: int, step: int, encoding: str) -> np.ndarray:
    """sensor_msgs/Image payload -> HxWx3 uint8 RGB (step-aware; rgb8/bgr8/rgba8/bgra8, alpha dropped)."""
    spec = _RAW_ENCODINGS.get(encoding.lower())
    if spec is None:
        raise UnsupportedEncodingError(encoding)
    channels, order = spec
    if step < width * channels:
        raise ValueError(f"image step {step} is smaller than width {width} x {channels} channels ({encoding})")
    if len(data) != height * step:
        raise ValueError(f"image has {len(data)} bytes, expected height {height} x step {step} = {height * step}")
    array = np.frombuffer(data, dtype=np.uint8).reshape(height, step)[:, : width * channels]
    array = array.reshape(height, width, channels)
    return np.ascontiguousarray(array[:, :, list(order)])  # fancy indexing copies: writable, RGB


def decode_compressed_image(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes (sensor_msgs/CompressedImage) -> HxWx3 uint8 RGB. cv2 is imported lazily."""
    import cv2

    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode the compressed image payload")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def to_policy_image(
    frame: np.ndarray,
    train_res: tuple[int, int] | None,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """HxWx3 uint8 RGB -> 224x224x3 uint8.

    `crop=(x, y, w, h)` is applied first (a site knob; the converter never crops) and must lie inside the
    frame, else ValueError. If `train_res=(W, H)` is given and differs from the (cropped) frame size, the
    frame is PIL-BILINEAR resized to it (the resolution the training images were stored at). Finally the
    frame is letterboxed to 224x224.
    """
    if crop is not None:
        x, y, w, h = crop
        frame_h, frame_w = frame.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > frame_w or y + h > frame_h:
            raise ValueError(f"crop (x, y, w, h)={tuple(crop)} does not fit inside the {frame_w}x{frame_h} (WxH) frame")
        frame = frame[y : y + h, x : x + w]
    if train_res is not None and (frame.shape[1], frame.shape[0]) != tuple(train_res):
        frame = np.asarray(Image.fromarray(frame).resize(tuple(train_res), Image.BILINEAR))
    frame = image_tools.resize_with_pad(frame, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)
    return image_tools.convert_to_uint8(frame)


# ----------------------------------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Snapshot:
    """The latest sample of every stream, assembled by the ROS node.

    images: camera name (head / wrist_left / wrist_right) -> HxWx3 uint8 RGB.
    arm_q: arm side -> 7 measured joint positions in policy joint order (see `map_joints`).
    knuckle: arm side -> gripper knuckle position (rad). A side whose gripper stream has not delivered a
        sample (only allowed for `required: false` grippers) is simply absent.
    ages: stream name (`cam/<name>`, `arm/<side>`, `grip/<side>`) -> seconds since its last sample.
    t_mono: monotonic time the snapshot was taken.
    """

    images: dict[str, np.ndarray]
    arm_q: dict[str, np.ndarray]
    knuckle: dict[str, float]
    ages: dict[str, float]
    t_mono: float


def map_joints(names: Sequence[str], positions: Sequence[float], wanted: Sequence[str]) -> np.ndarray:
    """Pick `wanted` joints out of a JointState (matched by name, any order, extras ignored) -> (len(wanted),)."""
    if len(names) != len(positions):
        raise ValueError(f"JointState has {len(names)} names but {len(positions)} positions")
    by_name = dict(zip(names, positions, strict=True))
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise MissingJointError(missing, names)
    return np.array([by_name[n] for n in wanted], dtype=np.float64)


def build_state(snap: Snapshot, pin: Mapping[int, float]) -> np.ndarray:
    """16-dim float32 state in munich16 order `[L 7 q, L knuckle, R 7 q, R knuckle]`.

    Both arms are always required in `snap.arm_q` (the policy state has both, whichever arms are active).
    `pin` (state dim -> constant) overwrites those dims after assembly, so a pinned dim never depends on the
    live value. A missing `snap.knuckle[side]` means the gripper stream is not required and has no sample (the
    node only fills streams it received); that is acceptable only if the knuckle dim is pinned, else StateError.
    Non-finite values in unpinned dims raise StateError.
    """
    state = np.full(layout.STATE_DIM, np.nan, dtype=np.float64)
    for side, arm_slice in layout.ARM_SLICES.items():
        q = snap.arm_q.get(side)
        if q is None:
            raise StateError(f"no measured joint positions for the {side} arm")
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (arm_slice.stop - arm_slice.start,):
            raise StateError(
                f"{side} arm has {q.shape} joint positions, expected ({arm_slice.stop - arm_slice.start},)"
            )
        state[arm_slice] = q
        knuckle = snap.knuckle.get(side)
        if knuckle is not None:
            state[layout.GRIPPER_INDEX[side]] = knuckle
    for dim, value in pin.items():
        state[dim] = value
    for side, index in layout.GRIPPER_INDEX.items():
        if np.isnan(state[index]) and side not in snap.knuckle:
            raise StateError(
                f"no {side} gripper sample and state dim {index} is not pinned (set state.pin.{index} or "
                f"make the {side} gripper stream required)"
            )
    bad = np.flatnonzero(~np.isfinite(state))
    if bad.size:
        raise StateError(f"state dim(s) {bad.tolist()} are not finite: {state[bad].tolist()}")
    return state.astype(np.float32)


def state_range_report(state: np.ndarray, norm_stats: dict | None) -> list[str]:
    """Per-dim messages for state values outside the training [q01, q99] widened by 10 % of that range.

    `norm_stats` is the parsed `norm_stats.json` (`{"norm_stats": {"state": {...}}}`), its `state` entry, or
    None (no check -> empty list). Non-finite values are always reported.
    """
    if norm_stats is None:
        return []
    stats = norm_stats.get("norm_stats", norm_stats)
    stats = stats.get("state", stats)
    q01, q99 = np.asarray(stats["q01"], dtype=np.float64), np.asarray(stats["q99"], dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if q01.shape != state.shape or q99.shape != state.shape:
        raise ValueError(
            f"norm stats cover {q01.shape[0]} state dims but the state has {state.shape[0]}: wrong-layout checkpoint?"
        )
    margin = _RANGE_MARGIN * (q99 - q01)
    report = []
    for dim, value in enumerate(state):
        lo, hi = q01[dim] - margin[dim], q99[dim] + margin[dim]
        if not np.isfinite(value):
            report.append(f"dim {dim}: value {value} is not finite")
        elif not lo <= value <= hi:
            report.append(f"dim {dim}: value {value:.4f} outside the training range [{lo:.4f}, {hi:.4f}]")
    return report


# ----------------------------------------------------------------------------------------------------
# Observation
# ----------------------------------------------------------------------------------------------------


def build_observation(
    snap: Snapshot,
    cameras: Mapping[str, CameraConfig],
    prompt: str,
    pin: Mapping[int, float] | None = None,
) -> dict:
    """The policy-server observation dict for one snapshot.

    Each image goes through `to_policy_image(img, cfg.train_res, cfg.crop)` of its camera; `pin` is the site's
    `state.pin` (see `build_state`, which also defines the missing-gripper convention).
    """
    missing = [name for name in IMAGE_KEYS if name not in snap.images]
    if missing:
        raise ValueError(f"snapshot has no image for camera(s) {missing}; have {sorted(snap.images)}")
    obs = {
        key: to_policy_image(snap.images[name], cameras[name].train_res, cameras[name].crop)
        for name, key in IMAGE_KEYS.items()
    }
    obs["observation/state"] = build_state(snap, pin or {})
    obs["prompt"] = prompt
    return obs
