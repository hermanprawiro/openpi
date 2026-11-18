"""
Convert the EBIM task2 Munich real-robot dataset (LeRobot v3.0, 42-dim state / 17-dim action) to
the LeRobot v2.1 format used by openpi, in the 16-dim `munich16` layout (see robot_client/layout.py).

This module currently holds the pure-numpy helpers (frame alignment, layout slicing, validity,
hold-out split, start-pose statistics, per-episode sanity checks). They import without lerobot/av;
those are only imported inside `main`.
"""

import numpy as np
from robot_client import layout

JOINT_LIMIT_RAD = 3.2


def align_frame_indices(frame_ts: np.ndarray, row_ts: np.ndarray, eps: float = 1e-3) -> tuple[np.ndarray, int]:
    """For each row, the index of the latest frame with `frame_ts <= row_ts + eps`.

    Rows before the first frame map to 0. Returns `(indices, n_rows_before_first_frame)`.
    """
    frame_ts = np.asarray(frame_ts, dtype=np.float64)
    row_ts = np.asarray(row_ts, dtype=np.float64)
    raw = np.searchsorted(frame_ts, row_ts + eps, side="right") - 1
    return np.maximum(raw, 0), int(np.sum(raw < 0))


def extract_state16(state42: np.ndarray) -> np.ndarray:
    """Select the munich16 state from the dataset's 42-dim `observation.state` (last axis)."""
    return np.asarray(state42)[..., list(layout.DATASET_STATE_INDICES)].astype(np.float32)


def extract_action16(action17: np.ndarray) -> np.ndarray:
    """Select the munich16 action from the dataset's 17-dim `action` (last axis)."""
    return np.asarray(action17)[..., list(layout.DATASET_ACTION_INDICES)].astype(np.float32)


def episode_is_valid(validity: np.ndarray) -> bool:
    """True iff every per-frame validity value is 1. Mixed values within an episode raise ValueError."""
    validity = np.asarray(validity)
    valid = validity == 1
    if valid.all():
        return True
    if not valid.any():
        return False
    raise ValueError(f"mixed validity within one episode: {np.unique(validity).tolist()}")


def split_holdout(valid_ids: list[int], n: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministically hold out `n` episodes. Returns `(train_ids, holdout_ids)`, both sorted."""
    ids = sorted(valid_ids)
    held = sorted(int(i) for i in np.random.default_rng(seed).choice(ids, n, replace=False))
    held_set = set(held)
    return [i for i in ids if i not in held_set], held


def start_pose_stats(first_actions: np.ndarray, episode_ids: list[int]) -> dict:
    """Per-arm statistics of the episode start poses (first action row of each episode, 16-dim).

    Per arm: `mean`, `std`, `medoid` (7 joints each) and `medoid_episode`; plus `n`. The medoid is
    the real episode start minimising the summed L2 distance (per arm) to all other starts; use it,
    not the mean, as the site start pose (the mean can be far from every real start).
    """
    first_actions = np.asarray(first_actions, dtype=np.float64)
    if len(first_actions) != len(episode_ids):
        raise ValueError(f"{len(first_actions)} start poses but {len(episode_ids)} episode ids")
    out: dict = {"n": len(episode_ids)}
    for arm, sl in layout.ARM_SLICES.items():
        q = first_actions[:, sl]
        dist = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1).sum(axis=1)
        m = int(np.argmin(dist))
        out[arm] = {
            "mean": q.mean(axis=0).tolist(),
            "std": q.std(axis=0).tolist(),
            "medoid": q[m].tolist(),
            "medoid_episode": int(episode_ids[m]),
        }
    return out


def validate_episode(images: dict[str, np.ndarray], state: np.ndarray, action: np.ndarray) -> list[str]:
    """Sanity-check one converted episode; returns human-readable warnings (empty if clean)."""
    warnings: list[str] = []
    n_rows = len(action)
    if len(state) != n_rows:
        warnings.append(f"state has {len(state)} rows but action has {n_rows}")
    for name, frames in images.items():
        if len(frames) != n_rows:
            warnings.append(f"image '{name}' has {len(frames)} frames but there are {n_rows} rows")
        black = np.flatnonzero(frames.reshape(len(frames), -1).max(axis=1) == 0)
        if len(black):
            warnings.append(f"image '{name}' has {len(black)} all-black frame(s), first at {int(black[0])}")
    for name, arr in (("state", state), ("action", action)):
        if not np.isfinite(arr).all():
            warnings.append(f"{name} has non-finite values")
        joints = np.abs(np.asarray(arr)[..., list(layout.ARM_DIMS)])
        if (joints > JOINT_LIMIT_RAD).any():
            warnings.append(f"{name} has |joint| > {JOINT_LIMIT_RAD} rad (max {np.nanmax(joints):.2f})")
    width = np.asarray(action)[..., list(layout.GRIPPER_INDEX.values())]
    if ((width < 0) | (width > 1)).any():
        warnings.append("action gripper width outside [0, 1]")
    return warnings
