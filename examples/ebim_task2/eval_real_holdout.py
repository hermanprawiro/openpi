#!/usr/bin/env python3
"""Open-loop evaluation of a Munich (munich16) checkpoint on the held-out episodes.

For every `stride`-th frame of every hold-out episode the policy is queried with the recorded observation and
its 50-step action chunk is compared with the recorded actions that follow. Metrics: per-dim MAE over the first
10/25/50 steps, gripper switch timing, and how far the first predicted arm target is from the measured state.
Results go to `<checkpoint_dir>/holdout_eval.json` by default.

    uv run examples/ebim_task2/eval_real_holdout.py --config-name pi05_lora_ebim_task2_munich_real \
        --checkpoint-dir checkpoints/pi05_lora_ebim_task2_munich_real/munich_v1/2500
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from robot_client.layout import ACTION_HORIZON
from robot_client.layout import ARM_DIMS
from robot_client.layout import GRIPPER_INDEX
from robot_client.layout import PROMPT

HORIZONS = (10, 25, 50)
IMAGE_KEYS = ("image", "left_wrist_image", "right_wrist_image")


# --- pure metrics ---------------------------------------------------------------------------------------------


def chunk_errors(pred: np.ndarray, gt: np.ndarray, horizons=HORIZONS) -> dict[int, np.ndarray]:
    """Per-dim MAE over the first `h` steps, truncated to the shorter of pred and gt (episode ended early)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n = min(len(pred), len(gt))
    return {h: np.abs(pred[: min(h, n)] - gt[: min(h, n)]).mean(axis=0) for h in horizons}


def gripper_switch_steps(width: np.ndarray, threshold: float = 0.5) -> list[int]:
    """Steps at which the gripper width crosses `threshold` (index of the first step on the new side)."""
    closed = np.asarray(width) < threshold
    return (np.flatnonzero(closed[1:] != closed[:-1]) + 1).tolist()


def gripper_timing_error(pred_w: np.ndarray, gt_w: np.ndarray) -> float | None:
    """Absolute step difference of the first gripper switch; None if either side has no switch.

    Only the steps both sequences cover are compared.
    """
    n = min(len(pred_w), len(gt_w))
    pred_sw = gripper_switch_steps(np.asarray(pred_w)[:n])
    gt_sw = gripper_switch_steps(np.asarray(gt_w)[:n])
    if not pred_sw or not gt_sw:
        return None
    return float(abs(pred_sw[0] - gt_sw[0]))


def gripper_switch_missed(pred_w: np.ndarray, gt_w: np.ndarray) -> bool:
    """True if the ground truth switches the gripper within the steps both cover and the prediction never does."""
    n = min(len(pred_w), len(gt_w))
    return bool(gripper_switch_steps(np.asarray(gt_w)[:n])) and not gripper_switch_steps(np.asarray(pred_w)[:n])


def gripper_timing_summary(errors: list[float | None], missed: list[bool]) -> dict[str, Any]:
    """Aggregate per-sample timing errors (None: no switch on one side) and missed GT switches of one gripper."""
    found = [e for e in errors if e is not None]
    return {
        "n_both_switch": len(found),
        "n_gt_switch_missed": int(sum(missed)),
        "n_samples": len(errors),
        "mean_abs_steps": float(np.mean(found)) if found else None,
        "max_abs_steps": float(np.max(found)) if found else None,
    }


def first_step_offset(pred: np.ndarray, state: np.ndarray) -> float:
    """Max |pred[0] - state| over the arm joints (grippers excluded: width % vs knuckle rad)."""
    return float(np.max(np.abs(np.asarray(pred)[0, list(ARM_DIMS)] - np.asarray(state)[list(ARM_DIMS)])))


# --- data -----------------------------------------------------------------------------------------------------


def load_dataset(repo_id: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id)


def build_obs(frame: dict) -> dict[str, Any]:
    """Turn a LeRobotDataset frame (CHW float images in [0, 1]) into the policy input dict (HWC uint8)."""
    obs: dict[str, Any] = {}
    for key in IMAGE_KEYS:
        chw = np.asarray(frame[key], dtype=np.float32)
        obs[f"observation/{key}"] = np.round(255 * chw).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    obs["observation/state"] = np.asarray(frame["state"], dtype=np.float32)
    obs["prompt"] = PROMPT
    return obs


def norm_stats_sha256(checkpoint_dir: str | Path, asset_id: str | None) -> str | None:
    """SHA-256 of `<checkpoint_dir>/assets/<asset_id>/norm_stats.json`, or None if there is none."""
    if asset_id is None:
        return None
    path = Path(checkpoint_dir) / "assets" / asset_id / "norm_stats.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# --- evaluation -----------------------------------------------------------------------------------------------


def _summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def evaluate(policy, dataset, stride: int = 20) -> dict[str, Any]:
    """Query `policy` (anything with `.infer(obs) -> {"actions": (T, 16)}`) at every `stride`-th hold-out frame."""
    all_actions = np.stack([np.asarray(a) for a in dataset.hf_dataset["actions"]])
    starts = dataset.episode_data_index["from"].tolist()
    ends = dataset.episode_data_index["to"].tolist()

    errors: dict[int, list[np.ndarray]] = {h: [] for h in HORIZONS}
    offsets: list[float] = []
    timing: dict[str, list[float | None]] = {name: [] for name in GRIPPER_INDEX}
    missed: dict[str, list[bool]] = {name: [] for name in GRIPPER_INDEX}
    for start, end in zip(starts, ends, strict=True):
        for i in range(start, end, stride):
            frame = dataset[i]
            pred = np.asarray(policy.infer(build_obs(frame))["actions"])
            gt = all_actions[i : min(i + ACTION_HORIZON, end)]
            for h, e in chunk_errors(pred, gt).items():
                errors[h].append(e)
            offsets.append(first_step_offset(pred, np.asarray(frame["state"])))
            for name, idx in GRIPPER_INDEX.items():
                timing[name].append(gripper_timing_error(pred[:, idx], gt[:, idx]))
                missed[name].append(gripper_switch_missed(pred[:, idx], gt[:, idx]))

    mae = {h: np.mean(errors[h], axis=0) for h in HORIZONS}
    grip = list(GRIPPER_INDEX.values())
    gripper_timing = {name: gripper_timing_summary(timing[name], missed[name]) for name in GRIPPER_INDEX}
    return {
        "num_samples": len(offsets),
        "num_episodes": len(starts),
        "stride": stride,
        "mae_per_dim": {str(h): mae[h].tolist() for h in HORIZONS},
        "arm_mae": {str(h): float(mae[h][list(ARM_DIMS)].mean()) for h in HORIZONS},
        "gripper_mae": {str(h): float(mae[h][grip].mean()) for h in HORIZONS},
        "gripper_timing": gripper_timing,
        "first_step_offset": _summary(offsets),
    }


# --- CLI ------------------------------------------------------------------------------------------------------


def _create_policy(config_name: str, checkpoint_dir: str):
    from openpi.policies import policy_config
    from openpi.training import config as _config

    return policy_config.create_trained_policy(_config.get_config(config_name), checkpoint_dir)


def _asset_id(config_name: str) -> str | None:
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    return cfg.data.create(cfg.assets_dirs, cfg.model).asset_id


def main(
    config_name: str,
    checkpoint_dir: str,
    holdout_repo_id: str = "ebim/task2_munich_real_holdout",
    stride: int = 20,
    out: str | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on the hold-out episodes and write `holdout_eval.json`.

    Tests replace the module-level `_create_policy` to run this without a model.
    """
    policy = _create_policy(config_name, checkpoint_dir)
    results = evaluate(policy, load_dataset(holdout_repo_id), stride=stride)
    results.update(
        config_name=config_name,
        checkpoint_dir=str(checkpoint_dir),
        holdout_repo_id=holdout_repo_id,
        norm_stats_sha256=norm_stats_sha256(checkpoint_dir, _asset_id(config_name)),
    )
    out_path = Path(out) if out else Path(checkpoint_dir) / "holdout_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(
        f"arm MAE@10={results['arm_mae']['10']:.4f} @25={results['arm_mae']['25']:.4f} "
        f"@50={results['arm_mae']['50']:.4f}; wrote {out_path}"
    )
    return results


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
