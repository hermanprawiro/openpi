"""
Convert the EBIM task2 Munich real-robot dataset to the LeRobot v2.1 dataset used by openpi, in the
16-dim `munich16` layout (see robot_client/layout.py).

The raw dataset is LeRobot v2.1 on paper, but it cannot be trained on directly: state/action are
42/17-dim, video frame rates differ from the 20 Hz rows and from `info.json` (the head video is
336x188 at ~10 fps in the early episodes and 1280x720 at ~23 fps in the later ones; wrists are 640x480
at ~29 fps), episode indices have gaps, and 89 of 238 episodes are marked invalid
(`annotation.human.validity != 1`). This script reads the per-episode parquet + AV1 videos directly
(pyav), keeps the valid episodes, aligns every row to the latest frame of each camera, and turns every
frame into a 224x224 policy image with `robot_client.preprocess.to_policy_image` -- the same function
and arguments the real-robot client uses (head resized to `layout.HEAD_TRAIN_RES` if its size differs, then
letterboxed; wrists letterboxed only). It writes a train repo plus a seeded hold-out repo.

Outputs (both repos are removed and recreated, so re-runs are idempotent):
  $HF_LEROBOT_HOME/<repo_id>                         train episodes
  $HF_LEROBOT_HOME/<holdout_repo_id>                 hold-out episodes (offline checks only)
  $HF_LEROBOT_HOME/<repo_id>/conversion/conversion_report.json   per-episode alignment + warnings
  $HF_LEROBOT_HOME/<repo_id>/conversion/start_pose_stats.json    start poses of the train episodes

Usage:
uv run examples/ebim_task2/convert_task2_real_to_lerobot.py \
    --raw-dir /home/mislab/datasets/data1/ebim/ebim_task2_realrobotdata/task2_munich

The pure-numpy helpers below import without lerobot/av; those are only imported by the I/O code.
"""

import json
import pathlib
import shutil

import numpy as np
from robot_client import layout
from robot_client import preprocess
import tyro

JOINT_LIMIT_RAD = 3.2


def align_frame_indices(frame_ts: np.ndarray, row_ts: np.ndarray, eps: float = 1e-3) -> tuple[np.ndarray, int]:
    """For each row, the index of the latest frame with `frame_ts <= row_ts + eps`.

    Rows before the first frame map to 0. Returns `(indices, n_rows_before_first_frame)`.
    Raises ValueError if there are no frames.
    """
    frame_ts = np.asarray(frame_ts, dtype=np.float64)
    row_ts = np.asarray(row_ts, dtype=np.float64)
    if len(frame_ts) == 0:
        raise ValueError("no frames to align rows to")
    raw = np.searchsorted(frame_ts, row_ts + eps, side="right") - 1
    return np.maximum(raw, 0), int(np.sum(raw < 0))


def extract_state16(state42: np.ndarray) -> np.ndarray:
    """Select the munich16 state from the dataset's 42-dim `observation.state` (last axis)."""
    return np.asarray(state42)[..., list(layout.DATASET_STATE_INDICES)].astype(np.float32)


def extract_action16(action17: np.ndarray) -> np.ndarray:
    """Select the munich16 action from the dataset's 17-dim `action` (last axis)."""
    return np.asarray(action17)[..., list(layout.DATASET_ACTION_INDICES)].astype(np.float32)


def episode_is_valid(validity: np.ndarray) -> bool:
    """True iff every per-frame validity value is 1 (an empty episode is invalid).

    Mixed values within an episode raise ValueError.
    """
    validity = np.asarray(validity)
    if validity.size == 0:
        return False
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
        if len(frames) == 0:
            continue
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


# Raw video key -> output feature name (same names as the sim converter / LeRobotEbimTask2DataConfig).
CAMERAS = {
    "observation.images.head": "image",
    "observation.images.wrist_left": "left_wrist_image",
    "observation.images.wrist_right": "right_wrist_image",
}
HEAD_VIDEO_KEY = "observation.images.head"
# Head frames are resized to layout.HEAD_TRAIN_RES (W, H) before letterboxing (the client's `train_res`);
# wrists: None.
TRAIN_RES = {key: (layout.HEAD_TRAIN_RES if key == HEAD_VIDEO_KEY else None) for key in CAMERAS}
# Warn when a camera's last frame is this much older than the episode's last row (frozen tail).
MAX_TAIL_GAP_S = 0.5
FPS = 20
VALIDITY_KEY = "annotation.human.validity"
ROBOT_TYPE = "fr3duo_munich"


def load_task(raw_dir: pathlib.Path) -> str:
    tasks = [json.loads(line)["task"] for line in (raw_dir / "meta" / "tasks.jsonl").read_text().splitlines() if line]
    if len(tasks) != 1:
        raise ValueError(f"expected exactly one task in meta/tasks.jsonl, found {tasks}")
    if tasks[0] != layout.PROMPT:
        raise ValueError(
            f"meta/tasks.jsonl prompt {tasks[0]!r} differs from layout.PROMPT {layout.PROMPT!r}; "
            "the client and the policy server use layout.PROMPT, so training on another string would be refused at deploy time"
        )
    return tasks[0]


def episode_parquet_path(raw_dir: pathlib.Path, episode_index: int) -> pathlib.Path:
    return raw_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def decode_video(
    video_path: pathlib.Path, train_res: tuple[int, int] | None
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Decode every frame. Returns `(policy_images (N, 224, 224, 3), frame_ts (N,), native (W, H))`.

    Frames are letterboxed as they are decoded (the full-resolution wrist stream would not fit in RAM).
    """
    import av

    images, ts = [], []
    native = None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError(f"{video_path}: frame without a timestamp")
            rgb = frame.to_ndarray(format="rgb24")
            native = (rgb.shape[1], rgb.shape[0])
            images.append(preprocess.to_policy_image(rgb, train_res))
            ts.append(frame.time)
    if not images:
        raise ValueError(f"{video_path}: no frames decoded")
    ts = np.asarray(ts, dtype=np.float64)
    if (np.diff(ts) <= 0).any():
        raise ValueError(f"{video_path}: frame timestamps are not strictly increasing")
    return np.stack(images), ts, native


def load_episode(raw_dir: pathlib.Path, episode_index: int) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Read one raw episode. Returns `(images per output feature, state16, action16, report entry)`."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        episode_parquet_path(raw_dir, episode_index), columns=["timestamp", "observation.state", "action"]
    )
    row_ts = table.column("timestamp").to_numpy()
    state = extract_state16(np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)))
    action = extract_action16(np.stack(table.column("action").to_numpy(zero_copy_only=False)))

    images, cameras, warnings = {}, {}, []
    for video_key, feature in CAMERAS.items():
        video_path = raw_dir / "videos" / "chunk-000" / video_key / f"episode_{episode_index:06d}.mp4"
        frames, frame_ts, native = decode_video(video_path, TRAIN_RES[video_key])
        idx, before = align_frame_indices(frame_ts, row_ts)
        images[feature] = frames[idx]
        tail_gap = float(row_ts[-1] - frame_ts[-1])
        cameras[video_key] = {
            "frames": len(frame_ts),
            "native_wh": list(native),
            "first_ts": float(frame_ts[0]),
            "last_ts": float(frame_ts[-1]),
            "rows_before_first_frame": before,
            "frames_used": len(np.unique(idx)),
        }
        if before:
            warnings.append(f"{video_key}: {before} row(s) before the first frame (held at frame 0)")
        if tail_gap > MAX_TAIL_GAP_S:
            warnings.append(f"{video_key}: last frame is {tail_gap:.2f}s before the last row")

    warnings += validate_episode(images, state, action)
    report = {
        "episode_index": episode_index,
        "rows": len(row_ts),
        "cameras": cameras,
        "warnings": warnings,
    }
    return images, state, action, report


def _create_dataset(repo_id: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    size = preprocess.POLICY_IMAGE_SIZE
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features={
            **{
                name: {"dtype": "image", "shape": (size, size, 3), "names": ["height", "width", "channel"]}
                for name in CAMERAS.values()
            },
            "state": {"dtype": "float32", "shape": (layout.STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (layout.ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def _head_size_counts(episode_reports: list[dict]) -> dict[str, int]:
    sizes = ["x".join(map(str, r["cameras"][HEAD_VIDEO_KEY]["native_wh"])) for r in episode_reports]
    return {size: sizes.count(size) for size in sorted(set(sizes))}


def main(
    raw_dir: str,
    repo_id: str = "ebim/task2_munich_real",
    holdout_repo_id: str = "ebim/task2_munich_real_holdout",
    num_holdout: int = 10,
    seed: int = 0,
    max_episodes: int | None = None,
):
    """Convert the Munich real-robot dataset into a train repo and a hold-out repo.

    Args:
        raw_dir: Raw Munich dataset (meta/, data/chunk-000/, videos/chunk-000/).
        repo_id: Output train repo id, written to $HF_LEROBOT_HOME/<repo_id>.
        holdout_repo_id: Output hold-out repo id.
        num_holdout: Number of valid episodes to hold out (seeded).
        seed: Hold-out selection seed.
        max_episodes: If set, only the first this-many valid episodes are used (smoke tests).
    """
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    import pyarrow.parquet as pq

    raw = pathlib.Path(raw_dir)
    task = load_task(raw)

    episodes = sorted(
        int(json.loads(line)["episode_index"])
        for line in (raw / "meta" / "episodes.jsonl").read_text().splitlines()
        if line
    )
    valid_ids, invalid_ids = [], []
    for ep in episodes:
        if max_episodes is not None and len(valid_ids) >= max_episodes:
            break
        validity = pq.read_table(episode_parquet_path(raw, ep), columns=[VALIDITY_KEY]).column(0).to_numpy()
        (valid_ids if episode_is_valid(validity) else invalid_ids).append(ep)
    train_ids, holdout_ids = split_holdout(valid_ids, num_holdout, seed)
    print(
        f"{len(valid_ids)} valid / {len(invalid_ids)} invalid episodes scanned -> "
        f"{len(train_ids)} train + {len(holdout_ids)} hold-out {holdout_ids}"
    )

    for rid in (repo_id, holdout_repo_id):
        if (HF_LEROBOT_HOME / rid).exists():
            shutil.rmtree(HF_LEROBOT_HOME / rid)

    episode_reports = []
    first_actions = []
    totals = {}
    for split, rid, ids in (("train", repo_id, train_ids), ("holdout", holdout_repo_id, holdout_ids)):
        dataset = _create_dataset(rid)
        for ep in ids:
            images, state, action, report = load_episode(raw, ep)
            for i in range(len(action)):
                dataset.add_frame(
                    {
                        **{name: images[name][i] for name in images},
                        "state": state[i],
                        "actions": action[i],
                        "task": task,
                    }
                )
            dataset.save_episode()
            if split == "train":
                first_actions.append(action[0])
            episode_reports.append({**report, "split": split})
            note = f" WARN: {'; '.join(report['warnings'])}" if report["warnings"] else ""
            print(f"[{split}] episode {ep}: {len(action)} rows{note}")
        dataset.stop_image_writer()
        totals[split] = {"episodes": len(ids), "frames": dataset.meta.total_frames}

    out_dir = HF_LEROBOT_HOME / repo_id / "conversion"
    out_dir.mkdir(parents=True, exist_ok=True)
    conversion_report = {
        "raw_dir": str(raw),
        "repo_id": repo_id,
        "holdout_repo_id": holdout_repo_id,
        "task": task,
        "seed": seed,
        "num_holdout": num_holdout,
        "max_episodes": max_episodes,
        "num_valid": len(valid_ids),
        "skipped_invalid": invalid_ids,
        "train_episodes": train_ids,
        "holdout_episodes": holdout_ids,
        "totals": totals,
        "num_warnings": sum(len(r["warnings"]) for r in episode_reports),
        "head_native_wh_counts": _head_size_counts(episode_reports),
        "episodes": sorted(episode_reports, key=lambda r: r["episode_index"]),
    }
    (out_dir / "conversion_report.json").write_text(json.dumps(conversion_report, indent=2) + "\n")
    stats = start_pose_stats(np.stack(first_actions), train_ids)
    (out_dir / "start_pose_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Done: {totals} -> {HF_LEROBOT_HOME / repo_id}, {HF_LEROBOT_HOME / holdout_repo_id}")
    print(f"{conversion_report['num_warnings']} warning(s); report in {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
