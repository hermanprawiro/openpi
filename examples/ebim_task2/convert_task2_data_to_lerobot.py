"""
Convert the EBIM task2_thermalpad dataset (LeRobot v3.0 format) to the LeRobot v2.1 format
used by openpi's pinned lerobot version.

The raw dataset was recorded by ebim/benchmark/task2_isaacsim/services/recording/record_task2.py.
It cannot be loaded by openpi directly because openpi pins an older lerobot (v2.1 dataset
format), so this script reads the v3.0 parquet/video files directly (pyarrow + pyav) and
rewrites them through the pinned lerobot's LeRobotDataset.

Raw layout (per DATASET.md in the recording stack):
  - action (20,): [0:3] base twist vx,vy,wz | [3:10] left arm joint targets (abs, rad)
    | [10:17] right arm joint targets | [17] left gripper open-fraction
    | [18] right gripper open-fraction | [19] spine height target (m)
  - observation.state (37,): world-frame EE poses [0:14], joints [14:28], spine [28],
    grippers [29:31], base odom [31:34], base vel [34:37]

We keep the full 20-dim action and select a 20-dim state that mirrors the action layout
(base vel, arm joints, grippers, spine) so that the DeltaActions transform in the training
config is element-wise consistent. World-frame EE poses and odometry are dropped since they
do not generalize across scene placements.

Episodes labeled success=false in task2_extras/episodes_task2.jsonl are skipped by default.

Usage (see examples/ebim_task2/README.md for the canonical dataset ids):
uv run examples/ebim_task2/convert_task2_data_to_lerobot.py \
    --raw-repo-id <hf-user>/<raw-dataset> --repo-id <output-repo-id>
or, from a local raw recording:
uv run examples/ebim_task2/convert_task2_data_to_lerobot.py \
    --raw-dir /path/to/raw/dataset --repo-id <output-repo-id>

The resulting dataset is saved to $HF_LEROBOT_HOME/<repo-id>.
"""

import os

# The converter only uses jax for image resizing; keep it off the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import pathlib
import re
import shutil

import av
import huggingface_hub
import numpy as np
import pyarrow.parquet as pq
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.shared import image_tools

# Raw video keys -> output feature names (eval_camera is intentionally excluded: it is a
# static scene camera used for evaluation only).
CAMERAS = {
    "observation.images.head": "image",
    "observation.images.wrist_left": "left_wrist_image",
    "observation.images.wrist_right": "right_wrist_image",
}

IMAGE_SIZE = 224
RESIZE_BATCH = 64

# observation.state indices for the 20-dim state matching the action layout:
# base vel vx,vy,wz | left joints 1-7 | right joints 1-7 | left grip | right grip | spine
STATE_INDICES = [34, 35, 36, *range(14, 21), *range(21, 28), 29, 30, 28]


def normalize_hf_repo_id(raw_repo_id: str) -> str:
    """Accepts a plain repo id ('org/name') or a full URL ('https://huggingface.co/datasets/org/name')."""
    match = re.match(r"https?://huggingface\.co/datasets/([^/]+/[^/?#]+)", raw_repo_id)
    return match.group(1) if match else raw_repo_id


def load_success_labels(raw_dir: pathlib.Path) -> dict[int, bool]:
    labels_path = raw_dir / "task2_extras" / "episodes_task2.jsonl"
    if not labels_path.exists():
        return {}
    labels = {}
    with open(labels_path) as f:
        for line in f:
            record = json.loads(line)
            labels[int(record["episode_index"])] = bool(record.get("success", True))
    return labels


def decode_video_segment(video_path: pathlib.Path, from_timestamp: float, num_frames: int, fps: float) -> np.ndarray:
    """Decodes `num_frames` frames starting at `from_timestamp` (seconds) from a video file."""
    eps = 0.5 / fps
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # container.seek expects av.time_base (microsecond) units; lands on the previous keyframe.
        container.seek(int(from_timestamp / av.time_base), backward=True)
        for frame in container.decode(stream):
            if frame.time is None or frame.time < from_timestamp - eps:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == num_frames:
                break
    if len(frames) != num_frames:
        raise ValueError(
            f"Expected {num_frames} frames at t={from_timestamp:.3f}s in {video_path}, decoded {len(frames)}"
        )
    return np.stack(frames)


def resize_frames(frames: np.ndarray) -> np.ndarray:
    """Letterbox-resizes (B, H, W, C) uint8 frames to IMAGE_SIZE using the same function as inference."""
    out = []
    for start in range(0, len(frames), RESIZE_BATCH):
        batch = frames[start : start + RESIZE_BATCH]
        out.append(np.asarray(image_tools.resize_with_pad(batch, IMAGE_SIZE, IMAGE_SIZE)))
    return np.concatenate(out)


def main(
    repo_id: str,
    *,
    raw_dir: str | None = None,
    raw_repo_id: str | None = None,
    max_episodes: int | None = None,
    include_failed: bool = False,
    push_to_hub: bool = False,
):
    """Converts the raw LeRobot v3.0 task2 dataset to the v2.1 format used by openpi.

    Args:
        repo_id: Output repo id; the converted dataset is written to $HF_LEROBOT_HOME/<repo_id>.
        raw_dir: Local directory containing the raw v3.0 dataset. Mutually exclusive with raw_repo_id.
        raw_repo_id: Hugging Face dataset to download (repo id or full URL). Mutually exclusive with raw_dir.
        max_episodes: If set, stop after converting this many (successful) episodes.
        include_failed: Also convert episodes labeled success=false.
        push_to_hub: Push the converted dataset to the Hugging Face hub.
    """
    if (raw_dir is None) == (raw_repo_id is None):
        raise ValueError("Provide exactly one of --raw-dir (local path) or --raw-repo-id (Hugging Face dataset).")
    if raw_dir is not None:
        raw_path = pathlib.Path(raw_dir)
    else:
        assert raw_repo_id is not None
        raw_path = pathlib.Path(
            huggingface_hub.snapshot_download(repo_id=normalize_hf_repo_id(raw_repo_id), repo_type="dataset")
        )
    info = json.loads((raw_path / "meta" / "info.json").read_text())
    fps = info["fps"]

    episodes = []
    for episodes_file in sorted((raw_path / "meta" / "episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(episodes_file).to_pylist())
    episodes.sort(key=lambda ep: ep["episode_index"])

    success = load_success_labels(raw_path)

    # dataset_from_index/dataset_to_index in the episode metadata are global row indices,
    # so record each data file's starting offset to translate them into file-local indices.
    data = {}
    row_offset = 0
    for data_file in sorted((raw_path / "data").rglob("*.parquet")):
        table = pq.read_table(data_file, columns=["action", "observation.state"])
        chunk_file = (int(data_file.parent.name.split("-")[1]), int(data_file.stem.split("-")[1]))
        data[chunk_file] = {
            "action": np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32),
            "state": np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32),
            "row_offset": row_offset,
        }
        row_offset += table.num_rows

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=info.get("robot_type", "fr3duo_mobile_task2"),
        fps=fps,
        features={
            **{
                name: {
                    "dtype": "image",
                    "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                    "names": ["height", "width", "channel"],
                }
                for name in CAMERAS.values()
            },
            "state": {"dtype": "float32", "shape": (len(STATE_INDICES),), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (20,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    num_written = 0
    for ep in episodes:
        if max_episodes is not None and num_written >= max_episodes:
            print(f"Reached --max-episodes={max_episodes}, stopping.")
            break
        ep_index = ep["episode_index"]
        if not include_failed and not success.get(ep_index, True):
            print(f"Skipping episode {ep_index} (success=false)")
            continue

        length = ep["length"]
        ep_data = data[(ep["data/chunk_index"], ep["data/file_index"])]
        start = ep["dataset_from_index"] - ep_data["row_offset"]
        end = ep["dataset_to_index"] - ep_data["row_offset"]
        actions = ep_data["action"][start:end]
        states = ep_data["state"][start:end][:, STATE_INDICES]
        if len(actions) != length:
            raise ValueError(f"Episode {ep_index}: expected {length} rows, got {len(actions)}")

        images = {}
        for video_key, feature_name in CAMERAS.items():
            video_path = (
                raw_path
                / "videos"
                / video_key
                / f"chunk-{ep[f'videos/{video_key}/chunk_index']:03d}"
                / f"file-{ep[f'videos/{video_key}/file_index']:03d}.mp4"
            )
            frames = decode_video_segment(video_path, ep[f"videos/{video_key}/from_timestamp"], length, fps)
            images[feature_name] = resize_frames(frames)

        task = str(ep["tasks"][0])
        for i in range(length):
            dataset.add_frame(
                {
                    **{name: images[name][i] for name in CAMERAS.values()},
                    "state": states[i],
                    "actions": actions[i],
                    "task": task,
                }
            )
        dataset.save_episode()
        num_written += 1
        print(f"Wrote episode {ep_index} ({length} frames)")

    print(f"Done: {num_written} episodes, {dataset.meta.total_frames} frames -> {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["ebim", "task2", "isaacsim", "fr3"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
