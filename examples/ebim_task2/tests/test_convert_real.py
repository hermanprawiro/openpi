import json
import pathlib

import convert_task2_real_to_lerobot as convert
import numpy as np
import pytest
from robot_client import layout
from robot_client import preprocess

SMOKE_REPO = "ebim/test_munich_smoke"


def test_align_holds_10fps_head_and_subsamples_29fps_wrist():
    rows = np.arange(6) * 0.05
    head = np.arange(3) * 0.1
    idx, before = convert.align_frame_indices(head, rows)
    assert idx.tolist() == [0, 0, 1, 1, 2, 2]
    assert before == 0
    wrist = np.arange(9) / 29.0
    assert convert.align_frame_indices(wrist, rows)[0].tolist() == [0, 1, 2, 4, 5, 7]


def test_rows_before_first_frame_counted():
    idx, before = convert.align_frame_indices(np.array([0.12, 0.2]), np.array([0.0, 0.05, 0.15]))
    assert idx.tolist() == [0, 0, 0]
    assert before == 2


def test_state_and_action_slices():
    s = np.arange(42, dtype=np.float32)
    out = convert.extract_state16(s)
    assert out.tolist() == [*range(7), 7, *range(21, 28), 28]
    assert out.dtype == np.float32
    a = convert.extract_action16(np.arange(17.0))
    assert a.tolist() == list(range(16))
    assert a.dtype == np.float32


def test_state_and_action_slices_batched():
    assert convert.extract_state16(np.zeros((5, 42))).shape == (5, 16)
    assert convert.extract_action16(np.zeros((5, 17))).shape == (5, 16)


def test_validity():
    assert convert.episode_is_valid(np.array([1, 1, 1])) is True
    assert convert.episode_is_valid(np.array([0, 0])) is False


def test_validity_mixed_raises():
    with pytest.raises(ValueError, match="mixed validity"):
        convert.episode_is_valid(np.array([1, 1, 0]))


def test_start_pose_medoid_is_a_real_start():
    starts = np.zeros((3, 16))
    starts[1, 8:15] = 0.1
    starts[2, 8:15] = 1.0
    st = convert.start_pose_stats(starts, [4, 7, 9])
    assert st["n"] == 3
    assert st["right"]["medoid_episode"] == 7
    assert st["right"]["medoid"] == [0.1] * 7
    assert len(st["right"]["mean"]) == 7
    assert len(st["right"]["std"]) == 7
    assert st["left"]["medoid"] == [0.0] * 7


def test_holdout_split_deterministic_and_disjoint():
    a = convert.split_holdout(list(range(100)), 10, 0)
    b = convert.split_holdout(list(range(100)), 10, 0)
    assert a == b
    assert not set(a[0]) & set(a[1])
    assert len(a[1]) == 10
    assert a[0] == sorted(a[0])
    assert a[1] == sorted(a[1])
    assert len(a[0]) == 90


# A pose inside the FR3 limits on every joint (joint 4 must be < -0.1518, joint 6 > 0.5445), grippers 0 / 1.
_POSE16 = [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0, 1.0]


def _good_episode(n=4):
    images = {"head": np.full((n, 8, 8, 3), 5, np.uint8), "wrist": np.full((n, 8, 8, 3), 5, np.uint8)}
    pose = np.tile(np.asarray(_POSE16, np.float32), (n, 1))
    return images, pose.copy(), pose.copy()


def test_validate_episode_clean():
    assert convert.validate_episode(*_good_episode()) == []


def test_validate_episode_flags_problems():
    images, state, action = _good_episode()
    images["head"] = images["head"][:3]  # frame count mismatch
    images["wrist"][2] = 0  # black frame
    state[0, 0] = np.nan
    action[1, 2] = 3.5
    action[1, 15] = 1.5
    msgs = " | ".join(convert.validate_episode(images, state, action))
    for needle in ("head", "black", "non-finite", "FR3 joint limits", "joint 3", "gripper"):
        assert needle in msgs


def test_validate_episode_uses_per_joint_fr3_limits():
    """Right joint 6 up to 4.5169 rad is legal on the FR3 (a flat 3.2 rad bound flagged it: ruling R5)."""
    images, state, action = _good_episode()
    state[:, 13] = 4.4  # right joint 6
    assert convert.validate_episode(images, state, action) == []
    state[0, 11] = 0.0  # right joint 4 above its q_max -0.1518
    msgs = " | ".join(convert.validate_episode(images, state, action))
    assert "state" in msgs
    assert "right joint 4" in msgs


def test_align_with_no_frames_raises():
    with pytest.raises(ValueError, match="no frames"):
        convert.align_frame_indices(np.array([]), np.array([0.0, 0.05]))


def test_validity_empty_is_invalid():
    assert convert.episode_is_valid(np.array([], dtype=np.int64)) is False


def test_validate_episode_zero_frame_stream_warns():
    images, state, action = _good_episode()
    images["head"] = np.zeros((0, 8, 8, 3), np.uint8)
    msgs = " | ".join(convert.validate_episode(images, state, action))
    assert "'head' has 0 frames" in msgs


def test_train_res_wiring_matches_layout():
    assert convert.TRAIN_RES == {
        "observation.images.head": layout.HEAD_TRAIN_RES,
        "observation.images.wrist_left": None,
        "observation.images.wrist_right": None,
    }
    assert set(convert.TRAIN_RES) == set(convert.CAMERAS)


def _write_tasks(raw, *tasks):
    (raw / "meta").mkdir(parents=True)
    (raw / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks))
    )


def test_load_task_returns_layout_prompt(tmp_path):
    _write_tasks(tmp_path, layout.PROMPT)
    assert convert.load_task(tmp_path) == layout.PROMPT


def test_load_task_rejects_other_prompt(tmp_path):
    _write_tasks(tmp_path, layout.PROMPT + ".")
    with pytest.raises(ValueError, match="layout.PROMPT"):
        convert.load_task(tmp_path)


def test_converted_smoke_dataset():
    """Checks the smoke conversion against the raw data; skips if it is absent. Produce it with:

    uv run examples/ebim_task2/convert_task2_real_to_lerobot.py --raw-dir <task2_munich> \
        --repo-id ebim/test_munich_smoke --holdout-repo-id ebim/test_munich_smoke_holdout \
        --num-holdout 1 --max-episodes 3
    """
    lerobot_dataset = pytest.importorskip("lerobot.common.datasets.lerobot_dataset")
    pq = pytest.importorskip("pyarrow.parquet")
    av = pytest.importorskip("av")
    report_path = lerobot_dataset.HF_LEROBOT_HOME / SMOKE_REPO / "conversion" / "conversion_report.json"
    if not report_path.exists():
        pytest.skip(f"smoke dataset {SMOKE_REPO} not converted")
    report = json.loads(report_path.read_text())
    raw = pathlib.Path(report["raw_dir"])
    if not raw.exists():
        pytest.skip(f"raw dataset {raw} not available")

    def to_hwc(x):
        return (x * 255).round().byte().permute(1, 2, 0).numpy()

    item = lerobot_dataset.LeRobotDataset(SMOKE_REPO)[0]
    for key in convert.CAMERAS.values():
        assert to_hwc(item[key]).shape == (224, 224, 3)
    assert tuple(item["state"].shape) == (16,)
    assert tuple(item["actions"].shape) == (16,)
    assert item["task"] == layout.PROMPT

    episode = report["train_episodes"][0]
    row0 = pq.read_table(convert.episode_parquet_path(raw, episode), columns=["observation.state"]).column(0)[0]
    np.testing.assert_array_equal(item["state"].numpy(), convert.extract_state16(np.asarray(row0.as_py())))

    # Train/deploy image parity: the stored head image is what the client makes of the raw head frame.
    head_video = raw / "videos" / "chunk-000" / convert.HEAD_VIDEO_KEY / f"episode_{episode:06d}.mp4"
    with av.open(str(head_video)) as container:
        frame = next(container.decode(video=0))
    assert frame.time == 0
    expected = preprocess.to_policy_image(frame.to_ndarray(format="rgb24"), layout.HEAD_TRAIN_RES)
    diff = np.abs(to_hwc(item["image"]).astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 1
