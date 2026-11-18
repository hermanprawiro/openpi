import convert_task2_real_to_lerobot as convert
import numpy as np
import pytest


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


def _good_episode(n=4):
    images = {"head": np.full((n, 8, 8, 3), 5, np.uint8), "wrist": np.full((n, 8, 8, 3), 5, np.uint8)}
    return images, np.zeros((n, 16), np.float32), np.zeros((n, 16), np.float32)


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
    for needle in ("head", "black", "non-finite", "3.2", "gripper"):
        assert needle in msgs
