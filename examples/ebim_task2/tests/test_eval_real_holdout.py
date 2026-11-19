import json
from pathlib import Path

import eval_real_holdout as ev
import numpy as np
import pytest


def test_chunk_errors_truncates_at_episode_end():
    pred = np.ones((50, 16))
    gt = np.zeros((20, 16))
    e = ev.chunk_errors(pred, gt)
    assert e[10].tolist() == [1.0] * 16 and e[50].tolist() == [1.0] * 16


def test_chunk_errors_per_dim_mae_over_first_h_steps():
    pred = np.zeros((50, 16))
    pred[:10, 3] = 2.0  # error only in the first 10 steps of dim 3
    pred[10:25, 4] = -1.0
    e = ev.chunk_errors(pred, np.zeros((50, 16)))
    assert e[10][3] == pytest.approx(2.0) and e[25][3] == pytest.approx(2.0 * 10 / 25)
    assert e[10][4] == 0.0 and e[25][4] == pytest.approx(15 / 25) and e[50][4] == pytest.approx(15 / 50)


def test_gripper_timing():
    gt = np.r_[np.ones(10), np.zeros(10)]
    pr = np.r_[np.ones(13), np.zeros(7)]
    assert ev.gripper_timing_error(pr, gt) == 3
    assert ev.gripper_timing_error(np.ones(20), gt) is None


def test_gripper_switch_steps_both_directions():
    w = np.r_[np.ones(3), np.zeros(2), np.ones(2)]
    assert ev.gripper_switch_steps(w) == [3, 5]
    assert ev.gripper_switch_steps(np.ones(5)) == []


def test_gripper_timing_compares_only_overlapping_steps():
    # prediction is longer than ground truth; the late switch is beyond the episode end
    gt = np.ones(10)
    pr = np.r_[np.ones(15), np.zeros(5)]
    assert ev.gripper_timing_error(pr, gt) is None


def test_first_step_offset_ignores_grippers():
    pred = np.zeros((50, 16))
    pred[0, 7] = 1.0
    pred[0, 15] = 0.7
    pred[0, 9] = 0.2
    assert ev.first_step_offset(pred, np.zeros(16)) == pytest.approx(0.2)


class FakePolicy:
    """Constant-chunk stand-in for a policy: lets us check the iteration end to end on CPU."""

    def __init__(self, bias):
        self.calls = []
        self.bias = bias

    def infer(self, obs):
        self.calls.append(obs)
        return {"actions": np.zeros((50, 16)) + self.bias}


HOLDOUT = Path.home() / ".cache/huggingface/lerobot/ebim/task2_munich_real_holdout"


@pytest.mark.skipif(not HOLDOUT.exists(), reason="converted holdout dataset not present")
def test_evaluate_on_holdout_with_fake_policy():
    ds = ev.load_dataset("ebim/task2_munich_real_holdout")
    policy = FakePolicy(bias=0.0)
    res = ev.evaluate(policy, ds, stride=400)
    assert len(policy.calls) == res["num_samples"] > 0
    obs = policy.calls[0]
    assert set(obs) == {
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
        "observation/state",
        "prompt",
    }
    for k in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        assert obs[k].shape == (224, 224, 3) and obs[k].dtype == np.uint8
    assert obs["observation/state"].shape == (16,)
    assert obs["prompt"] == "Pick up the thermal pad and place it on the target RAM board"
    # every episode contributes its frame 0
    assert res["num_episodes"] == 10
    for h in ("10", "25", "50"):
        assert len(res["mae_per_dim"][h]) == 16
        assert np.isfinite(res["arm_mae"][h])
    assert set(res["first_step_offset"]) == {"p50", "p95", "max"}
    json.dumps(res)  # must be serialisable


def test_build_obs_converts_chw_float_to_hwc_uint8():
    chw = np.zeros((3, 4, 4), dtype=np.float32)
    chw[0] = 130 / 255
    frame = {
        "image": chw,
        "left_wrist_image": chw,
        "right_wrist_image": chw,
        "state": np.arange(16, dtype=np.float32),
    }
    obs = ev.build_obs(frame)
    assert obs["observation/image"].shape == (4, 4, 3)
    assert obs["observation/image"].dtype == np.uint8
    assert obs["observation/image"][0, 0, 0] == 130  # exact despite float round-trip


def test_norm_stats_sha256(tmp_path):
    d = tmp_path / "assets" / "ebim" / "task2_munich_real"
    d.mkdir(parents=True)
    (d / "norm_stats.json").write_text("{}")
    h = ev.norm_stats_sha256(tmp_path, "ebim/task2_munich_real")
    assert len(h) == 64
    assert ev.norm_stats_sha256(tmp_path, "missing/asset") is None


@pytest.mark.skipif(not HOLDOUT.exists(), reason="converted holdout dataset not present")
def test_main_writes_json_with_fake_policy(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    stats = ckpt / "assets" / "ebim" / "task2_munich_real"
    stats.mkdir(parents=True)
    (stats / "norm_stats.json").write_text("{}")
    monkeypatch.setattr(ev, "_create_policy", lambda config_name, checkpoint_dir: FakePolicy(bias=0.1))
    ev.main("pi05_lora_ebim_task2_munich_real", str(ckpt), stride=1000)
    out = json.loads((ckpt / "holdout_eval.json").read_text())
    assert out["norm_stats_sha256"] == ev.norm_stats_sha256(ckpt, "ebim/task2_munich_real")
    assert out["norm_stats_sha256"] is not None
    assert out["config_name"] == "pi05_lora_ebim_task2_munich_real"
    assert out["num_samples"] == 10 + 1  # frame 0 of each episode, plus frame 1000 of episode 0 (1228 frames)


def test_gripper_switch_missed_counts_gt_switches_without_a_prediction():
    gt = np.r_[np.ones(10), np.zeros(10)]
    assert ev.gripper_switch_missed(np.ones(20), gt) is True  # GT switches, the prediction never does
    assert ev.gripper_switch_missed(np.r_[np.ones(12), np.zeros(8)], gt) is False
    assert ev.gripper_switch_missed(np.r_[np.ones(12), np.zeros(8)], np.ones(20)) is False  # spurious, not missed
    assert ev.gripper_switch_missed(np.ones(25), np.r_[np.ones(25)]) is False
    summary = ev.gripper_timing_summary([3.0, None, None, 1.0], [False, True, False, False])
    assert summary == {
        "n_both_switch": 2,
        "n_gt_switch_missed": 1,
        "n_samples": 4,
        "mean_abs_steps": 2.0,
        "max_abs_steps": 3.0,
    }
