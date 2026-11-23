"""Guards that the openpi TrainConfigs and the robot client's layout module agree.

Skips when `openpi` is not importable (the robot-client Docker image does not ship it).
"""

import pathlib

import pytest
from robot_client import layout

pytest.importorskip("openpi")

from openpi.models import pi0_config
from openpi.training import config as _config

MUNICH_CONFIGS = [
    "pi05_ebim_task2_munich_real",
    "pi05_lora_ebim_task2_munich_real",
    "pi05_ebim_task2_munich_real_from_sim",
]


@pytest.mark.parametrize("name", MUNICH_CONFIGS)
def test_munich_configs_match_layout(name):
    cfg = _config.get_config(name)
    assert cfg.policy_metadata == layout.POLICY_METADATA
    assert cfg.model.action_horizon == layout.ACTION_HORIZON
    assert cfg.data.action_dim == 16
    assert cfg.data.use_delta_actions is False
    assert cfg.data.repo_id == "ebim/task2_munich_real"


def test_both_configs_share_data_config():
    full = _config.get_config("pi05_ebim_task2_munich_real")
    lora = _config.get_config("pi05_lora_ebim_task2_munich_real")
    assert full.data is lora.data


def test_delta_actions_with_16_dims_rejected():
    data_config = _config.LeRobotEbimTask2DataConfig(
        repo_id="ebim/task2_munich_real",
        use_delta_actions=True,
        action_dim=16,
    )
    with pytest.raises(ValueError, match="action_dim"):
        data_config.create(pathlib.Path("/nonexistent_assets"), pi0_config.Pi0Config(pi05=True))


def test_from_sim_config_remaps_to_sim_slots_and_back():
    import numpy as np

    cfg = _config.get_config("pi05_ebim_task2_munich_real_from_sim")
    assert cfg.data.sim_layout_remap is True
    assert _config.get_config("pi05_ebim_task2_munich_real").data.sim_layout_remap is False
    data = cfg.data.create(pathlib.Path("/nonexistent_assets"), cfg.model)
    frame = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/left_wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/right_wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.arange(16, dtype=np.float32),
        "actions": np.tile(np.arange(16, dtype=np.float32), (50, 1)),
        "prompt": layout.POLICY_METADATA["prompt"],
    }
    for t in data.data_transforms.inputs:
        frame = t(frame)
    assert frame["state"].shape == (20,)
    assert frame["actions"].shape == (50, 20)
    model_out = {"actions": np.concatenate([frame["actions"], np.zeros((50, 12), np.float32)], axis=-1)}
    for t in data.data_transforms.outputs:
        model_out = t(model_out)
    np.testing.assert_array_equal(model_out["actions"], np.tile(np.arange(16, dtype=np.float32), (50, 1)))


@pytest.mark.parametrize(("action_dim", "delta"), [(20, False), (16, True)])
def test_sim_layout_remap_needs_absolute_munich16(action_dim, delta):
    data_config = _config.LeRobotEbimTask2DataConfig(
        repo_id="ebim/task2_munich_real",
        use_delta_actions=delta,
        action_dim=action_dim,
        sim_layout_remap=True,
    )
    with pytest.raises(ValueError, match="sim_layout_remap"):
        data_config.create(pathlib.Path("/nonexistent_assets"), pi0_config.Pi0Config(pi05=True))


def _write_norm_stats(directory, dim):
    import numpy as np

    from openpi.shared import normalize

    stats = normalize.NormStats(mean=np.zeros(dim), std=np.ones(dim), q01=-np.ones(dim), q99=np.ones(dim))
    normalize.save(directory, {"state": stats, "actions": stats})


@pytest.mark.parametrize(
    ("config_name", "wrong_dim"),
    [("pi05_ebim_task2_munich_real", 20), ("pi05_ebim_task2_munich_real_from_sim", 16)],
)
def test_server_refuses_checkpoint_with_other_configs_norm_stats(tmp_path, config_name, wrong_dim):
    """A from_sim checkpoint served with the 16-dim config (or the reverse) must fail before the model loads."""
    from openpi.policies import policy_config

    _write_norm_stats(tmp_path / "assets" / "ebim" / "task2_munich_real", wrong_dim)
    # No params/ in tmp_path: reaching the model load would raise something else, so the norm stats check runs first.
    with pytest.raises(ValueError, match=rf"state norm stats are {wrong_dim}-dim"):
        policy_config.create_trained_policy(_config.get_config(config_name), tmp_path)


@pytest.mark.parametrize(
    ("config_name", "dim"),
    [
        ("pi05_ebim_task2_munich_real", 16),
        ("pi05_lora_ebim_task2_munich_real", 16),
        ("pi05_ebim_task2_munich_real_from_sim", 20),
        ("pi05_ebim_task2_fixpos_200_and_v5", 20),
    ],
)
def test_norm_stats_dim_matches_each_config(tmp_path, config_name, dim):
    import numpy as np

    cfg = _config.get_config(config_name)
    data = cfg.data.create(pathlib.Path("/nonexistent_assets"), cfg.model)
    assert data.norm_stats_dim == dim
    _write_norm_stats(tmp_path, dim)
    from openpi.shared import normalize

    data.check_norm_stats(normalize.load(tmp_path), "test")  # matching dims: no error
    with pytest.raises(ValueError, match="norm stats"):
        data.check_norm_stats({"state": normalize.NormStats(mean=np.zeros(dim + 4), std=np.ones(dim + 4))}, "test")


def test_training_refuses_assets_with_other_configs_norm_stats(tmp_path):
    cfg = _config.get_config("pi05_ebim_task2_munich_real_from_sim")
    _write_norm_stats(tmp_path / cfg.name / "ebim" / "task2_munich_real", 16)
    with pytest.raises(ValueError, match="state norm stats are 16-dim"):
        cfg.data.create(tmp_path / cfg.name, cfg.model)
