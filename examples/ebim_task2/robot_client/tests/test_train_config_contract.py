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
