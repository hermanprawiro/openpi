"""Guards that the openpi TrainConfigs and the robot client's layout module agree.

Skips when `openpi` is not importable (the robot-client Docker image does not ship it).
"""

import pathlib

import pytest
from robot_client import layout

pytest.importorskip("openpi")

from openpi.models import pi0_config
from openpi.training import config as _config

MUNICH_CONFIGS = ["pi05_ebim_task2_munich_real", "pi05_lora_ebim_task2_munich_real"]


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
