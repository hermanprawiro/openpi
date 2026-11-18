import numpy as np

from openpi.policies import ebim_task2_policy


def test_outputs_default_20_and_configurable_16():
    a = np.zeros((50, 32))
    assert ebim_task2_policy.EbimTask2Outputs()({"actions": a})["actions"].shape == (50, 20)
    assert ebim_task2_policy.EbimTask2Outputs(action_dim=16)({"actions": a})["actions"].shape == (50, 16)
