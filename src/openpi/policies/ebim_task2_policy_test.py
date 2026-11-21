import numpy as np

from openpi.policies import ebim_task2_policy


def test_outputs_default_20_and_configurable_16():
    a = np.zeros((50, 32))
    assert ebim_task2_policy.EbimTask2Outputs()({"actions": a})["actions"].shape == (50, 20)
    assert ebim_task2_policy.EbimTask2Outputs(action_dim=16)({"actions": a})["actions"].shape == (50, 16)


def _munich16(prefix_shape=()):
    # L arm 1..7, L knuckle 0.5, R arm 11..17, R knuckle 0.7 (state) / width % (actions).
    x = np.zeros((*prefix_shape, 16))
    x[..., 0:7] = np.arange(1, 8)
    x[..., 7] = 0.5
    x[..., 8:15] = np.arange(11, 18)
    x[..., 15] = 0.7
    return x


def test_munich_to_sim_layout_places_arms_and_grippers():
    out = ebim_task2_policy.MunichToSimLayout()({"state": _munich16(), "actions": _munich16((50,)), "prompt": "p"})
    s, a = out["state"], out["actions"]
    assert s.shape == (20,)
    assert a.shape == (50, 20)
    np.testing.assert_array_equal(s[3:10], np.arange(1, 8))
    np.testing.assert_array_equal(s[10:17], np.arange(11, 18))
    # Base twist and spine are not used on the real robot.
    np.testing.assert_array_equal(s[[0, 1, 2, 19]], 0)
    # State gripper is knuckle rad (0 = open), negated so it rises with opening like sim open-fraction.
    np.testing.assert_allclose(s[17:19], [-0.5, -0.7])
    # Action gripper is width % (high = open), same direction as sim: unchanged.
    np.testing.assert_allclose(a[:, 17:19], np.broadcast_to([0.5, 0.7], (50, 2)))
    assert out["prompt"] == "p"


def test_munich_to_sim_layout_without_actions():
    out = ebim_task2_policy.MunichToSimLayout()({"state": _munich16()})
    assert "actions" not in out


def test_sim_to_munich_layout_inverts_the_action_remap():
    munich = _munich16((50,))
    sim = ebim_task2_policy.MunichToSimLayout()({"state": _munich16(), "actions": munich})["actions"]
    padded = np.concatenate([sim, np.full((50, 12), 99.0)], axis=-1)  # model output is padded to 32
    out = ebim_task2_policy.SimToMunichLayout()({"actions": padded, "state": np.zeros(32)})
    assert set(out) == {"actions"}
    np.testing.assert_array_equal(out["actions"], munich)


def test_munich_to_sim_layout_rejects_wrong_width():
    import pytest

    with pytest.raises(ValueError, match="16"):
        ebim_task2_policy.MunichToSimLayout()({"state": np.zeros(20)})
