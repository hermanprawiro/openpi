import numpy as np
import pytest
from robot_client import layout


def test_constants():
    assert layout.LAYOUT_ID == "munich16"
    assert layout.STATE_DIM == layout.ACTION_DIM == 16
    assert len(layout.DATASET_STATE_INDICES) == layout.STATE_DIM
    assert len(layout.DATASET_ACTION_INDICES) == layout.ACTION_DIM
    assert (*range(7), *range(8, 15)) == layout.ARM_DIMS
    assert layout.GRIPPER_INDEX == {"left": 7, "right": 15}


def test_metadata_matches_policy():
    layout.validate_server_metadata(dict(layout.POLICY_METADATA), layout.PROMPT)


def test_metadata_mismatch_lists_keys():
    meta = dict(layout.POLICY_METADATA, action_dim=20, config="pi05_ebim_task2_fixpos_200_and_v5")
    with pytest.raises(layout.LayoutMismatchError, match="action_dim.*16.*20"):
        layout.validate_server_metadata(meta, layout.PROMPT)


def test_sim_server_without_layout_is_refused():
    with pytest.raises(layout.LayoutMismatchError, match="not a munich16 policy"):
        layout.validate_server_metadata({"config": "pi05_ebim_task2_fixpos_200_and_v5"}, layout.PROMPT)


def test_prompt_mismatch_refused():
    with pytest.raises(layout.LayoutMismatchError, match="prompt"):
        layout.validate_server_metadata(dict(layout.POLICY_METADATA), layout.PROMPT + ".")


def test_validate_chunk_requires_exact_width_and_finite():
    for w in (15, 20, 32):
        with pytest.raises(ValueError, match="width"):
            layout.validate_chunk(np.zeros((50, w)))
    bad = np.zeros((50, 16))
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        layout.validate_chunk(bad)
    assert layout.validate_chunk(np.zeros((50, 16))).shape == (50, 16)


def test_validate_chunk_rejects_bad_ndim_and_empty():
    with pytest.raises(ValueError, match="2-D"):
        layout.validate_chunk(np.zeros(16))
    with pytest.raises(ValueError, match="no rows"):
        layout.validate_chunk(np.zeros((0, 16)))


def test_validate_chunk_returns_float64():
    assert layout.validate_chunk(np.zeros((5, 16), np.float32)).dtype == np.float64
