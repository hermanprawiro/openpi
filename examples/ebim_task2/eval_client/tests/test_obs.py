import math

from eval_client import constants
from eval_client import obs
import numpy as np
import pytest


def _full_joint_map(**overrides):
    joint_map = {}
    for i, name in enumerate(constants.LEFT_JOINTS):
        joint_map[name] = 0.1 * (i + 1)
    for i, name in enumerate(constants.RIGHT_JOINTS):
        joint_map[name] = -0.1 * (i + 1)
    joint_map[constants.LEFT_GRIPPER_DRIVER] = 0.2
    joint_map[constants.RIGHT_GRIPPER_DRIVER] = 0.6
    joint_map[constants.SPINE_JOINT] = 0.35
    joint_map.update(overrides)
    return joint_map


class TestResolveJoint:
    def test_canonical_name(self):
        assert obs.resolve_joint({"left_fr3v2_joint3": 1.5}, "left_fr3v2_joint3") == 1.5

    def test_usd_arm_fallback(self):
        assert obs.resolve_joint({"left_fr3v2_1_joint3": 1.5}, "left_fr3v2_joint3") == 1.5

    def test_usd_finger_fallbacks(self):
        assert obs.resolve_joint({"left_fr3v2_finger_joint1": 0.4}, "left_right_finger_joint") == 0.4
        assert obs.resolve_joint({"right_fr3v2_finger_joint1": 0.4}, "right_right_finger_joint") == 0.4

    def test_missing_returns_nan(self):
        assert math.isnan(obs.resolve_joint({}, "left_fr3v2_joint1"))

    def test_non_finite_value_falls_through(self):
        joint_map = {"left_fr3v2_joint3": math.nan, "left_fr3v2_1_joint3": 2.0}
        assert obs.resolve_joint(joint_map, "left_fr3v2_joint3") == 2.0


class TestGripperOpenFraction:
    def test_fully_open(self):
        assert obs.gripper_open_fraction(0.0) == 1.0

    def test_fully_closed(self):
        assert obs.gripper_open_fraction(constants.GRIPPER_CLOSED_RAD) == 0.0

    def test_midpoint(self):
        assert obs.gripper_open_fraction(0.4) == pytest.approx(0.5)

    def test_clipped_beyond_closed(self):
        assert obs.gripper_open_fraction(1.2) == 0.0

    def test_nan_propagates(self):
        assert math.isnan(obs.gripper_open_fraction(math.nan))


class TestBuildState:
    def test_layout(self):
        state = obs.build_state(_full_joint_map(), (0.5, -0.25, 1.2))
        assert state.shape == (20,)
        assert state.dtype == np.float32
        np.testing.assert_allclose(state[0:3], [0.5, -0.25, 1.2])
        np.testing.assert_allclose(state[3:10], [0.1 * (i + 1) for i in range(7)], rtol=1e-6)
        np.testing.assert_allclose(state[10:17], [-0.1 * (i + 1) for i in range(7)], rtol=1e-6)
        assert state[17] == pytest.approx(1.0 - 0.2 / 0.8)
        assert state[18] == pytest.approx(1.0 - 0.6 / 0.8)
        assert state[19] == pytest.approx(0.35)

    def test_missing_joint_is_nan(self):
        joint_map = _full_joint_map()
        del joint_map["right_fr3v2_joint4"]
        state = obs.build_state(joint_map, (0.0, 0.0, 0.0))
        assert math.isnan(state[13])
        assert np.isfinite(np.delete(state, 13)).all()

    def test_fallback_joint_names(self):
        joint_map = _full_joint_map()
        joint_map["left_fr3v2_1_joint2"] = joint_map.pop("left_fr3v2_joint2")
        state = obs.build_state(joint_map, (0.0, 0.0, 0.0))
        assert state[4] == pytest.approx(0.2)


class TestDecodeImage:
    def test_rgb8_with_row_padding(self):
        height, width = 2, 3
        row = bytes(range(width * 3))
        padded = row + b"\x00\x00\x00"  # step = 12, 3 pad bytes per row
        image = obs.decode_image(padded * height, height, width, step=12, encoding="rgb8")
        assert image.shape == (2, 3, 3)
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(image[0].ravel(), np.frombuffer(row, dtype=np.uint8))

    def test_bgr8_flips_channels(self):
        pixel = bytes([10, 20, 30])
        image = obs.decode_image(pixel, 1, 1, step=3, encoding="bgr8")
        np.testing.assert_array_equal(image[0, 0], [30, 20, 10])

    def test_result_is_contiguous(self):
        image = obs.decode_image(bytes(3), 1, 1, step=3, encoding="bgr8")
        assert image.flags["C_CONTIGUOUS"]


class TestMakeObservation:
    def test_keys_shapes_dtypes(self):
        state = np.zeros(20, dtype=np.float32)
        images = {
            "head": np.zeros((720, 1280, 3), dtype=np.uint8),
            "wrist_left": np.zeros((480, 848, 3), dtype=np.uint8),
            "wrist_right": np.zeros((480, 848, 3), dtype=np.uint8),
        }
        observation = obs.make_observation(state, images, "do the task")
        assert set(observation) == {
            "observation/image",
            "observation/left_wrist_image",
            "observation/right_wrist_image",
            "observation/state",
            "prompt",
        }
        assert observation["prompt"] == "do the task"
        assert observation["observation/state"] is state
        for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
            assert observation[key].shape == (224, 224, 3)
            assert observation[key].dtype == np.uint8
