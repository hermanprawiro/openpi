from eval_client import actions
from eval_client import constants
import numpy as np
import pytest


class TestGripperDriverRad:
    def test_open(self):
        assert actions.gripper_driver_rad(1.0) == 0.0

    def test_closed(self):
        assert actions.gripper_driver_rad(0.0) == constants.GRIPPER_CLOSED_RAD

    def test_round_trip_with_open_fraction(self):
        from eval_client import obs

        for frac in (0.0, 0.25, 0.5, 1.0):
            assert obs.gripper_open_fraction(actions.gripper_driver_rad(frac)) == pytest.approx(frac)

    def test_clips_out_of_range(self):
        assert actions.gripper_driver_rad(1.5) == 0.0
        assert actions.gripper_driver_rad(-0.5) == constants.GRIPPER_CLOSED_RAD


class TestPedalToken:
    def test_dead_band(self):
        assert actions.pedal_token(0.1, 0.1, 0.2, threshold=0.25) == "NONE"

    def test_at_threshold_is_active(self):
        assert actions.pedal_token(0.25, 0.0, 0.0, threshold=0.25) == "FWD"

    def test_linear_tokens(self):
        assert actions.pedal_token(0.5, 0.0, 0.0, threshold=0.25) == "FWD"
        assert actions.pedal_token(-0.5, 0.0, 0.0, threshold=0.25) == "BACK"
        assert actions.pedal_token(0.0, 0.5, 0.0, threshold=0.25) == "A"
        assert actions.pedal_token(0.0, -0.5, 0.0, threshold=0.25) == "B"

    def test_angular_tokens(self):
        assert actions.pedal_token(0.0, 0.0, 1.2, threshold=0.25) == "A+C"
        assert actions.pedal_token(0.0, 0.0, -1.2, threshold=0.25) == "B+C"

    def test_yaw_rescaled_before_comparison(self):
        # wz=1.2 scales to 0.5, beating vx=0.3.
        assert actions.pedal_token(0.3, 0.0, 1.2, threshold=0.25) == "A+C"
        # wz=0.6 scales to 0.25, losing to vx=0.3.
        assert actions.pedal_token(0.3, 0.0, 0.6, threshold=0.25) == "FWD"

    def test_scaled_yaw_below_threshold(self):
        # wz=0.5 scales to ~0.21 < 0.25.
        assert actions.pedal_token(0.0, 0.0, 0.5, threshold=0.25) == "NONE"


class TestSplitAction:
    def _action(self):
        action = np.zeros(20, dtype=np.float32)
        action[0:3] = [0.5, 0.0, 0.0]
        action[3:10] = np.linspace(0.1, 0.7, 7)
        action[10:17] = np.linspace(-0.1, -0.7, 7)
        action[17] = 1.0
        action[18] = 0.0
        action[19] = 0.4
        return action

    def test_fields(self):
        cmd = actions.split_action(self._action(), pedal_threshold=0.25)
        np.testing.assert_allclose(cmd.left_arm, np.linspace(0.1, 0.7, 7), rtol=1e-6)
        np.testing.assert_allclose(cmd.right_arm, np.linspace(-0.1, -0.7, 7), rtol=1e-6)
        assert cmd.left_grip_rad == 0.0
        assert cmd.right_grip_rad == constants.GRIPPER_CLOSED_RAD
        assert cmd.spine == pytest.approx(0.4)
        assert cmd.pedal == "FWD"

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="Bad action shape"):
            actions.split_action(np.zeros(19), pedal_threshold=0.25)
