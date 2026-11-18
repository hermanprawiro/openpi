import math

import numpy as np
import pytest
from robot_client import safety
from robot_client.config import SafetyConfig
from robot_client.safety import Mode

CFG = SafetyConfig()  # hold 1.0 s, freeze 5.0 s, measured dead 1.5 s, inference dry 5.0 s


def _ages(**overrides: float) -> dict[str, float]:
    ages = {
        "cam/head": 0.03,
        "cam/wrist_left": 0.03,
        "cam/wrist_right": 0.03,
        "arm/left": 0.01,
        "arm/right": 0.01,
        "grip/right": 0.02,
    }
    ages.update({k.replace("__", "/"): v for k, v in overrides.items()})
    return ages


def test_all_fresh_is_run():
    assert safety.evaluate(_ages(), 0.0, fatal_inference=False, cfg=CFG) == (Mode.RUN, "ok")


def test_freeze_on_dead_arm_before_hold_on_camera():
    mode, reason = safety.evaluate(_ages(arm__right=2.0, cam__head=1.2), 0.0, fatal_inference=False, cfg=CFG)
    assert mode is Mode.FREEZE
    assert reason.startswith("arm/right")
    assert "2.00" in reason


def test_hold_then_run():
    mode, reason = safety.evaluate(_ages(cam__wrist_left=1.2), 0.0, fatal_inference=False, cfg=CFG)
    assert mode is Mode.HOLD
    assert "cam/wrist_left" in reason
    assert "1.20" in reason
    assert safety.evaluate(_ages(cam__wrist_left=0.05), 0.0, fatal_inference=False, cfg=CFG)[0] is Mode.RUN


def test_hold_reason_names_the_worst_stream_first():
    mode, reason = safety.evaluate(_ages(cam__head=1.1, grip__right=3.0), 0.0, fatal_inference=False, cfg=CFG)
    assert mode is Mode.HOLD
    assert reason.startswith("grip/right")
    assert "cam/head" in reason


def test_never_seen_stream_is_freeze():
    mode, reason = safety.evaluate(_ages(cam__wrist_right=math.inf), 0.0, fatal_inference=False, cfg=CFG)
    assert mode is Mode.FREEZE
    assert "cam/wrist_right" in reason
    assert "never" in reason


def test_nan_age_is_freeze():
    assert safety.evaluate(_ages(grip__right=math.nan), 0.0, fatal_inference=False, cfg=CFG)[0] is Mode.FREEZE


@pytest.mark.parametrize(
    ("ages", "expected"),
    [
        (_ages(arm__left=1.2), Mode.HOLD),  # past hold, below measured_dead_s
        (_ages(arm__left=1.6), Mode.FREEZE),  # past measured_dead_s (arm only)
        (_ages(grip__right=1.6), Mode.HOLD),  # grippers freeze only past obs_stale_freeze_s
        (_ages(grip__right=5.1), Mode.FREEZE),
        (_ages(cam__head=4.9), Mode.HOLD),
        (_ages(cam__head=5.1), Mode.FREEZE),
    ],
)
def test_per_stream_thresholds(ages, expected):
    assert safety.evaluate(ages, 0.0, fatal_inference=False, cfg=CFG)[0] is expected


def test_inference_dry_and_fatal_freeze():
    mode, reason = safety.evaluate(_ages(), 5.5, fatal_inference=False, cfg=CFG)
    assert mode is Mode.FREEZE
    assert "inference" in reason
    assert safety.evaluate(_ages(), 4.0, fatal_inference=False, cfg=CFG)[0] is Mode.RUN
    mode, reason = safety.evaluate(_ages(), 0.0, fatal_inference=True, cfg=CFG)
    assert mode is Mode.FREEZE
    assert "fatal" in reason


def test_unknown_stream_key_raises():
    with pytest.raises(ValueError, match="lidar/top"):
        safety.evaluate({"lidar/top": 0.0}, 0.0, fatal_inference=False, cfg=CFG)


def test_clamp_step_flags():
    prev = np.zeros(7)
    target = np.array([0.05, -0.05, 0.2, -0.3, 0.0, 0.1, -0.1])
    out, clamped = safety.clamp_step(prev, target, 0.1)
    np.testing.assert_allclose(out, [0.05, -0.05, 0.1, -0.1, 0.0, 0.1, -0.1])
    assert clamped
    out, clamped = safety.clamp_step(prev, target * 0.1, 0.1)
    np.testing.assert_array_equal(out, target * 0.1)
    assert not clamped


def test_limits_with_margin():
    q_min, q_max = np.full(7, -1.0), np.full(7, 1.0)
    q = np.array([0.0, 0.96, 0.94, -0.97, -2.0, 3.0, 0.5])
    out, clamped = safety.clamp_limits(q, q_min, q_max, 0.05)
    np.testing.assert_allclose(out, [0.0, 0.95, 0.94, -0.95, -0.95, 0.95, 0.5])
    assert clamped
    out, clamped = safety.clamp_limits(np.zeros(7), q_min, q_max, 0.05)
    assert not clamped
    with pytest.raises(ValueError, match="margin"):
        safety.clamp_limits(np.zeros(7), q_min, q_max, 1.5)


@pytest.mark.parametrize("fn", [safety.clamp_limits, safety.clamp_step])
def test_non_finite_target_raises(fn):
    q = np.zeros(7)
    q[3] = math.nan
    args = (q, -np.ones(7), np.ones(7), 0.05) if fn is safety.clamp_limits else (np.zeros(7), q, 0.1)
    with pytest.raises(ValueError, match="finite"):
        fn(*args)


def test_lag_and_start_offset():
    measured = np.zeros(7)
    cmd = np.array([0.0, 0.2, 0.0, -0.49, 0.0, 0.0, 0.0])
    assert not safety.lag_exceeded(cmd, measured, 0.5)
    cmd[2] = 0.6
    assert safety.lag_exceeded(cmd, measured, 0.5)
    cmd[2] = math.nan
    assert safety.lag_exceeded(cmd, measured, 0.5)
    assert safety.start_offset(np.array([0.1, -0.35, 0.0, 0, 0, 0, 0]), measured) == pytest.approx(0.35)
    assert safety.start_offset(np.full(7, math.nan), measured) == math.inf
    with pytest.raises(ValueError, match="shape"):
        safety.start_offset(np.zeros(6), measured)
