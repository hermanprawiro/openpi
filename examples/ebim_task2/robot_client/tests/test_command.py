import numpy as np
import pytest
from robot_client.command import CommandAdapter

SIGNS = np.array([1, 1, -1, 1, 1, 1, 1])  # joint 2 flipped
Q_ACT = np.array([0.1, -0.2, 0.3, -1.5, 0.0, 1.6, 0.4])
REF = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_relative_mode_math():
    a = CommandAdapter("relative_to_activation", SIGNS)
    a.activate(q_activation=Q_ACT, ref_cmd=REF)
    target = Q_ACT + np.array([0.01, 0.02, 0.05, -0.03, 0.0, 0.1, -0.2])
    np.testing.assert_allclose(a(target), REF + SIGNS * (target - Q_ACT), atol=1e-12)
    # Joint 2 moves the other way on the command side; the rest move with the target.
    out = a(target) - REF
    assert out[2] == pytest.approx(-0.05)
    assert out[1] == pytest.approx(0.02)
    # At the activation pose the command is exactly the reference (no jump at activation).
    np.testing.assert_array_equal(a(Q_ACT), REF)


def test_relative_adapter_identity_until_activate():
    a = CommandAdapter("relative_to_activation", SIGNS)
    target = Q_ACT + 0.05
    np.testing.assert_array_equal(a(target), target)
    a.activate(q_activation=Q_ACT, ref_cmd=REF)
    assert not np.allclose(a(target), target)
    a.deactivate()
    np.testing.assert_array_equal(a(target), target)


def test_absolute_mode_is_identity_even_when_activated():
    a = CommandAdapter("absolute", SIGNS)
    target = Q_ACT + 0.05
    a.activate(q_activation=Q_ACT, ref_cmd=REF)
    np.testing.assert_array_equal(a(target), target)


def test_adapter_returns_a_copy():
    a = CommandAdapter("absolute", SIGNS)
    target = Q_ACT.copy()
    out = a(target)
    out[0] = 99.0
    assert target[0] == Q_ACT[0]


@pytest.mark.parametrize(
    ("mode", "signs"),
    [("delta", SIGNS), ("absolute", np.array([1, 0, 1, 1, 1, 1, 1])), ("absolute", np.ones((7, 1)))],
)
def test_adapter_rejects_bad_config(mode, signs):
    with pytest.raises(ValueError, match="mode|signs"):
        CommandAdapter(mode, signs)


def test_activate_rejects_bad_reference():
    a = CommandAdapter("relative_to_activation", SIGNS)
    with pytest.raises(ValueError, match="q_activation: shape"):
        a.activate(q_activation=Q_ACT[:6], ref_cmd=REF)
    with pytest.raises(ValueError, match="ref_cmd: not finite"):
        a.activate(q_activation=Q_ACT, ref_cmd=np.full(7, np.nan))
    np.testing.assert_array_equal(a(Q_ACT), Q_ACT)  # a refused activation leaves it inactive
