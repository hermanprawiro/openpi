import logging
import sys

import numpy as np
import pytest
from robot_client import smoother as sm
from robot_client.config import SmootherConfig

DT = 0.01
N = 7
# The Munich defaults after velocity_scale 0.5.
V_MAX = np.array(SmootherConfig().v_max) * 0.5
A_MAX = np.full(N, 5.0)
J_MAX = np.full(N, 2500.0)
EPS = 1e-9


@pytest.fixture(params=["rate_limiter", "ruckig"])
def make(request):
    if request.param == "ruckig":
        pytest.importorskip("ruckig")
        return lambda: sm.RuckigSmoother(V_MAX, A_MAX, J_MAX, DT)
    return lambda: sm.RateLimiter(V_MAX, A_MAX, DT)


def _run(s, targets, target_vels=None):
    out = []
    for i, target in enumerate(targets):
        vel = np.zeros(N) if target_vels is None else target_vels[i]
        out.append(s.step(target, vel))
    return np.array(out)


def _check_limits(q0, qs, *, jerk=None):
    traj = np.vstack([q0, q0, qs])  # starts at rest
    vel = np.diff(traj, axis=0) / DT
    acc = np.diff(vel, axis=0) / DT
    assert np.all(np.abs(np.diff(traj, axis=0)) <= V_MAX * DT + EPS)
    assert np.all(np.abs(np.diff(vel, axis=0)) <= A_MAX * DT + 1e-6)
    if jerk is not None:
        assert np.all(np.abs(np.diff(acc, axis=0)) <= jerk * DT + 1e-3)


def test_rate_limiter_respects_v_and_a():
    s = sm.RateLimiter(V_MAX, A_MAX, DT)
    q0 = np.zeros(N)
    s.reset(q0)
    target = np.full(N, 1.0)
    qs = _run(s, [target] * 500)  # 5 s
    _check_limits(q0, qs)
    np.testing.assert_allclose(qs[-1], target, atol=1e-6)
    # Small slow joints still need the same limits: a negative step on a subset.
    target2 = target.copy()
    target2[[0, 4]] = -0.3
    qs2 = _run(s, [target2] * 500)
    _check_limits(target, qs2)
    np.testing.assert_allclose(qs2[-1], target2, atol=1e-6)


def test_ruckig_respects_limits():
    pytest.importorskip("ruckig")
    s = sm.RuckigSmoother(V_MAX, A_MAX, J_MAX, DT)
    q0 = np.zeros(N)
    s.reset(q0)
    target = np.linspace(-1.0, 1.0, N)
    qs = _run(s, [target] * 500)
    _check_limits(q0, qs, jerk=J_MAX)
    np.testing.assert_allclose(qs[-1], target, atol=1e-6)


def test_tracks_moving_target_without_stopgo(make):
    s = make()
    q0 = np.zeros(N)
    s.reset(q0)
    speed = 0.2
    t = np.arange(1, 401) * DT  # 4 s ramp
    targets = q0 + speed * t[:, None] * np.ones(N)
    vels = np.full((len(t), N), speed)
    qs = _run(s, targets, vels)
    v = np.diff(np.vstack([q0, qs]), axis=0) / DT
    assert np.all(v >= -1e-9), "velocity flipped sign while tracking a forward ramp"
    steady = v[-100:]
    np.testing.assert_allclose(steady, speed, rtol=0.05)
    np.testing.assert_allclose(qs[-1], targets[-1], atol=0.01)


def test_abrupt_target_stop_lands_without_return(make):
    # A sync-mode chunk end: the interpolated target moves at 1 rad/s, then stops dead.
    s = make()
    q0 = np.zeros(N)
    s.reset(q0)
    speed = 1.0
    t = np.arange(1, 101) * DT
    targets = np.vstack([speed * t[:, None] * np.ones(N), np.full((200, N), speed * t[-1])])
    vels = np.vstack([np.full((100, N), speed), np.zeros((200, N))])
    qs = _run(s, targets, vels)
    assert np.all(np.diff(np.vstack([q0, qs]), axis=0) >= -1e-12), "overshot and came back"
    np.testing.assert_allclose(qs[-1], targets[-1], atol=1e-9)


def test_stop_decelerates_without_return(make):
    s = make()
    q0 = np.zeros(N)
    s.reset(q0)
    far = np.full(N, 2.0)
    qs = _run(s, [far] * 40)  # at speed
    assert np.all(qs[-1] - qs[-2] > 0.5 * A_MAX * DT * DT)
    s.stop()
    # The target is ignored after stop(): even one behind us must not pull the arm back.
    after = _run(s, [q0] * 200)
    traj = np.vstack([qs[-1], after])
    dq = np.diff(traj, axis=0)
    assert np.all(dq >= -1e-12), "position reversed after stop()"
    assert np.all(np.abs(np.diff(dq / DT, axis=0)) <= A_MAX * DT + 1e-6)
    np.testing.assert_allclose(dq[-50:], 0.0, atol=1e-12)  # came to rest and stays
    assert np.all(after[-1] < far)
    # reset() leaves the stopped state and tracks again.
    s.reset(after[-1])
    resumed = _run(s, [after[-1] + 0.05] * 200)
    np.testing.assert_allclose(resumed[-1], after[-1] + 0.05, atol=1e-6)


def test_step_before_reset_and_bad_input_raise(make):
    s = make()
    with pytest.raises(sm.SmootherError, match="reset"):
        s.step(np.zeros(N), np.zeros(N))
    s.reset(np.zeros(N))
    bad = np.zeros(N)
    bad[2] = np.nan
    with pytest.raises(sm.SmootherError, match="finite"):
        s.step(bad, np.zeros(N))
    with pytest.raises(sm.SmootherError, match="shape"):
        s.step(np.zeros(N - 1), np.zeros(N - 1))


def test_ruckig_error_result_raises(monkeypatch):
    ruckig = pytest.importorskip("ruckig")
    s = sm.RuckigSmoother(V_MAX, A_MAX, J_MAX, DT)
    s.reset(np.zeros(N))
    monkeypatch.setattr(s, "_otg", type("Otg", (), {"update": lambda *_: ruckig.Result.ErrorInvalidInput})())
    with pytest.raises(sm.SmootherError, match="ErrorInvalidInput"):
        s.step(np.full(N, 0.1), np.zeros(N))


def test_make_smoother_scales_limits():
    cfg = SmootherConfig(impl="clamp", velocity_scale=0.5)
    s = sm.make_smoother(cfg, DT)
    assert isinstance(s, sm.RateLimiter)
    np.testing.assert_allclose(s.v_max, np.array(cfg.v_max) * 0.5)
    np.testing.assert_allclose(s.a_max, np.array(cfg.a_max) * 0.5)


def test_make_smoother_ruckig():
    pytest.importorskip("ruckig")
    s = sm.make_smoother(SmootherConfig(), DT)
    assert isinstance(s, sm.RuckigSmoother)
    np.testing.assert_allclose(s.j_max, np.array(SmootherConfig().j_max) * 0.5)


def test_make_smoother_fallback(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "ruckig", None)  # `import ruckig` now raises ImportError
    with caplog.at_level(logging.WARNING, logger=sm.__name__):
        s = sm.make_smoother(SmootherConfig(impl="ruckig"), DT)
    assert isinstance(s, sm.RateLimiter)
    assert "ruckig" in caplog.text
    assert "RateLimiter" in caplog.text
