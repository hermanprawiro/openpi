import dataclasses
import threading
import time

import numpy as np
import pytest
from robot_client.command import CommandAdapter
from robot_client.config import SafetyConfig
from robot_client.config import SmootherConfig
from robot_client.smoother import RateLimiter
from robot_client.smoother import RuckigSmoother
from robot_client.smoother import SmootherError
from robot_client.streamer import Streamer

N = 7
DT = 0.01
SIDES = ("left", "right")
HOLD = {
    "left": np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8]),
    "right": np.array([0.1, -0.2, 0.1, -2.1, 0.1, 2.0, 0.7]),
}
SAFETY = SafetyConfig()
LIMITS = {side: tuple(np.array(x) for x in SAFETY.joint_limits[side]) for side in SIDES}
# Realistic: the Munich defaults after velocity_scale 0.5.
V_MAX = np.array(SmootherConfig().v_max) * 0.5
A_MAX = np.full(N, 5.0)
J_MAX = np.full(N, 2500.0)


def generous():
    """Lands on any target within one period: published == interpolated target."""
    return RateLimiter(np.full(N, 100.0), np.full(N, 1e5), DT)


@pytest.fixture(params=["rate_limiter", "ruckig"])
def realistic(request):
    if request.param == "ruckig":
        pytest.importorskip("ruckig")
        return lambda: RuckigSmoother(V_MAX, A_MAX, J_MAX, DT)
    return lambda: RateLimiter(V_MAX, A_MAX, DT)


class Robot:
    """Records publishes; `measured` follows the published command unless pinned with `stuck`."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pubs = {side: [] for side in SIDES}
        self.pos = {side: HOLD[side].copy() for side in SIDES}
        self.stuck: dict[str, np.ndarray] = {}

    def publish(self, side, cmd):
        with self.lock:
            self.pubs[side].append(np.array(cmd, copy=True))
            self.pos[side] = np.array(cmd, copy=True)

    def measured(self, side):
        return self.stuck.get(side, self.pos[side]).copy()

    def arr(self, side):
        with self.lock:
            return np.array(self.pubs[side])


def make(robot, *, smoother_factory=generous, safety=SAFETY, adapters=None):
    adapters = adapters or {side: CommandAdapter("absolute", np.ones(N)) for side in SIDES}
    return Streamer(
        SIDES, robot.publish, robot.measured, smoother_factory, adapters, LIMITS, safety, hz=100.0, waypoint_dt=0.05
    )


class Ticker:
    """Fake 100 Hz clock driving `step` directly (no thread)."""

    def __init__(self, streamer):
        self.streamer = streamer
        self.t = 0.0

    def run(self, n):
        for _ in range(n):
            self.t += DT
            self.streamer.step(self.t)


def test_streamer_publishes_hold_before_any_waypoint():
    robot = Robot()
    s = make(robot)
    s.start(HOLD, thread=False)
    Ticker(s).run(10)
    for side in SIDES:
        pubs = robot.arr(side)
        assert len(pubs) == 10
        np.testing.assert_array_equal(pubs, np.tile(HOLD[side], (10, 1)))
        assert s.cycles_since_start(side) == 10
        np.testing.assert_array_equal(s.last_command(side), HOLD[side])
    assert s.cycles == 10


def test_ramp_reaches_waypoint_after_dt():
    robot = Robot()
    s = make(robot)
    s.start(HOLD, thread=False)
    tick = Ticker(s)
    wp = HOLD["left"] + np.linspace(-0.05, 0.05, N)
    s.set_waypoint("left", wp)
    tick.run(5)  # waypoint_dt = 0.05 s = 5 periods
    pubs = robot.arr("left")
    for k in range(5):  # linear interpolation, one fifth per period
        np.testing.assert_allclose(pubs[k], HOLD["left"] + (k + 1) / 5 * (wp - HOLD["left"]), atol=1e-12)
    np.testing.assert_array_equal(pubs[4], wp)
    tick.run(5)
    np.testing.assert_array_equal(robot.arr("left")[-1], wp)  # stays there
    np.testing.assert_array_equal(robot.arr("right"), np.tile(HOLD["right"], (10, 1)))  # other arm untouched


def test_leash_freezes_ramp():
    robot = Robot()
    s = make(robot)
    s.start(HOLD, thread=False)
    tick = Ticker(s)
    behind = HOLD["left"].copy()
    behind[0] -= 0.6
    robot.stuck["left"] = behind
    wp = HOLD["left"].copy()
    wp[0] += 0.1
    s.set_waypoint("left", wp)
    tick.run(10)
    np.testing.assert_array_equal(robot.arr("left"), np.tile(HOLD["left"], (10, 1)))  # target does not advance
    assert s.leash_engaged("left")
    assert not s.leash_engaged("right")
    assert s.leash_events == 1  # one event, not one per cycle
    # The arm catches up: the leash releases and the ramp resumes from where it froze.
    robot.stuck.pop("left")
    tick.run(10)
    assert not s.leash_engaged("left")
    np.testing.assert_array_equal(robot.arr("left")[-1], wp)
    assert s.leash_events == 1


def test_leash_bounds_lag_while_waypoints_march_ahead():
    robot = Robot()
    s = make(robot)
    s.start(HOLD, thread=False)
    tick = Ticker(s)
    robot.stuck["left"] = HOLD["left"].copy()  # the arm does not follow at all
    for k in range(1, 21):  # 0.05 rad per waypoint: would reach 1.0 rad ahead
        s.set_waypoint("left", HOLD["left"] + 0.05 * k)
        tick.run(5)
    lag = np.max(np.abs(robot.arr("left") - HOLD["left"]), axis=1)
    assert lag.max() <= SAFETY.max_track_lag_rad + 1e-12
    assert lag.max() > SAFETY.max_track_lag_rad - 0.05  # it did advance up to the leash
    assert s.leash_engaged("left")
    assert s.leash_events >= 1


def test_limit_clamp_applied_before_adapter():
    robot = Robot()
    signs = np.array([1, 1, -1, 1, 1, 1, 1])
    adapters = {side: CommandAdapter("relative_to_activation", signs) for side in SIDES}
    s = make(robot, adapters=adapters)
    hold = {side: HOLD[side].copy() for side in SIDES}
    hold["right"][2] = 2.8  # q_max[2] = 2.9007, margin 0.05
    robot.stuck["right"] = hold["right"].copy()
    s.start(hold, thread=False)
    tick = Ticker(s)
    tick.run(3)
    adapters["right"].activate(q_activation=robot.measured("right"), ref_cmd=s.last_command("right"))
    wp = hold["right"].copy()
    wp[2] = 3.2  # beyond q_max
    s.set_waypoint("right", wp)
    tick.run(10)
    clamped = wp.copy()
    clamped[2] = LIMITS["right"][1][2] - SAFETY.q_margin_rad
    np.testing.assert_allclose(robot.arr("right")[-1], adapters["right"](clamped), atol=1e-12)
    assert robot.arr("right")[-1][2] < hold["right"][2]  # sign -1: commanded the other way
    assert s.limit_clamps >= 1


def test_relative_activation_is_continuous_and_follows_signs():
    robot = Robot()
    signs = np.array([1, 1, -1, 1, 1, 1, 1])
    adapters = {side: CommandAdapter("relative_to_activation", signs) for side in SIDES}
    s = make(robot, adapters=adapters)
    s.start(HOLD, thread=False)
    tick = Ticker(s)
    tick.run(3)
    q_act = HOLD["left"].copy()
    adapters["left"].activate(q_activation=q_act, ref_cmd=s.last_command("left"))
    robot.stuck["left"] = q_act
    tick.run(3)
    np.testing.assert_array_equal(robot.arr("left"), np.tile(HOLD["left"], (6, 1)))  # no jump at activation
    wp = HOLD["left"].copy()
    wp[2] += 0.1
    wp[0] += 0.1
    s.set_waypoint("left", wp)
    tick.run(10)
    expected = HOLD["left"].copy()
    expected[2] -= 0.1
    expected[0] += 0.1
    np.testing.assert_allclose(robot.arr("left")[-1], expected, atol=1e-12)


def _assert_smooth(q0, pubs):
    traj = np.vstack([q0, q0, pubs])  # starts at rest
    vel = np.diff(traj, axis=0) / DT
    assert np.all(np.abs(vel) <= V_MAX + 1e-9)
    assert np.all(np.abs(np.diff(vel, axis=0)) <= A_MAX * DT + 1e-6)


def test_hold_decelerates_in_place_without_return(realistic):
    robot = Robot()
    s = make(robot, smoother_factory=realistic)
    s.start(HOLD, thread=False)
    tick = Ticker(s)
    for k in range(1, 11):  # 0.4 rad/s on every joint
        s.set_waypoint("left", HOLD["left"] + 0.02 * k)
        tick.run(5)
    n_before = len(robot.arr("left"))
    s.hold("left")
    tick.run(100)
    pubs = robot.arr("left")
    _assert_smooth(HOLD["left"], pubs)
    after = pubs[n_before - 1 :]
    assert np.all(np.diff(after, axis=0) >= -1e-12)  # brakes forward, never comes back
    np.testing.assert_allclose(after[-1], after[-20], atol=1e-12)  # at rest
    assert np.all(after[-1] <= HOLD["left"] + 0.2 + 1e-9)  # never past the last waypoint
    # set_waypoint after a hold resumes from where the arm stopped, without a jump.
    s.set_waypoint("left", HOLD["left"])
    tick.run(150)
    pubs = robot.arr("left")
    _assert_smooth(HOLD["left"], pubs)
    np.testing.assert_allclose(pubs[-1], HOLD["left"], atol=1e-9)


def test_set_waypoint_during_braking_waits_for_rest(realistic):
    robot = Robot()
    s = make(robot, smoother_factory=realistic)
    s.start(HOLD, thread=False)
    tick = Ticker(s)
    for k in range(1, 11):
        s.set_waypoint("left", HOLD["left"] + 0.02 * k)
        tick.run(5)
    s.hold()
    s.set_waypoint("left", HOLD["left"])  # straight away, while still moving forward
    tick.run(200)
    pubs = robot.arr("left")
    _assert_smooth(HOLD["left"], pubs)  # reset() happened at rest: no velocity or acceleration jump
    np.testing.assert_allclose(pubs[-1], HOLD["left"], atol=1e-9)
    np.testing.assert_array_equal(robot.arr("right"), np.tile(HOLD["right"], (len(robot.arr("right")), 1)))


def test_bad_inputs_are_rejected():
    robot = Robot()
    s = make(robot)
    with pytest.raises(ValueError, match="no pose for"):
        s.start({"left": HOLD["left"]}, thread=False)  # right missing
    s.start(HOLD, thread=False)
    with pytest.raises(ValueError, match="shape"):
        s.set_waypoint("left", HOLD["left"][:6])
    with pytest.raises(ValueError, match="not finite"):
        s.set_waypoint("left", np.full(N, np.nan))
    with pytest.raises(KeyError):
        s.set_waypoint("middle", HOLD["left"])


def test_wrong_measured_shape_raises_in_step():
    robot = Robot()
    s = make(robot)
    s.start(HOLD, thread=False)
    robot.stuck["left"] = HOLD["left"][:1]  # would silently broadcast in the lag check
    with pytest.raises(ValueError, match="measured"):
        s.step(0.01)


class FailingSmoother:
    def __init__(self, fail_at):
        self.inner = generous()
        self.fail_at = fail_at
        self.n = 0

    def reset(self, q):
        self.inner.reset(q)

    def stop(self):
        self.inner.stop()

    def step(self, target, target_vel):
        self.n += 1
        if self.n == self.fail_at:
            raise SmootherError("boom")
        return self.inner.step(target, target_vel)


def _wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.005)


def test_step_exception_sets_fault_and_keeps_publishing():
    robot = Robot()
    made = []

    def factory():
        made.append(FailingSmoother(fail_at=5) if not made else generous())
        return made[-1]

    s = make(robot, smoother_factory=factory)
    s.start(HOLD)
    try:
        s.set_waypoint("left", HOLD["left"] + 0.3)  # so the commands are moving when the fault hits
        s.set_waypoint("right", HOLD["right"] + 0.3)
        _wait_for(lambda: s.fault is not None)
        n_at_fault = len(robot.arr("left"))
        _wait_for(lambda: len(robot.arr("left")) >= n_at_fault + 10)  # thread alive, still publishing
        assert "boom" in s.fault
        assert not s.healthy()
        assert s.cycles == 4
        for side in SIDES:
            pubs = robot.arr(side)
            np.testing.assert_array_equal(pubs[3:], np.tile(pubs[3], (len(pubs) - 3, 1)))  # last command, repeated
            np.testing.assert_array_equal(s.last_command(side), pubs[3])
    finally:
        s.stop()
    assert not s.healthy()


def test_thread_publishes_at_rate():
    robot = Robot()
    s = make(robot)
    s.start(HOLD)
    try:
        time.sleep(0.2)
        assert s.healthy()
        time.sleep(0.8)
        assert s.fault is None
    finally:
        s.stop()
    for side in SIDES:
        assert 85 <= len(robot.arr(side)) <= 115
        assert s.max_gap_s[side] < 0.05
    n = len(robot.arr("left"))
    time.sleep(0.05)
    assert len(robot.arr("left")) == n  # publishing stops at stop()
    assert not s.healthy()


def test_unhealthy_before_start_and_without_thread():
    robot = Robot()
    s = make(robot)
    assert not s.healthy()
    s.start(HOLD, thread=False)
    Ticker(s).run(3)
    assert not s.healthy()  # no thread


def test_safety_lag_is_configurable():
    robot = Robot()
    s = make(robot, safety=dataclasses.replace(SAFETY, max_track_lag_rad=0.05))
    s.start(HOLD, thread=False)
    robot.stuck["left"] = HOLD["left"].copy()
    s.set_waypoint("left", HOLD["left"] + 0.1)
    Ticker(s).run(10)
    assert np.max(np.abs(robot.arr("left")[-1] - HOLD["left"])) <= 0.05 + 1e-12
    assert s.leash_engaged("left")


class OvershootingSmoother:
    """Tracks the target exactly but adds a fixed offset: forces an output beyond the joint limits."""

    def __init__(self, offset):
        self.offset = offset
        self.stopped = False
        self.resets = 0

    def reset(self, q):
        self.resets += 1

    def stop(self):
        self.stopped = True

    def step(self, target, target_vel):
        return np.asarray(target) + self.offset


def test_hard_clamp_after_smoother():
    robot = Robot()
    offset = np.zeros(N)
    offset[2] = 0.2  # q_max[2] = 2.9007; the margin-clamped target is 2.8507, +0.2 goes past q_max
    made = []

    def factory():
        made.append(OvershootingSmoother(offset))
        return made[-1]

    s = make(robot, smoother_factory=factory)
    hold = {side: HOLD[side].copy() for side in SIDES}
    hold["left"][2] = 2.8
    robot.stuck["left"] = hold["left"].copy()
    s.start(hold, thread=False)
    tick = Ticker(s)
    tick.run(3)
    assert s.hard_clamps == 3  # left only: 2.8 + 0.2 > q_max
    q_max = LIMITS["left"][1]
    pubs = robot.arr("left")
    assert np.all(pubs[:, 2] == q_max[2])
    np.testing.assert_array_equal(pubs[:, [0, 1, 3, 4, 5, 6]], np.tile(hold["left"][[0, 1, 3, 4, 5, 6]], (3, 1)))
    wp = hold["left"].copy()
    wp[2] = 3.2
    s.set_waypoint("left", wp)
    tick.run(10)
    assert np.all(robot.arr("left")[:, 2] <= q_max[2])
    assert s.hard_clamps == 13
    assert made[0].resets == 1  # only the seed in start(): the clamp never resets the smoother
    assert s.fault is None


def test_healthy_uses_stream_gap_freeze_s():
    robot = Robot()
    stalled, release = threading.Event(), threading.Event()
    offset = [0.0]

    def publish(side, cmd):
        if offset[0]:
            stalled.set()
            release.wait(5.0)
        robot.publish(side, cmd)

    s = Streamer(
        SIDES,
        publish,
        robot.measured,
        generous,
        {side: CommandAdapter("absolute", np.ones(N)) for side in SIDES},
        LIMITS,
        dataclasses.replace(SAFETY, stream_gap_freeze_s=0.5),
        clock=lambda: time.monotonic() + offset[0],
    )
    s.start(HOLD)
    try:
        _wait_for(lambda: len(robot.arr("right")) >= 5)
        offset[0] = 1e-9  # stall the next publish; the clock is unchanged for now
        assert stalled.wait(2.0)
        offset[0] = 0.3  # last publish ~0.3 s ago: within 0.5
        assert s.healthy()
        offset[0] = 0.6  # ~0.6 s ago: beyond 0.5
        assert not s.healthy()
    finally:
        release.set()
        s.stop()
