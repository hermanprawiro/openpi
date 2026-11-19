"""Takeover / release / Session ordering against fakes: a real `Streamer` driven by a fake sleep (no thread,
no real waiting), a FakeCM per arm recording every call into one shared event log, and a FakeGraph."""

import logging
from pathlib import Path

import numpy as np
import pytest
from robot_client import takeover as takeover_module
from robot_client.command import CommandAdapter
from robot_client.config import load_site_config
from robot_client.smoother import RateLimiter
from robot_client.streamer import Streamer
from robot_client.takeover import EXTERNAL_WAIT_S
from robot_client.takeover import RELEASE_RETRY_S
from robot_client.takeover import RELEASE_TAIL_S
from robot_client.takeover import ControllerStillActive
from robot_client.takeover import Session
from robot_client.takeover import TakeoverError
from robot_client.takeover import release
from robot_client.takeover import session_sides
from robot_client.takeover import takeover

MUNICH_YAML = Path(__file__).resolve().parents[1] / "robot_site.munich.yaml"
N = 7
DT = 0.01
SIDES = ("left", "right")
CTRL = "joint_impedance_controller"
MEASURED = {
    "left": np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8]),
    "right": np.array([0.1, -0.2, 0.1, -2.1, 0.1, 2.0, 0.7]),
}


class FakeTime:
    """Fake `sleep`: advances the clock and ticks the streamer at 100 Hz over the slept interval. `hooks`
    run at the start of every sleep (they may raise, like a Ctrl-C arriving during the sleep)."""

    def __init__(self):
        self.t = 0.0
        self.next_tick = 0.0
        self.streamer = None
        self.hooks = []
        self.sleeps = []

    def sleep(self, dt):
        self.sleeps.append(dt)
        for hook in list(self.hooks):
            hook(dt)
        end = self.t + dt
        while self.streamer is not None and self.streamer.running and self.next_tick <= end + 1e-12:
            self.streamer.step(self.next_tick)
            self.next_tick += DT
        self.t = end


class FakeGraph:
    def __init__(self, rig, uptime0):
        self.rig = rig
        self.uptime0 = uptime0
        self.measured_q = {side: q.copy() for side, q in MEASURED.items()}
        self.ages = dict.fromkeys(MEASURED, 0.005)  # seconds since the last measured sample, per side
        self.publishers = {}  # topic -> endpoints incl. ours (default 1)
        self.foreign = {}  # topic -> foreign messages in the window
        self.foreign_hook = None

    def uptime_s(self):
        return self.rig.time.t + self.uptime0

    def measured(self, side):
        q = self.measured_q.get(side)
        return None if q is None else q.copy()

    def measured_age(self, side):
        return self.ages.get(side, float("inf")) if self.measured_q.get(side) is not None else float("inf")

    def count_publishers(self, topic):
        return self.publishers.get(topic, 1)

    def foreign_messages(self, topic, window_s):
        self.rig.log.append(("foreign", topic, window_s))
        if self.foreign_hook is not None:
            self.foreign_hook(topic)
        return self.foreign.get(topic, 0)


class FakeCM:
    """One controller_manager (one arm). `apply=False`: switch returns True but nothing changes."""

    def __init__(self, side, rig):
        self.side = side
        self.rig = rig
        self.states = {CTRL: "inactive", "joint_state_broadcaster": "active"}
        self.apply = True
        self.list_error = None
        self.switch_error = None  # raised for any switch
        self.deactivate_error = None  # raised once, by the next deactivation switch
        self.after_switch_hook = None  # called after a switch took effect (may raise, like a Ctrl-C during the reply)
        self.list_hook = None
        self.operator_activates_on_list = None  # external mode: the operator activates on the n-th list call
        self.operator_deactivates_on_list = None  # the operator deactivates (own tooling) on the n-th list call
        self.refuse_deactivate = 0  # this many deactivation switches return False (and change nothing)
        self.lists = 0

    def list_controllers(self):
        self.lists += 1
        self.rig.log.append(("list", self.side))
        if self.list_hook is not None:
            self.list_hook()
        if self.list_error is not None:
            raise self.list_error
        if self.operator_activates_on_list is not None and self.lists >= self.operator_activates_on_list:
            self.operator_activates_on_list = None
            self.states[CTRL] = "active"
        if self.operator_deactivates_on_list is not None and self.lists >= self.operator_deactivates_on_list:
            self.operator_deactivates_on_list = None
            self.states[CTRL] = "inactive"
            self.rig.log.append(("operator_deactivate", self.side))
        return dict(self.states)

    def switch(self, activate, deactivate, timeout_s):
        self.rig.log.append(("switch", self.side, tuple(activate), tuple(deactivate), timeout_s, self.rig.time.t))
        if self.switch_error is not None:
            raise self.switch_error
        if deactivate and self.deactivate_error is not None:
            error, self.deactivate_error = self.deactivate_error, None  # one-shot
            raise error
        if deactivate and self.refuse_deactivate:
            self.refuse_deactivate -= 1
            return False
        if self.apply:
            for c in activate:
                self.states[c] = "active"
            for c in deactivate:
                self.states[c] = "inactive"
        if self.after_switch_hook is not None:
            self.after_switch_hook(activate, deactivate)
        return True


class SpyAdapter(CommandAdapter):
    def __init__(self, side, log, mode, signs):
        super().__init__(mode, signs)
        self.side = side
        self.log = log

    def activate(self, q_activation, ref_cmd):
        self.log.append(("activate", self.side, np.array(q_activation), np.array(ref_cmd)))
        super().activate(q_activation, ref_cmd)

    def deactivate(self):
        self.log.append(("adapter_deactivate", self.side))
        super().deactivate()


class ManualStreamer(Streamer):
    """The real streamer without its thread: ticked by FakeTime; healthy while started, unfaulted, unstopped."""

    def __init__(self, *args, log, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = log
        self.running = False
        self.stops = 0

    def start(self, hold, *, thread=True):
        super().start(hold, thread=False)
        self.running = True

    def hold(self, side=None):
        self.log.append(("hold", side))
        super().hold(side)

    def stop(self):
        self.stops += 1
        self.running = False
        self.log.append(("stop",))
        super().stop()

    def healthy(self):
        return self.running and self.fault is None


class FakeRecorder:
    def __init__(self):
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))


def generous():
    return RateLimiter(np.full(N, 100.0), np.full(N, 1e5), DT)


class Rig:
    def __init__(self, overrides=(), *, uptime0=10.0, sides=None):
        self.cfg = load_site_config(MUNICH_YAML, list(overrides), {})
        self.log = []
        self.time = FakeTime()
        self.graph = FakeGraph(self, uptime0)
        self.sides = tuple(sides or session_sides(self.cfg))
        self.cms = {side: FakeCM(side, self) for side in SIDES}
        self.adapters = {
            side: SpyAdapter(side, self.log, self.cfg.command_mode, np.array(self.cfg.joint_signs[side], dtype=float))
            for side in SIDES
        }
        limits = {side: tuple(np.array(x) for x in self.cfg.safety.joint_limits[side]) for side in SIDES}
        self.streamer = ManualStreamer(
            self.sides,
            self.publish,
            self.graph.measured,
            generous,
            {side: self.adapters[side] for side in self.sides},
            {side: limits[side] for side in self.sides},
            self.cfg.safety,
            hz=100.0,
            waypoint_dt=0.05,
            log=self.log,
        )
        self.time.streamer = self.streamer
        self.recorder = FakeRecorder()
        self.acks = 0
        self.ack_after = None  # the operator presses Enter on this many-th poll of the still-active prompt

    def ack(self):
        self.acks += 1
        assert self.acks < 100_000, "the release kept waiting for an operator Enter the test never gives"
        return self.ack_after is not None and self.acks >= self.ack_after

    def publish(self, side, cmd):
        self.log.append(("pub", side, np.array(cmd, copy=True), self.time.t))

    def session(self):
        return Session(
            self.cfg,
            self.sides,
            self.cms,
            self.graph,
            self.streamer,
            self.adapters,
            recorder=self.recorder,
            sleep=self.time.sleep,
            clock=lambda: self.time.t,
            ack=self.ack,
        )

    def start(self):
        self.streamer.start({side: MEASURED[side] for side in self.sides})

    def takeover(self, side, **kwargs):
        return takeover(
            side,
            self.cfg.arms[side],
            self.cms[side],
            self.graph,
            self.streamer,
            self.adapters[side],
            self.cfg.takeover,
            sleep=self.time.sleep,
            **kwargs,
        )

    def index(self, pred, start=0):
        return next(i for i in range(start, len(self.log)) if pred(self.log[i]))

    def switches(self, side=None, *, kind):
        """kind: 'activate' or 'deactivate' switch calls, in order."""
        pos = 2 if kind == "activate" else 3
        return [e for e in self.log if e[0] == "switch" and (side is None or e[1] == side) and e[pos] == (CTRL,)]

    def pubs(self, side, log=None):
        return [e for e in (self.log if log is None else log) if e[0] == "pub" and e[1] == side]


def lists_at_exit(rig):
    return rig.log.index(("exit",))


def is_switch(side, kind):
    pos = 2 if kind == "activate" else 3
    return lambda e: e[0] == "switch" and e[1] == side and e[pos] == (CTRL,)


def raise_once(exc, when=lambda dt: True):
    state = {"done": False}

    def hook(dt):
        if not state["done"] and when(dt):
            state["done"] = True
            raise exc

    return hook


# ---- ordering ----


@pytest.mark.parametrize("command_mode", ["absolute", "relative_to_activation"])
def test_takeover_order(command_mode):
    rig = Rig(
        [f"command_mode={command_mode}", "joint_signs.left=[1, -1, 1, 1, -1, 1, 1]"],
        uptime0=0.0,  # the node just started: the discovery wait applies
    )
    with rig.session() as session:
        assert session.activated_by_us == {"left": True, "right": True}
        for side in SIDES:
            i = rig.index(is_switch(side, "activate"))
            # adapter.activate immediately before switch, with q_activation = measured, ref = the streamed hold
            assert rig.log[i - 1][:2] == ("activate", side)
            np.testing.assert_allclose(rig.log[i - 1][2], MEASURED[side], atol=1e-12)
            np.testing.assert_allclose(rig.log[i - 1][3], MEASURED[side], atol=1e-12)
            # >= warm_cycles hold publishes equal to measured before the switch
            before = rig.pubs(side, rig.log[:i])
            assert len(before) >= rig.cfg.takeover.warm_cycles == 20
            for e in before:
                np.testing.assert_allclose(e[2], MEASURED[side], atol=1e-12)
            # the foreign-message check and the controller listing come first, after the discovery wait
            topic = rig.cfg.arms[side].command_topic
            assert any(e[:2] == ("foreign", topic) for e in rig.log[:i])
            assert ("list", side) in rig.log[:i]
            switch = rig.log[i]
            assert switch[4] == rig.cfg.takeover.service_timeout_s
            assert switch[5] >= rig.cfg.takeover.discovery_wait_s
        # every controller state is read before anything is published (an active one would be refused) ...
        first_pub = rig.index(lambda e: e[0] == "pub")
        for side in SIDES:
            assert rig.index(lambda e, side=side: e == ("list", side)) < first_pub
        # ... and the left switch happens before the right takeover touches its controller again
        assert rig.index(is_switch("left", "activate")) < rig.index(lambda e: e == ("list", "right"), first_pub)
        # the command stream stays continuous (equal to measured) across activation, in both command modes
        for side in SIDES:
            for e in rig.pubs(side):
                np.testing.assert_allclose(e[2], MEASURED[side], atol=1e-12)
        assert rig.streamer.stops == 0
        n_before_exit = len(rig.log)

    # release: hold, deactivate what we activated, then stop the stream exactly once, nothing published after
    for side in SIDES:
        assert len(rig.switches(side, kind="deactivate")) == 1
        assert rig.index(is_switch(side, "deactivate"), n_before_exit) > n_before_exit
    stop = rig.log.index(("stop",))
    assert stop > max(rig.index(is_switch(side, "deactivate")) for side in SIDES)
    assert rig.streamer.stops == 1
    assert not any(e[0] == "pub" for e in rig.log[stop:])
    # adapters go back to identity only after publishing stopped
    assert all(rig.log.index(("adapter_deactivate", side)) > stop for side in SIDES)


def test_stream_outlives_the_last_deactivation():
    """The hold keeps streaming for RELEASE_TAIL_S after the last deactivation, so even the arm released last sees
    commands after its controller went inactive (the stream never ends while a controller may still be active)."""
    rig = Rig()
    with rig.session():
        pass
    last_deactivate = max(rig.index(is_switch(side, "deactivate")) for side in SIDES)
    stop = rig.log.index(("stop",))
    for side in SIDES:
        tail = rig.pubs(side, rig.log[last_deactivate:stop])
        assert len(tail) >= round(RELEASE_TAIL_S * 100) - 1, f"{side}: {len(tail)} commands after the deactivation"
    assert RELEASE_TAIL_S in rig.time.sleeps

    # nothing was ever active (takeover refused on the left after streaming started): no deactivation, no tail
    rig = Rig()
    rig.cms["left"].list_hook = lambda: rig.cms["left"].lists == 2 and rig.graph.ages.__setitem__("left", 1.0)
    with pytest.raises(TakeoverError), rig.session():
        pytest.fail("not reached")
    assert rig.pubs("left")
    assert not rig.switches(kind="deactivate")
    assert RELEASE_TAIL_S not in rig.time.sleeps


def test_tolerance_rechecked_immediately_before_activation():
    rig = Rig()
    rig.start()

    def drift():  # the arm moves while the controller state is being read
        rig.graph.measured_q["left"] = MEASURED["left"] + 0.05

    rig.cms["left"].list_hook = drift
    with pytest.raises(TakeoverError, match="tol_rad"):
        rig.takeover("left")
    assert not any(e[0] in ("switch", "activate") for e in rig.log)


def test_unhealthy_stream_refuses_activation():
    rig = Rig()
    rig.start()
    rig.time.sleep(0.3)
    rig.streamer.fault = "RuntimeError: boom"
    with pytest.raises(TakeoverError, match="boom"):
        rig.takeover("left")
    assert not any(e[0] in ("switch", "activate") for e in rig.log)


def test_session_refuses_missing_measured():
    rig = Rig()
    rig.graph.measured_q["right"] = None
    with pytest.raises(TakeoverError, match="right"), rig.session():
        pytest.fail("not reached")
    assert not rig.streamer.running
    assert not any(e[0] in ("pub", "switch", "activate") for e in rig.log)


# ---- pre-checks ----


def test_session_refuses_stale_measured_before_streaming():
    """The measured stream died (e.g. while the operator sat at the reset prompt): the last sample is old, so
    neither the hold nor any tolerance check may be built from it."""
    rig = Rig()
    rig.graph.ages["left"] = 2.0
    with pytest.raises(TakeoverError) as excinfo, rig.session():
        pytest.fail("not reached")
    msg = str(excinfo.value)
    assert "left" in msg
    assert rig.cfg.arms["left"].measured_topic in msg
    assert "takeover.max_measured_age_s" in msg
    assert not any(e[0] in ("pub", "switch", "activate") for e in rig.log)


def test_takeover_refuses_measured_going_stale_before_activation():
    rig = Rig()
    rig.start()

    def die():  # the joint state broadcaster stops while the controller state is read
        rig.graph.ages["left"] = 1.0

    rig.cms["left"].list_hook = die
    with pytest.raises(TakeoverError, match="max_measured_age_s"):
        rig.takeover("left")
    assert not any(e[0] in ("switch", "activate") for e in rig.log)
    assert rig.cfg.takeover.max_measured_age_s == 0.3


@pytest.mark.parametrize("mode", ["switch", "external"])
def test_session_refuses_an_already_active_controller_before_streaming(mode):
    """An active controller (left on by bring-up, after external mode, after a failed release) would act on our
    very first message: refuse before publishing anything, and say how to get out of it."""
    rig = Rig([f"takeover.mode={mode}"])
    rig.cms["right"].states[CTRL] = "active"
    with pytest.raises(TakeoverError) as excinfo, rig.session():
        pytest.fail("not reached")
    msg = str(excinfo.value)
    assert "already active" in msg
    assert "/right/controller_manager" in msg
    assert "release --arm right" in msg
    assert "takeover.accept_active" in msg
    assert not any(e[0] in ("pub", "switch", "activate") for e in rig.log)
    assert ("list", "right") in rig.log
    assert rig.cms["right"].states[CTRL] == "active"


def test_session_refuses_an_unusable_controller_state_before_streaming():
    rig = Rig()
    rig.cms["left"].states[CTRL] = "unconfigured"
    with pytest.raises(TakeoverError, match="unconfigured"), rig.session():
        pytest.fail("not reached")
    assert not any(e[0] in ("pub", "switch", "activate") for e in rig.log)


def test_refuses_foreign_messages_warns_on_idle_endpoint(caplog):
    rig = Rig()
    topic = rig.cfg.arms["left"].command_topic
    rig.graph.foreign[topic] = 3
    rig.start()
    with pytest.raises(TakeoverError) as excinfo:
        rig.takeover("left")
    assert topic in str(excinfo.value)
    assert "is GELLO teleop still running?" in str(excinfo.value)
    assert ("foreign", topic, 1.0) in rig.log
    assert not any(e[0] in ("switch", "activate") for e in rig.log)

    # an idle foreign publisher endpoint (no messages) only warns
    rig = Rig()
    topic = rig.cfg.arms["right"].command_topic
    rig.graph.publishers[topic] = 2  # ours + one idle
    rig.start()
    with caplog.at_level(logging.WARNING):
        assert rig.takeover("right") is True
    assert any(topic in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_session_checks_foreign_messages_before_publishing_anything():
    rig = Rig(uptime0=0.0)
    with rig.session():
        pass
    first_pub = rig.index(lambda e: e[0] == "pub")
    for side in SIDES:
        topic = rig.cfg.arms[side].command_topic
        assert rig.index(lambda e, topic=topic: e[:2] == ("foreign", topic)) < first_pub
    assert rig.log[first_pub][3] >= rig.cfg.takeover.discovery_wait_s  # after the discovery wait

    # GELLO still publishing on the right arm: refuse before streaming on any arm
    rig = Rig()
    topic = rig.cfg.arms["right"].command_topic
    rig.graph.foreign[topic] = 5
    with pytest.raises(TakeoverError) as excinfo, rig.session():
        pytest.fail("not reached")
    assert topic in str(excinfo.value)
    assert "is GELLO teleop still running?" in str(excinfo.value)
    assert not rig.streamer.running
    assert not any(e[0] in ("pub", "switch", "activate") for e in rig.log)


# ---- controller states ----


def test_accept_active_streams_into_it_and_waits_for_the_operator_to_deactivate(caplog, capsys):
    """takeover.accept_active (explicit opt-in): stream into an already-active controller, never switch it; on
    release ask the operator to deactivate it and keep streaming the hold until it reads inactive."""
    rig = Rig(["takeover.accept_active=true"])
    left = rig.cms["left"]
    left.states[CTRL] = "active"
    with caplog.at_level(logging.WARNING), rig.session() as session:
        assert session.activated_by_us == {"left": False, "right": True}
        left.operator_deactivates_on_list = left.lists + 6  # the operator needs a few polls
        rig.log.append(("exit",))
    assert rig.switches("left", kind="activate") == []
    assert rig.switches("left", kind="deactivate") == []
    assert any(e[:2] == ("activate", "left") for e in rig.log)  # the adapter reference is still captured
    assert any("already active" in r.getMessage() for r in caplog.records)
    assert len(rig.switches("right", kind="deactivate")) == 1
    assert left.states[CTRL] == "inactive"
    assert f"[left] deactivate {CTRL} on /left/controller_manager now" in capsys.readouterr().out
    assert session.still_active == []
    # the hold kept streaming until the operator's deactivation was seen, then for the tail
    seen = rig.log.index(("operator_deactivate", "left"))
    stop = rig.log.index(("stop",))
    assert len(rig.pubs("left", rig.log[seen:stop])) >= round(RELEASE_TAIL_S * 100) - 1
    assert len(rig.pubs("left", rig.log[lists_at_exit(rig) : seen])) >= 50  # held while waiting for the operator


def test_unconfigured_controller_points_to_external_mode():
    rig = Rig()
    rig.cms["left"].states[CTRL] = "unconfigured"
    with pytest.raises(TakeoverError) as excinfo, rig.session():
        pytest.fail("not reached")
    msg = str(excinfo.value)
    assert "unconfigured" in msg
    assert CTRL in msg
    assert "takeover.mode: external" in msg
    assert not any(e[0] == "switch" for e in rig.log)
    assert rig.streamer.stops == 1


def test_refuses_missing_controller_lists_available():
    rig = Rig()
    rig.cms["right"].states = {"gravity_compensation_controller": "inactive", "joint_state_broadcaster": "active"}
    rig.start()
    with pytest.raises(TakeoverError) as excinfo:
        rig.takeover("right")
    msg = str(excinfo.value)
    assert CTRL in msg
    assert "/right/controller_manager" in msg
    assert "gravity_compensation_controller" in msg
    assert "joint_state_broadcaster" in msg
    assert "takeover.mode: external" in msg
    assert not any(e[0] == "switch" for e in rig.log)


def test_external_mode_never_switches(capsys):
    rig = Rig(["takeover.mode=external"])
    for side in SIDES:
        rig.cms[side].operator_activates_on_list = 3  # the operator activates while we poll
    with rig.session() as session:
        assert session.activated_by_us == {"left": False, "right": False}
        for side in SIDES:
            assert any(e[:2] == ("activate", side) for e in rig.log)
            rig.cms[side].operator_deactivates_on_list = rig.cms[side].lists + 5
    assert not any(e[0] == "switch" for e in rig.log)
    out = capsys.readouterr().out
    assert f"activate {CTRL} on /left/controller_manager now" in out
    assert f"activate {CTRL} on /right/controller_manager now" in out
    # release: the operator is asked to deactivate; the hold streams until both read inactive, then the tail
    assert f"[left] deactivate {CTRL} on /left/controller_manager now" in out
    assert f"[right] deactivate {CTRL} on /right/controller_manager now" in out
    stop = rig.log.index(("stop",))
    for side in SIDES:
        seen = rig.log.index(("operator_deactivate", side))
        assert seen < stop
        assert len(rig.pubs(side, rig.log[seen:stop])) >= round(RELEASE_TAIL_S * 100) - 1
    assert session.still_active == []
    assert rig.streamer.stops == 1

    # never activated: gives up after 60 s, still without switching
    rig = Rig(["takeover.mode=external"])
    rig.start()
    with pytest.raises(TakeoverError, match="not active"):
        rig.takeover("left")
    assert rig.time.t >= 60.0
    assert not any(e[0] in ("switch", "activate") for e in rig.log)


def test_service_timeout_raises_takeover_error():
    rig = Rig(["takeover.service_timeout_s=2.5"])
    rig.cms["left"].list_error = TimeoutError("list_controllers: no response within 2.5 s")
    rig.start()
    with pytest.raises(TakeoverError, match="/left/controller_manager"):
        rig.takeover("left")
    assert not any(e[0] in ("switch", "activate") for e in rig.log)

    # switch times out: it may have taken effect, so takeover deactivates before raising
    rig = Rig(["takeover.service_timeout_s=2.5"])
    rig.cms["left"].switch_error = TimeoutError("switch_controller: no response within 2.5 s")
    rig.start()
    with pytest.raises(TakeoverError, match="switch_controller"):
        rig.takeover("left")
    calls = [e for e in rig.log if e[0] == "switch"]
    assert [(e[2], e[3], e[4]) for e in calls] == [((CTRL,), (), 2.5), ((), (CTRL,), 2.5)]


def test_activation_not_confirmed_raises_and_releases():
    rig = Rig()
    rig.cms["left"].apply = False  # switch "succeeds" but the controller never becomes active
    with pytest.raises(TakeoverError, match="not confirmed active"), rig.session():
        pytest.fail("not reached")
    act = rig.index(is_switch("left", "activate"))
    deact = rig.index(is_switch("left", "deactivate"))
    assert act < deact < rig.log.index(("stop",))
    assert rig.log[deact][5] - rig.log[act][5] >= 1.0  # waited confirm_s
    assert not any(e[0] == "switch" and e[1] == "right" for e in rig.log)
    assert rig.streamer.stops == 1


def test_arm_jump_after_activation_deactivates():
    rig = Rig()
    cm = rig.cms["left"]

    def jump():  # the controller pulls the arm away once active
        if cm.states[CTRL] == "active":
            rig.graph.measured_q["left"] = MEASURED["left"] + 0.05

    cm.list_hook = jump
    rig.start()
    with pytest.raises(TakeoverError, match="after activation"):
        rig.takeover("left")
    assert rig.index(is_switch("left", "activate")) < rig.index(is_switch("left", "deactivate"))


# ---- interrupts and release ----


@pytest.mark.parametrize("where", ["left_confirm_sleep", "right_prechecks"])
def test_interrupt_during_takeover_releases_only_what_we_activated(where):
    rig = Rig()
    if where == "left_confirm_sleep":
        # Ctrl-C from the fake sleep right after left's switch (while confirming): takeover deactivates left
        left_switched = lambda dt: bool(rig.switches("left", kind="activate"))  # noqa: E731
        rig.time.hooks.append(raise_once(KeyboardInterrupt("ctrl-c"), left_switched))
    else:
        # left fully taken over; Ctrl-C while right's pre-checks run, before right is switched: Session releases left
        right_topic = rig.cfg.arms["right"].command_topic

        def interrupt(topic):  # the 1st check of the right topic is Session's, before streaming; the 2nd is takeover's
            if topic == right_topic and sum(e[:2] == ("foreign", right_topic) for e in rig.log) == 2:
                raise KeyboardInterrupt("ctrl-c")

        rig.graph.foreign_hook = interrupt

    with pytest.raises(KeyboardInterrupt) as excinfo, rig.session():
        pytest.fail("not reached")
    assert excinfo.value.args == ("ctrl-c",)  # the original interrupt, not masked
    assert len(rig.switches("left", kind="activate")) == 1
    assert len(rig.switches("left", kind="deactivate")) == 1
    assert not any(e[0] == "switch" and e[1] == "right" for e in rig.log)
    stop = rig.log.index(("stop",))
    assert stop > rig.index(is_switch("left", "deactivate"))
    assert rig.streamer.stops == 1
    assert not any(e[0] == "pub" for e in rig.log[stop:])


def test_second_interrupt_during_release_does_not_abort_it():
    rig = Rig()
    rig.time.hooks.append(raise_once(KeyboardInterrupt("second"), lambda dt: dt == 0.5))  # the release hold
    with pytest.raises(ValueError, match="policy boom"), rig.session():
        raise ValueError("policy boom")
    for side in SIDES:
        assert len(rig.switches(side, kind="deactivate")) == 1
    assert rig.streamer.stops == 1

    # without an original exception, the swallowed interrupt is re-raised once release is complete
    rig = Rig()
    rig.time.hooks.append(raise_once(KeyboardInterrupt("second"), lambda dt: dt == 0.5))
    with pytest.raises(KeyboardInterrupt), rig.session():
        pass
    for side in SIDES:
        assert len(rig.switches(side, kind="deactivate")) == 1
    assert rig.log[-1] == ("adapter_deactivate", "right")
    assert rig.streamer.stops == 1


def test_release_attempts_all_sides_on_error(capsys):
    rig = Rig()
    rig.cms["left"].deactivate_error = RuntimeError("left service down")
    with pytest.raises(ValueError, match="policy boom"), rig.session():
        raise ValueError("policy boom")  # the original exception is not masked by the release error
    assert len(rig.switches("left", kind="deactivate")) == 2  # failed once, retried while holding
    assert len(rig.switches("right", kind="deactivate")) == 1
    assert rig.log.index(("stop",)) > rig.index(is_switch("right", "deactivate"))
    assert rig.streamer.stops == 1
    err = capsys.readouterr().err
    assert "left" in err
    assert "left service down" in err
    kinds = [kind for kind, _ in rig.recorder.events]
    assert "release_errors" in kinds
    fields = dict(rig.recorder.events)["release_errors"]
    assert any("left service down" in e for e in fields["errors"])

    # release() itself collects errors instead of raising and still deactivates
    rig = Rig()
    rig.cms["left"].switch_error = RuntimeError("cm down")
    errors = release(
        "left", rig.cfg.arms["left"], rig.cms["left"], rig.streamer, activated_by_us=True, sleep=rig.time.sleep
    )
    assert len(errors) == 2  # hold failed (stream not started) + deactivation failed
    assert any("cm down" in e for e in errors)
    assert len(rig.switches("left", kind="deactivate")) == 1

    # not activated by us: hold only, never deactivate
    rig = Rig()
    rig.start()
    assert (
        release(
            "left", rig.cfg.arms["left"], rig.cms["left"], rig.streamer, activated_by_us=False, sleep=rig.time.sleep
        )
        == []
    )
    assert not any(e[0] == "switch" for e in rig.log)
    assert rig.time.sleeps == [0.5]


# ---- inactive arm ----


def test_inactive_arm_held_when_configured():
    rig = Rig(["active_arms=[right]"])
    assert rig.sides == ("left", "right")
    with rig.session() as session:
        assert session.activated_by_us == {"left": True, "right": True}
        # the policy moves the right arm; the inactive left arm only ever streams its measured pose
        rig.streamer.set_waypoint("right", MEASURED["right"] + 0.01)
        rig.graph.measured_q["right"] = MEASURED["right"] + 0.01
        rig.time.sleep(0.5)
    assert len(rig.pubs("left")) > 50
    for e in rig.pubs("left"):
        np.testing.assert_allclose(e[2], MEASURED["left"], atol=1e-12)
    np.testing.assert_allclose(rig.pubs("right")[-1][2], MEASURED["right"] + 0.01, atol=1e-12)
    assert len(rig.switches("left", kind="deactivate")) == 1

    rig = Rig(["active_arms=[right]", "takeover.hold_inactive_arms=false"])
    assert rig.sides == ("right",)
    with rig.session() as session:
        assert session.activated_by_us == {"right": True}
    assert rig.cms["left"].lists == 0
    assert not any(e[0] == "pub" and e[1] == "left" for e in rig.log)


# ---- release: a controller that does not go inactive is never left without the stream ----


def test_release_holds_every_side_then_waits_once():
    rig = Rig()
    with rig.session():
        n = len(rig.log)
    hold = rig.index(lambda e: e == ("hold", None), n)
    first_deactivate = min(rig.index(is_switch(side, "deactivate"), n) for side in SIDES)
    assert hold < first_deactivate
    assert rig.time.sleeps.count(0.5) == 1


def test_refused_deactivation_is_retried_while_holding():
    rig = Rig()
    rig.cms["left"].refuse_deactivate = 2
    with rig.session() as session:
        pass
    assert len(rig.switches("left", kind="deactivate")) == 3
    assert rig.cms["left"].states[CTRL] == "inactive"
    assert session.still_active == []
    stop = rig.log.index(("stop",))
    last = max(i for i, e in enumerate(rig.log) if is_switch("left", "deactivate")(e))
    assert len(rig.pubs("left", rig.log[last:stop])) >= round(RELEASE_TAIL_S * 100) - 1


@pytest.mark.parametrize("original", [None, ValueError("policy boom")])
def test_failed_deactivation_keeps_holding_and_reports_loudly(original, capsys):
    """switch_controller keeps refusing to deactivate: retry for RELEASE_RETRY_S while streaming the hold, then a loud
    message, and keep holding until the operator confirms with Enter; never report success."""
    rig = Rig()
    rig.cms["left"].refuse_deactivate = 10**6
    rig.ack_after = 30  # Enter after ~30 polls of the loud prompt

    def body():
        with rig.session():
            if original is not None:
                raise original

    with pytest.raises(ControllerStillActive) as excinfo:
        body()
    if original is not None:
        assert excinfo.value.__cause__ is original  # the original exception is kept
    assert "/left/controller_manager" in str(excinfo.value)
    err = capsys.readouterr().err
    assert f"CONTROLLER STILL ACTIVE on /left/controller_manager ({CTRL})" in err
    assert "E-stop" in err
    assert len(rig.switches("left", kind="deactivate")) > 3  # retried
    assert len(rig.switches("right", kind="deactivate")) == 1
    assert rig.cms["left"].states[CTRL] == "active"
    stop = rig.log.index(("stop",))
    first = rig.index(is_switch("left", "deactivate"))
    held = rig.pubs("left", rig.log[first:stop])
    assert held[-1][3] - held[0][3] >= RELEASE_RETRY_S  # streamed the whole time
    for e in held:
        np.testing.assert_allclose(e[2], MEASURED["left"], atol=1e-12)  # a hold, never a move
    assert rig.acks >= 30
    kinds = dict(rig.recorder.events)
    assert kinds["released"]["still_active"] == ["left"]


def test_failed_deactivation_ends_when_the_operator_deactivates_it():
    rig = Rig()
    left = rig.cms["left"]
    left.refuse_deactivate = 10**6

    def operator(dt):  # the operator uses their own tooling once the loud message is out
        if rig.acks == 5:
            left.operator_deactivates_on_list = left.lists + 1

    rig.time.hooks.append(operator)
    with rig.session() as session:
        pass
    assert session.still_active == []
    assert left.states[CTRL] == "inactive"
    assert rig.log.index(("operator_deactivate", "left")) < rig.log.index(("stop",))


def test_external_release_never_deactivated_reports_loudly(capsys):
    rig = Rig(["takeover.mode=external"])
    for side in SIDES:
        rig.cms[side].operator_activates_on_list = 3
    rig.ack_after = 3
    with pytest.raises(ControllerStillActive), rig.session():
        pass
    assert rig.time.t >= EXTERNAL_WAIT_S
    assert "CONTROLLER STILL ACTIVE on /left/controller_manager" in capsys.readouterr().err
    assert not any(e[0] == "switch" for e in rig.log)


def test_retry_after_an_interrupted_deactivation_reads_the_state_first():
    rig = Rig()
    left = rig.cms["left"]

    def interrupted_reply(activate, deactivate):  # Ctrl-C after the controller_manager applied the switch
        if deactivate:
            left.after_switch_hook = None
            raise KeyboardInterrupt("ctrl-c")

    left.after_switch_hook = interrupted_reply
    with pytest.raises(KeyboardInterrupt), rig.session() as session:
        pass
    assert len(rig.switches("left", kind="deactivate")) == 1  # inactive already: no second switch
    assert session.still_active == []
    errors = dict(rig.recorder.events).get("release_errors", {}).get("errors", [])
    assert not any("refused" in e or "still active" in e for e in errors)


def test_interrupt_right_after_takeover_returned_still_deactivates(monkeypatch):
    """Ctrl-C between takeover() returning True and Session recording it: the switch is marked pending before it is
    called, so the release still deactivates the controller."""
    rig = Rig()
    real = takeover_module.takeover

    def takeover_then_ctrl_c(side, *args, **kwargs):
        result = real(side, *args, **kwargs)
        if side == "left":
            raise KeyboardInterrupt("ctrl-c")
        return result

    monkeypatch.setattr(takeover_module, "takeover", takeover_then_ctrl_c)
    with pytest.raises(KeyboardInterrupt), rig.session():
        pytest.fail("not reached")
    assert len(rig.switches("left", kind="deactivate")) == 1
    assert rig.cms["left"].states[CTRL] == "inactive"
    assert rig.log.index(("stop",)) > rig.index(is_switch("left", "deactivate"))


def test_interrupt_on_entering_the_release_does_not_skip_it():
    """A Ctrl-C landing in __exit__ before the release starts (e.g. before the CLI's signal deferral is entered)
    must not skip the release."""

    class Flaky(Session):
        interrupts = 1

        def _close(self):
            if self.interrupts:
                self.interrupts -= 1
                raise KeyboardInterrupt("ctrl-c on entry")
            return super()._close()

    rig = Rig()
    session = Flaky(
        rig.cfg, rig.sides, rig.cms, rig.graph, rig.streamer, rig.adapters, sleep=rig.time.sleep, clock=lambda: 0.0
    )
    with pytest.raises(KeyboardInterrupt), session:
        pass
    for side in SIDES:
        assert len(rig.switches(side, kind="deactivate")) == 1
    assert rig.streamer.stops == 1


def test_frame_id_is_unique_per_process():
    """Every client process has its own frame_id, so a second client instance counts as a foreign publisher."""
    import re
    import subprocess
    import sys

    assert re.fullmatch(r"robot_client:[0-9a-f]{8}", takeover_module.OUR_FRAME_ID)
    code = "from robot_client.takeover import OUR_FRAME_ID; print(OUR_FRAME_ID)"
    pkg_parent = str(Path(takeover_module.__file__).resolve().parents[1])
    ids = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=pkg_parent
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(ids) == 2
    assert takeover_module.OUR_FRAME_ID not in ids


def test_release_survives_a_closed_terminal(monkeypatch):
    """After a SIGHUP the terminal is gone and writes fail: the release must still hold, wait and stop the stream."""
    import builtins

    rig = Rig(["takeover.mode=external"])
    for side in SIDES:
        rig.cms[side].operator_activates_on_list = 3
    with rig.session():
        real_print = builtins.print

        def broken_print(*args, **kwargs):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(builtins, "print", broken_print)
        for side in SIDES:
            rig.cms[side].operator_deactivates_on_list = rig.cms[side].lists + 3
    monkeypatch.setattr(builtins, "print", real_print)
    assert rig.streamer.stops == 1
    for side in SIDES:
        assert rig.log.index(("operator_deactivate", side)) < rig.log.index(("stop",))


def test_unreadable_state_at_release_is_retried_and_gets_the_tail():
    """A side we did not activate whose state cannot be read at release is not assumed inactive: it is re-read while
    the hold streams, and once it reads inactive the stream still outlives it by RELEASE_TAIL_S."""
    rig = Rig(["takeover.mode=external"])
    right = rig.cms["right"]
    for side in SIDES:
        rig.cms[side].operator_activates_on_list = 3
    with rig.session():
        for side in SIDES:
            rig.cms[side].operator_deactivates_on_list = rig.cms[side].lists + 1
        right.list_error = TimeoutError("list_controllers: no response")
        rig.time.hooks.append(lambda dt: dt == 0.25 and setattr(right, "list_error", None))
        n = len(rig.time.sleeps)
    assert right.states[CTRL] == "inactive"
    assert rig.time.sleeps[-1] == RELEASE_TAIL_S
    assert RELEASE_TAIL_S in rig.time.sleeps[n:]


def test_enter_at_the_loud_hold_drops_only_the_announced_side(capsys):
    """Enter answers the loud CONTROLLER STILL ACTIVE message for the sides it was shown for. A side still waiting
    for the operator (never announced) keeps its hold until it reads inactive."""
    rig = Rig(["takeover.accept_active=true"])
    left, right = rig.cms["left"], rig.cms["right"]
    right.states[CTRL] = "active"  # already active: the operator deactivates it (up to EXTERNAL_WAIT_S)
    left.refuse_deactivate = 10**6  # ours, never deactivates: loud after RELEASE_RETRY_S
    rig.ack_after = 3

    def operator(dt):  # deactivates the right arm well after the Enter for the left one
        if rig.acks >= 3 and right.operator_deactivates_on_list is None and right.states[CTRL] == "active":
            right.operator_deactivates_on_list = right.lists + 20

    rig.time.hooks.append(operator)
    with pytest.raises(ControllerStillActive) as excinfo, rig.session():
        pass
    err = capsys.readouterr().err
    assert "CONTROLLER STILL ACTIVE on /left/controller_manager" in err
    assert "CONTROLLER STILL ACTIVE on /right/controller_manager" not in err
    assert "/right/" not in str(excinfo.value)
    assert dict(rig.recorder.events)["released"]["still_active"] == ["left"]
    assert right.states[CTRL] == "inactive"
    seen = rig.log.index(("operator_deactivate", "right"))
    stop = rig.log.index(("stop",))
    assert len(rig.pubs("right", rig.log[seen:stop])) >= round(RELEASE_TAIL_S * 100) - 1  # held until it read inactive
    assert rig.acks == 3  # no further Enter was asked for


@pytest.mark.parametrize("healthy", [True, False])
def test_release_messages_say_whether_the_stream_still_holds(healthy, capsys):
    """The loud and the deactivate-now messages claim the client is streaming the hold only while the stream is
    healthy."""
    rig = Rig(["takeover.accept_active=true"])
    rig.cms["right"].states[CTRL] = "active"
    rig.cms["left"].refuse_deactivate = 10**6
    rig.ack_after = 1  # Enter answers the left side's loud message ...
    rig.cms["right"].operator_deactivates_on_list = 40  # ... the operator deactivates the right one meanwhile

    def body():
        with rig.session():
            if not healthy:
                rig.streamer.fault = "publish failed for left: OSError: boom"

    with pytest.raises(ControllerStillActive):
        body()
    out, err = capsys.readouterr()
    keeps = "keeps streaming the hold until it reads inactive"
    if healthy:
        assert keeps in out
        assert keeps in err
        assert "unhealthy" not in out + err
    else:
        assert keeps not in out + err
        assert "the command stream is unhealthy (publish failed for left: OSError: boom)" in out
        assert "may not be holding the arm" in out
        assert "The command stream is unhealthy (publish failed for left: OSError: boom)" in err
        assert "may NOT be holding the arm" in err
