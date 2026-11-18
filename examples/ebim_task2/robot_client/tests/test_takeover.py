"""Takeover / release / Session ordering against fakes: a real `Streamer` driven by a fake sleep (no thread,
no real waiting), a FakeCM per arm recording every call into one shared event log, and a FakeGraph."""

import logging
from pathlib import Path

import numpy as np
import pytest
from robot_client.command import CommandAdapter
from robot_client.config import load_site_config
from robot_client.smoother import RateLimiter
from robot_client.streamer import Streamer
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
        self.publishers = {}  # topic -> endpoints incl. ours (default 1)
        self.foreign = {}  # topic -> foreign messages in the window
        self.foreign_hook = None

    def uptime_s(self):
        return self.rig.time.t + self.uptime0

    def measured(self, side):
        q = self.measured_q.get(side)
        return None if q is None else q.copy()

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
        self.deactivate_error = None  # raised for deactivation switches only
        self.list_hook = None
        self.operator_activates_on_list = None  # external mode: the operator activates on the n-th list call
        self.lists = 0

    def list_controllers(self):
        self.lists += 1
        self.rig.log.append(("list", self.side))
        if self.list_hook is not None:
            self.list_hook()
        if self.list_error is not None:
            raise self.list_error
        if self.operator_activates_on_list is not None and self.lists >= self.operator_activates_on_list:
            self.states[CTRL] = "active"
        return dict(self.states)

    def switch(self, activate, deactivate, timeout_s):
        self.rig.log.append(("switch", self.side, tuple(activate), tuple(deactivate), timeout_s, self.rig.time.t))
        if self.switch_error is not None:
            raise self.switch_error
        if deactivate and self.deactivate_error is not None:
            raise self.deactivate_error
        if self.apply:
            for c in activate:
                self.states[c] = "active"
            for c in deactivate:
                self.states[c] = "inactive"
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
        # the left switch happens before anything of the right takeover touches its controller
        assert rig.index(is_switch("left", "activate")) < rig.index(lambda e: e == ("list", "right"))
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


def test_already_active_not_switched_not_deactivated(caplog):
    rig = Rig()
    rig.cms["left"].states[CTRL] = "active"
    with caplog.at_level(logging.WARNING), rig.session() as session:
        assert session.activated_by_us == {"left": False, "right": True}
    assert rig.switches("left", kind="activate") == []
    assert rig.switches("left", kind="deactivate") == []
    assert any(e[:2] == ("activate", "left") for e in rig.log)  # the adapter reference is still captured
    assert any("already active" in r.getMessage() for r in caplog.records)
    assert len(rig.switches("right", kind="deactivate")) == 1
    assert rig.cms["left"].states[CTRL] == "active"


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
    assert not any(e[0] == "switch" for e in rig.log)
    out = capsys.readouterr().out
    assert f"activate {CTRL} on /left/controller_manager now" in out
    assert f"activate {CTRL} on /right/controller_manager now" in out
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
    assert len(rig.switches("left", kind="deactivate")) == 1
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
