"""The CLI with monkeypatched ROS factories (fake node, controller managers, motion backend, console), the real
Streamer / Session / Runner and the stub policy server. No ROS."""

import io
import json
import os
from pathlib import Path
import signal
import threading
import time
from typing import ClassVar

import numpy as np
import pytest
from robot_client import checks
from robot_client import cli
from robot_client.config import load_site_config
from robot_client.keys import ENTER
from robot_client.prepare import PrepareReport
from robot_client.prepare import StepResult
from robot_client.preprocess import Snapshot
from robot_client.runner import EpisodeOutcome
from robot_client.runner import required_stream_keys

MUNICH_YAML = Path(__file__).resolve().parents[1] / "robot_site.munich.yaml"
CTRL = "joint_impedance_controller"


class ScriptedConsole:
    """`lines` answer read_line() in order (None = EOF once exhausted); `keys` answer poll() in order."""

    def __init__(self, lines=(), keys=()):
        self.lines = list(lines)
        self.keys = list(keys)
        self.prompts = []
        self.polls = 0
        self.discards = 0
        self.closed = False

    def read_line(self, prompt):
        self.prompts.append(prompt)
        return self.lines.pop(0) if self.lines else None

    def poll(self):
        self.polls += 1
        assert self.polls < 2000, "the CLI kept polling for a key that the script never sends"
        return self.keys.pop(0) if self.keys else None

    def discard_pending(self):
        self.discards += 1

    def close(self):
        self.closed = True


class FakeNode:
    def __init__(self, cfg, command_mode, world):
        self.cfg = cfg
        self.command_mode = command_mode
        self.world = world
        self.q = {side: np.array(arm.start_pose, dtype=float) for side, arm in cfg.arms.items()}
        self.follow = False  # measured jumps to every published command (a perfect controller)
        self.arm_msgs = dict.fromkeys(cfg.arms, 0)
        self.gripper_msgs = []
        self.started = self.shut = False

    def start(self):
        self.started = True

    def shutdown(self):
        self.shut = True

    def measured(self, side):
        return self.q[side].copy()

    def measured_age(self, side):
        return 0.005

    def publish_arm(self, side, q):
        self.arm_msgs[side] += 1
        if self.follow:
            self.q[side] = np.array(q, dtype=float)

    def publish_gripper(self, side, width):
        self.gripper_msgs.append((side, width))

    def snapshot(self):
        images = {name: np.full((48, 64, 3), 100, dtype=np.uint8) for name in self.cfg.cameras}
        ages = {f"cam/{name}": 0.01 for name in self.cfg.cameras}
        ages |= {f"arm/{side}": 0.01 for side in self.cfg.arms}
        ages |= {"grip/right": 0.01, "grip/left": float("inf")}
        return Snapshot(
            images=images,
            arm_q={side: q.copy() for side, q in self.q.items()},
            knuckle={"right": 0.1},
            ages=ages,
            t_mono=time.monotonic(),
        )

    def uptime_s(self):
        return 100.0

    def count_publishers(self, topic):
        return 1

    def foreign_messages(self, topic, window_s):
        return self.world.foreign.get(topic, 0)

    def stream_stats(self):
        stats = {}
        for name, cam in self.cfg.cameras.items():
            stats[f"cam/{name}"] = _stream(
                cam.topic, 30.0, {"height": 48, "width": 64, "encoding": "rgb8", "step": 192}
            )
        for side, arm in self.cfg.arms.items():
            names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]
            stats[f"arm/{side}"] = _stream(arm.measured_topic, 200.0, {"names": names})
            if side == "right":
                stats[f"grip/{side}"] = _stream(arm.gripper.state_topic, 50.0, {"names": [arm.gripper.joint]})
            else:
                stats[f"grip/{side}"] = {**_stream(arm.gripper.state_topic, None, {}), "count": 0, "age_s": None}
        stats["commands"] = {}
        return stats


def _stream(topic, rate, raw):
    return {
        "topic": topic,
        "count": 100,
        "rate_hz": rate,
        "max_gap_s": 0.02,
        "age_s": 0.01,
        "raw": raw,
        "decode_errors": 0,
        "last_error": None,
    }


class FakeCM:
    def __init__(self, side, world):
        self.side = side
        self.world = world
        self.state = "inactive"
        self.on_deactivate = None
        self.refuse_deactivate = False  # switch_controller refuses every deactivation

    def list_controllers(self):
        return {CTRL: self.state, "joint_state_broadcaster": "active"}

    def switch(self, activate, deactivate, timeout_s):
        if deactivate and self.on_deactivate is not None:
            self.on_deactivate()
        self.world.log.append(("switch", self.side, tuple(activate), tuple(deactivate)))
        if deactivate and self.refuse_deactivate:
            return False
        for _ in activate:
            self.state = "active"
        for _ in deactivate:
            self.state = "inactive"
        return True


class FakeBackend:
    def spine_position(self, timeout_s):
        return 0.434


class World:
    """Installs the fakes into the cli module; records what was constructed and called."""

    def __init__(self, monkeypatch, console, *, follow=False, cm_hook=None):
        self.follow = follow  # nodes jump to every published command (a perfect controller)
        self.cm_hook = cm_hook  # called with the controller managers right after they are built
        self.log = []
        self.nodes = []
        self.foreign = {}
        self.cms = {}
        self.console = console
        self.prepare_ok = True
        monkeypatch.setattr(cli, "make_node", self._make_node)
        monkeypatch.setattr(cli, "make_controller_managers", self._make_cms)
        monkeypatch.setattr(cli, "make_backend", self._make_backend)
        monkeypatch.setattr(cli, "make_console", lambda: console)
        monkeypatch.setattr(cli, "prepare", self._prepare)

    def _make_node(self, cfg, command_mode):
        node = FakeNode(cfg, command_mode, self)
        node.follow = self.follow
        node.start()
        self.nodes.append(node)
        self.log.append(("node", command_mode))
        return node

    def _make_cms(self, node, cfg):
        self.log.append(("cms",))
        self.cms = {side: FakeCM(side, self) for side in cfg.arms}
        if self.cm_hook is not None:
            self.cm_hook(self.cms)
        return self.cms

    def _make_backend(self, node, cfg):
        self.log.append(("backend",))
        return FakeBackend()

    def _prepare(self, cfg, cm, backend, streamer_active):
        self.log.append(("prepare", streamer_active))
        return PrepareReport([StepResult("ptp left", ok=self.prepare_ok, detail="fake")])

    def kinds(self):
        return [entry[0] for entry in self.log]

    def index(self, entry):
        return self.log.index(entry)


def base_args(tmp_path, *extra_sets):
    sets = ["out_dir=" + str(tmp_path), "smoother.impl=clamp", "safety.episode_timeout_s=0.5", *extra_sets]
    args = ["--site", str(MUNICH_YAML), "--run-name", "t"]
    for item in sets:
        args += ["--set", item]
    return args


def server_env(port):
    return {"POLICY_SERVER_HOST": "127.0.0.1", "POLICY_SERVER_PORT": str(port)}


def episode_dirs(tmp_path):
    return sorted((tmp_path / "t").glob("episode_*"))


# ---- parsing / config ----


def test_set_is_repeatable_and_site_defaults_to_the_env(monkeypatch):
    monkeypatch.setenv("ROBOT_SITE", "/somewhere/site.yaml")
    args = cli.parse_args(["run", "--set", "a.b=1", "--set", "c=[1, 2]", "--episodes", "3", "--arms", "right"])
    assert args.set == ["a.b=1", "c=[1, 2]"]
    assert args.site == "/somewhere/site.yaml"
    assert args.episodes == 3
    assert args.arms == ["right"]
    monkeypatch.delenv("ROBOT_SITE")
    assert cli.parse_args(["check-obs"]).site == "/config/robot_site.yaml"


def test_bad_yaml_exits_4(tmp_path, capsys):
    site = tmp_path / "site.yaml"
    site.write_text("cameras: [unclosed\n")
    assert cli.main(["check-infer", "--site", str(site)], env={}) == 4
    assert "not valid YAML" in capsys.readouterr().err


def test_bad_override_and_bad_run_name_exit_4(tmp_path, capsys):
    assert cli.main(["check-infer", "--site", str(MUNICH_YAML), "--set", "safety.max_step_rad=-1"], env={}) == 4
    assert "safety.max_step_rad" in capsys.readouterr().err
    assert cli.main(["check-infer", "--site", str(MUNICH_YAML), "--run-name", "a/b"], env={}) == 4


def test_usage_error_exits_4():
    assert cli.main(["check-motion"], env={}) == 4  # --arm is required


# ---- check-infer ----


def test_check_infer_hints_at_policy_server_host_and_saves_the_report(
    tmp_path, monkeypatch, capsys, stub_server, free_port
):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    monkeypatch.setattr(cli, "make_node", lambda cfg, mode: (_ for _ in ()).throw(ImportError("no rclpy")))
    # POLICY_SERVER_HOST unset: the hint is printed before connecting (to localhost, where nothing listens)
    args = ["check-infer", *base_args(tmp_path, "server.connect_timeout_s=1"), "--n", "1"]
    code = cli.main(args, env={"POLICY_SERVER_PORT": str(free_port)})
    out = capsys.readouterr()
    assert code == 2
    assert "POLICY_SERVER_HOST" in out.out
    env = server_env(server.port)
    assert cli.main(["check-infer", *base_args(tmp_path), "--n", "1"], env=env) == 0
    saved = json.loads((tmp_path / "t" / "check_infer.json").read_text())
    assert saved["passed"] is True
    assert "POLICY_SERVER_HOST is not set" not in capsys.readouterr().out


def test_check_infer_exits_2_against_a_sim20_server(tmp_path, monkeypatch, capsys, stub_server):
    server = stub_server(layout_id="sim20", latency_s=0.01)
    monkeypatch.setattr(cli, "make_node", lambda cfg, mode: (_ for _ in ()).throw(ImportError("no rclpy")))
    assert cli.main(["check-infer", *base_args(tmp_path), "--n", "1"], env=server_env(server.port)) == 2
    assert "munich16" in capsys.readouterr().out


# ---- check-obs ----


def test_check_obs_uses_a_node_without_publishers(tmp_path, monkeypatch, capsys):
    world = World(monkeypatch, ScriptedConsole())
    assert cli.main(["check-obs", *base_args(tmp_path), "--seconds", "0"], env={}) == 0
    assert world.log[0] == ("node", "none")
    assert "backend" in world.kinds()  # spine position read only
    assert (tmp_path / "t" / "check_obs.json").is_file()
    assert (tmp_path / "t" / "check_obs_head.jpg").is_file()
    assert world.nodes[0].shut


# ---- check-motion ----


def test_check_motion_takes_over_only_the_tested_arm(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(checks, "MOTION_RAMP_S", 0.1)
    monkeypatch.setattr(checks, "MOTION_HOLD_S", 0.1)
    real_session = cli.GuardedSession
    sessions = []

    def spy_session(cfg, sides, cm, *args, **kwargs):
        sessions.append((tuple(sides), tuple(cm)))
        return real_session(cfg, sides, cm, *args, **kwargs)

    monkeypatch.setattr(cli, "GuardedSession", spy_session)
    world = World(monkeypatch, ScriptedConsole(lines=[""]), follow=True)
    code = cli.main(["check-motion", *base_args(tmp_path), "--arm", "right", "--joint", "2"], env={})
    out = capsys.readouterr().out
    assert code == 0, out
    assert sessions == [(("right",), ("right",))]
    switches = [e for e in world.log if e[0] == "switch"]
    assert switches == [("switch", "right", (CTRL,), ()), ("switch", "right", (), (CTRL,))]
    assert "follows" in out
    assert world.nodes[0].command_mode == "live"
    assert (tmp_path / "t" / "check_motion_right.json").is_file()


def test_check_motion_waits_for_enter(tmp_path, monkeypatch):
    world = World(monkeypatch, ScriptedConsole(lines=["q"]))
    assert cli.main(["check-motion", *base_args(tmp_path), "--arm", "left"], env={}) == 0
    assert "switch" not in world.kinds()


# ---- run ----


def test_run_refuses_wrong_server_before_prepare(tmp_path, monkeypatch, capsys, stub_server):
    server = stub_server(layout_id="sim20", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole(lines=["", "s", ""], keys=[ENTER]))
    assert cli.main(["run", *base_args(tmp_path)], env=server_env(server.port)) == 2
    assert "prepare" not in world.kinds()
    assert "node" not in world.kinds()  # refused before anything ROS was even built
    assert "munich16" in capsys.readouterr().err


def test_run_refuses_an_unreachable_server_naming_it(tmp_path, monkeypatch, capsys, free_port):
    world = World(monkeypatch, ScriptedConsole())
    args = ["run", *base_args(tmp_path, "server.connect_timeout_s=1")]
    assert cli.main(args, env=server_env(free_port)) == 2
    assert f"127.0.0.1:{free_port}" in capsys.readouterr().err
    assert world.log == []


def test_run_refuses_a_ros_env_mismatch(tmp_path, monkeypatch, capsys, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole())
    env = server_env(server.port) | {"ROS_DOMAIN_ID": "7"}
    assert cli.main(["run", *base_args(tmp_path)], env=env) == 2
    err = capsys.readouterr().err
    assert "ROS_DOMAIN_ID is 7" in err
    assert "the compose file passes ROS_DOMAIN_ID through" in err
    assert world.log == []


def test_ros_env_refusal_without_a_domain_mismatch_has_no_domain_hint(tmp_path, monkeypatch, capsys, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole())
    env = server_env(server.port) | {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}
    assert cli.main(["run", *base_args(tmp_path)], env=env) == 2
    err = capsys.readouterr().err
    assert "REFUSED: RMW_IMPLEMENTATION is 'rmw_cyclonedds_cpp'" in err
    assert "ROS_DOMAIN_ID" not in err
    assert world.log == []


def test_prepare_waits_for_enter(tmp_path, monkeypatch, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    # neither "x" nor "maybe" is a choice; then stdin ends (EOF counts as quit)
    world = World(monkeypatch, ScriptedConsole(lines=["x", "maybe"]))
    assert cli.main(["run", *base_args(tmp_path)], env=server_env(server.port)) == 0
    assert "prepare" not in world.kinds()
    assert "switch" not in world.kinds()
    assert len(world.console.prompts) == 3


def test_prepare_command_waits_for_enter(tmp_path, monkeypatch):
    world = World(monkeypatch, ScriptedConsole(lines=[]))
    assert cli.main(["prepare", *base_args(tmp_path)], env={}) == 0
    assert "prepare" not in world.kinds()
    world.console.lines = [""]
    assert cli.main(["prepare", *base_args(tmp_path)], env={}) == 0
    assert ("prepare", False) in world.log


def test_failed_prepare_refuses_with_exit_2(tmp_path, monkeypatch, capsys, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole(lines=[""]))
    world.prepare_ok = False
    assert cli.main(["run", *base_args(tmp_path)], env=server_env(server.port)) == 2
    assert "switch" not in world.kinds()
    assert "prepare" in capsys.readouterr().err


def test_live_episode_order_prepare_takeover_run_release_outcome(tmp_path, monkeypatch, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    console = ScriptedConsole(lines=["", "s", "went fine"], keys=[ENTER])
    world = World(monkeypatch, console)
    assert cli.main(["run", *base_args(tmp_path), "--episodes", "1"], env=server_env(server.port)) == 0
    kinds = world.kinds()
    assert kinds[:3] == ["node", "cms", "backend"]
    for side in ("left", "right"):
        activate = world.index(("switch", side, (CTRL,), ()))
        deactivate = world.index(("switch", side, (), (CTRL,)))
        assert world.index(("prepare", False)) < activate < deactivate
    node = world.nodes[0]
    assert node.command_mode == "live"
    assert node.arm_msgs["left"] > 20
    assert node.arm_msgs["right"] > 20
    assert node.gripper_msgs  # the runner commanded the grippers
    (ep,) = episode_dirs(tmp_path)
    result = json.loads((ep / "result.json").read_text())
    assert result["outcome"] == "s"
    assert result["note"] == "went fine"
    assert result["stats"]["chunks"] >= 1
    assert node.shut


def test_dry_run_builds_no_motion_objects(tmp_path, monkeypatch, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole(lines=["s", ""], keys=[ENTER]))

    def forbidden(name):
        def fail(*args, **kwargs):
            world.log.append(("forbidden", name))
            raise AssertionError(f"{name} must not be used in a dry run")

        return fail

    for name in ("make_controller_managers", "make_backend", "prepare", "GuardedSession", "Session"):
        monkeypatch.setattr(cli, name, forbidden(name))
    assert cli.main(["run", *base_args(tmp_path), "--dry-run", "--episodes", "1"], env=server_env(server.port)) == 0
    assert world.log == [("node", "preview")]
    node = world.nodes[0]
    assert node.arm_msgs["left"] > 20  # preview stream ran
    assert node.arm_msgs["right"] > 20
    assert node.gripper_msgs == []
    (ep,) = episode_dirs(tmp_path)
    assert json.loads((ep / "result.json").read_text())["outcome"] == "s"


def test_quit_at_the_start_gate_releases_without_running(tmp_path, monkeypatch, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole(lines=["k"], keys=[None, "q"]))
    monkeypatch.setattr(cli, "Runner", lambda *a, **k: pytest.fail("the runner must not be built"))
    assert cli.main(["run", *base_args(tmp_path)], env=server_env(server.port)) == 0
    assert "prepare" not in world.kinds()  # k: keep the current pose
    for side in ("left", "right"):
        assert world.index(("switch", side, (CTRL,), ())) < world.index(("switch", side, (), (CTRL,)))


class _Runner:
    """Stands in for Runner: records the generation, ends by operator q at once."""

    generations: ClassVar[list[int]] = []

    def __init__(self, *args, **kwargs):
        pass

    def run_episode(self, generation):
        type(self).generations.append(generation)
        return EpisodeOutcome("operator", "operator pressed q", 3, {"steps": 3})


def test_freeze_ends_the_run_with_exit_3(tmp_path, monkeypatch, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole(lines=["k", "the arm stopped"], keys=[ENTER]))

    class Freezing(_Runner):
        def run_episode(self, generation):
            return EpisodeOutcome("freeze", "cam/head age 6.00 s > obs_stale_freeze_s 5.0", 7, {"steps": 7})

    monkeypatch.setattr(cli, "Runner", Freezing)
    assert cli.main(["run", *base_args(tmp_path), "--episodes", "3"], env=server_env(server.port)) == 3
    assert len(episode_dirs(tmp_path)) == 1
    result = json.loads((episode_dirs(tmp_path)[0] / "result.json").read_text())
    assert result["outcome"] == "freeze"
    assert result["note"] == "the arm stopped"
    assert world.cms["left"].state == world.cms["right"].state == "inactive"


def test_episodes_get_distinct_generations(tmp_path, monkeypatch, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    World(monkeypatch, ScriptedConsole(lines=["f", "", "x", ""], keys=[ENTER, ENTER]))
    _Runner.generations = []
    monkeypatch.setattr(cli, "Runner", _Runner)
    assert cli.main(["run", *base_args(tmp_path), "--dry-run", "--episodes", "2"], env=server_env(server.port)) == 0
    assert len(set(_Runner.generations)) == 2
    assert [json.loads((d / "result.json").read_text())["outcome"] for d in episode_dirs(tmp_path)] == ["f", "x"]


def test_dry_run_start_gate_waits_for_the_first_hold_command(tmp_path, monkeypatch, stub_server):
    """The dry-run streamer starts right before the start gate; a first tick that has not happened yet is not a
    stopped stream (it used to end the episode 'FREEZE before the start' whenever the thread was slow to tick)."""
    server = stub_server(layout_id="munich16", latency_s=0.01)
    World(monkeypatch, ScriptedConsole(lines=["f"], keys=[ENTER]))
    _Runner.generations = []
    monkeypatch.setattr(cli, "Runner", _Runner)
    run = cli.Streamer._run  # noqa: SLF001

    def late_first_tick(self):
        time.sleep(0.03)  # the first hold command comes 30 ms after start()
        run(self)

    monkeypatch.setattr(cli.Streamer, "_run", late_first_tick)
    assert cli.main(["run", *base_args(tmp_path), "--dry-run", "--episodes", "1"], env=server_env(server.port)) == 0
    assert _Runner.generations == [1]


def test_dry_run_waits_for_discovery_before_judging_the_streams(tmp_path, monkeypatch, stub_server):
    """Like a Session, a dry run gives DDS discovery `takeover.discovery_wait_s` from the node's start before the start
    gate judges the streams: a camera whose first frame has not arrived yet is not a dead one (it used to end the
    episode 'FREEZE before the start: cam/head never seen' when the node had only just started)."""
    server = stub_server(layout_id="munich16", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole(lines=["f"], keys=[ENTER]))
    _Runner.generations = []
    monkeypatch.setattr(cli, "Runner", _Runner)
    make_node = world._make_node  # noqa: SLF001

    def fresh_node(cfg, command_mode):
        node = make_node(cfg, command_mode)
        t0 = time.monotonic()
        node.uptime_s = lambda: time.monotonic() - t0
        snapshot = node.snapshot

        def head_after_discovery():
            snap = snapshot()
            if node.uptime_s() < 0.2:  # the head camera's first frame arrives 0.2 s after the node started
                snap.ages["cam/head"] = float("inf")
                del snap.images["head"]
            return snap

        node.snapshot = head_after_discovery
        return node

    monkeypatch.setattr(cli, "make_node", fresh_node)
    args = base_args(tmp_path, "takeover.discovery_wait_s=0.4")
    assert cli.main(["run", *args, "--dry-run", "--episodes", "1"], env=server_env(server.port)) == 0
    assert _Runner.generations == [1]


def test_second_sigint_does_not_abort_release(tmp_path, monkeypatch, capfd, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)

    class Interrupted(_Runner):
        def run_episode(self, generation):
            signal.raise_signal(signal.SIGINT)  # the operator's first Ctrl-C
            time.sleep(5)
            raise AssertionError("the first Ctrl-C did not interrupt the episode")

    monkeypatch.setattr(cli, "Runner", Interrupted)

    def second_ctrl_c():
        signal.raise_signal(signal.SIGINT)  # a second Ctrl-C while deactivating

    def hook(cms):
        for cm in cms.values():
            cm.on_deactivate = second_ctrl_c

    world = World(monkeypatch, ScriptedConsole(lines=["k"], keys=[ENTER]), cm_hook=hook)
    code = cli.main(["run", *base_args(tmp_path)], env=server_env(server.port))
    assert code == 130
    assert world.cms["left"].state == world.cms["right"].state == "inactive"
    assert ("switch", "left", (), (CTRL,)) in world.log
    assert ("switch", "right", (), (CTRL,)) in world.log
    assert "releasing — wait, or use the E-stop" in capfd.readouterr().err
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored


def test_sigint_during_a_normal_release_is_ignored(tmp_path, monkeypatch, capfd, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)

    def hook(cms):
        cms["left"].on_deactivate = lambda: signal.raise_signal(signal.SIGINT)

    world = World(monkeypatch, ScriptedConsole(lines=["k", "s", ""], keys=[ENTER]), cm_hook=hook)
    monkeypatch.setattr(cli, "Runner", _Runner)
    assert cli.main(["run", *base_args(tmp_path), "--episodes", "1"], env=server_env(server.port)) == 0
    assert world.cms["left"].state == world.cms["right"].state == "inactive"
    assert "releasing — wait, or use the E-stop" in capfd.readouterr().err
    assert json.loads((episode_dirs(tmp_path)[0] / "result.json").read_text())["outcome"] == "s"


def test_takeover_refused_by_a_foreign_publisher_exits_2(tmp_path, monkeypatch, capsys, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    world = World(monkeypatch, ScriptedConsole(lines=["k"], keys=[ENTER]))
    world.foreign["/right/gello/joint_states"] = 12
    assert cli.main(["run", *base_args(tmp_path)], env=server_env(server.port)) == 2
    assert "/right/gello/joint_states" in capsys.readouterr().err
    assert "switch" not in world.kinds()
    assert world.nodes[0].arm_msgs == {"left": 0, "right": 0}  # refused before streaming anything


def test_failed_deactivation_exits_5_and_never_says_released(tmp_path, monkeypatch, capsys, stub_server):
    """The left controller cannot be deactivated: the hold keeps streaming, the operator gets a loud message and
    confirms with Enter; the run exits 5 (not 0, not 'Released.')."""
    from robot_client import takeover

    monkeypatch.setattr(takeover, "RELEASE_RETRY_S", 0.5)
    server = stub_server(layout_id="munich16", latency_s=0.01)

    def hook(cms):
        cms["left"].refuse_deactivate = True

    console = ScriptedConsole(lines=["k"], keys=[ENTER, None, "x", ENTER])
    world = World(monkeypatch, console, cm_hook=hook)
    monkeypatch.setattr(cli, "Runner", _Runner)
    code = cli.main(["run", *base_args(tmp_path), "--episodes", "1"], env=server_env(server.port))
    out = capsys.readouterr()
    assert code == cli.EXIT_STILL_ACTIVE == 5
    assert "Released." not in out.out
    assert "CONTROLLER STILL ACTIVE on /left/controller_manager" in out.err
    assert "has been released" not in out.err
    assert world.cms["left"].state == "active"
    assert world.cms["right"].state == "inactive"
    assert len([e for e in world.log if e == ("switch", "left", (), (CTRL,))]) > 1  # retried
    assert console.discards >= 2  # start gate + the still-active prompt: type-ahead never confirms
    assert not console.keys  # the operator's Enter ended the wait
    (ep,) = episode_dirs(tmp_path)
    assert json.loads((ep / "result.json").read_text())["outcome"] == "freeze"


def test_operator_ack_discards_type_ahead():
    console = ScriptedConsole(keys=[ENTER, ENTER])
    ack = cli._operator_ack(console)  # noqa: SLF001
    assert ack() is False  # the first call only discards what was typed before the message
    assert console.discards == 1
    assert ack() is True


def test_operator_ack_starts_over_after_an_enter():
    """A later CONTROLLER STILL ACTIVE message (another arm) needs its own Enter typed after it was shown."""
    console = ScriptedConsole(keys=[ENTER, ENTER, ENTER])
    ack = cli._operator_ack(console)  # noqa: SLF001
    assert ack() is False
    assert ack() is True
    assert ack() is False  # the next message: type-ahead discarded again
    assert console.discards == 2


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_guard_turns_signals_into_one_interrupt(sig, capfd):
    """SIGHUP too: python is PID 1 in the container, so an unhandled SIGHUP (closed terminal) would be ignored."""
    guard = cli.SignalGuard()
    guard.install()
    try:
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(sig)
        signal.raise_signal(sig)  # after the first: only the message
        assert cli.RELEASING_MESSAGE in capfd.readouterr().err
    finally:
        guard.restore()
    assert signal.getsignal(signal.SIGHUP) is not guard._handle  # noqa: SLF001


def test_signal_guard_survives_a_hung_up_terminal(monkeypatch):
    """SIGHUP closes the terminal: writing the releasing message then fails (EIO); that must not abort the release."""
    guard = cli.SignalGuard()

    def eio(fd, data):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(cli.os, "write", eio)
    with guard.releasing():
        guard._handle(signal.SIGHUP, None)  # noqa: SLF001 - no exception


# ---- release ----


def test_release_deactivates_even_what_we_did_not_activate(tmp_path, monkeypatch, capsys):
    def all_active(cms):
        for cm in cms.values():
            cm.state = "active"

    world = World(monkeypatch, ScriptedConsole(), cm_hook=all_active)
    assert cli.main(["release", *base_args(tmp_path), "--arm", "right"], env={}) == 0
    assert world.log[0] == ("node", "none")
    assert ("switch", "right", (), (CTRL,)) in world.log
    assert world.cms["left"].state == "active"  # not asked for
    assert world.cms["right"].state == "inactive"


# ---- console ----


def test_console_reads_keys_and_lines_from_a_pipe():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"k\n\nsome note\n")
    os.close(write_fd)
    with os.fdopen(read_fd) as stream:
        console = cli.Console(stream)
        assert console.read_line("? ") == "k"
        assert console.poll() == ENTER
        assert console.read_line("note: ") == "some note"
        assert console.poll() is None
        assert console.read_line("? ") is None  # EOF
        console.close()


# ---- stale keys on a TTY never answer a motion gate ----


@pytest.fixture
def pty_console():
    """A real Console on a pseudo-terminal (cbreak like on the operator's TTY); yields (console, type_fn)."""
    pty = pytest.importorskip("pty")
    try:
        master, slave = pty.openpty()
    except OSError as e:
        pytest.skip(f"no pty available: {e}")
    stream = os.fdopen(slave, "r")
    console = cli.Console(stream, out=io.StringIO())
    assert console._tty  # noqa: SLF001 - the test needs the TTY path

    timers = []

    def type_keys(data: bytes, after_s: float = 0.0):
        if after_s:
            timer = threading.Timer(after_s, os.write, (master, data))
            timers.append(timer)
            timer.start()
        else:
            os.write(master, data)
            time.sleep(0.05)  # let the line discipline queue it

    yield console, type_keys
    for timer in timers:
        timer.join()
    console.close()
    stream.close()
    os.close(master)


def test_tty_enter_typed_before_a_prompt_does_not_answer_it(pty_console):
    console, type_keys = pty_console
    type_keys(b"\n")  # Enter pressed while the client was busy (connecting, prepare, PTP, takeover)
    type_keys(b"k\n", after_s=0.3)  # the operator's real answer, after the prompt is shown
    assert console.read_line(cli.RESET_PROMPT) == "k"
    type_keys(b"\n")  # pressed right after answering, before the next prompt
    type_keys(b"q\n", after_s=0.3)
    assert console.read_line("Workspace clear, E-stop at hand? [Enter] start / [q] quit: ") == "q"


class _HealthyStreamer:
    fault = None

    def healthy(self):
        return True


class _LiveWorker:
    error = None

    def is_alive(self):
        return True


def test_tty_enter_typed_during_takeover_does_not_start_the_policy(pty_console, monkeypatch):
    console, type_keys = pty_console
    cfg = load_site_config(MUNICH_YAML, [], {})
    run = cli._RunCommand.__new__(cli._RunCommand)  # noqa: SLF001 - only what _wait_start reads
    run.cfg, run.console, run.worker = cfg, console, _LiveWorker()
    run.required = required_stream_keys(cfg)
    run.node = FakeNode(cfg, "live", world=None)
    type_keys(b"\n")  # Enter pressed while prepare / takeover were still running
    type_keys(b"q", after_s=0.3)
    verdict, _ = run._wait_start(_HealthyStreamer())  # noqa: SLF001
    assert verdict == "quit"


def test_start_gate_discards_pending_keys_before_prompting(tmp_path, monkeypatch, stub_server):
    server = stub_server(layout_id="munich16", latency_s=0.01)
    console = ScriptedConsole(lines=["k"], keys=["q"])
    World(monkeypatch, console)
    assert cli.main(["run", *base_args(tmp_path)], env=server_env(server.port)) == 0
    assert console.discards == 1


def test_dry_run_streamer_follows_its_own_output_not_the_static_robot():
    """A dry run's preview never moves the robot, so judging the leash against the real measured joints would stall
    the preview after max_track_lag_rad; the preview streamer's `measured` is its own last output."""
    cfg = load_site_config(MUNICH_YAML, ["smoother.impl=clamp"], {})
    node = FakeNode(cfg, "preview", world=None)  # measured stays at the start pose
    sides = ("left", "right")
    adapters = cli.make_adapters(cfg, sides)
    streamer = cli.make_streamer(cfg, node, sides, adapters, preview=True)
    start = {side: node.measured(side) for side in sides}
    streamer.start(start, thread=False)
    target = start["right"].copy()
    target[0] += 0.8  # beyond max_track_lag_rad (0.5) from the static measured pose
    t = 0.0
    for k in range(1, 401):
        if k % 5 == 1:
            wp = start["right"] + (target - start["right"]) * min(1.0, k / 200)
            streamer.set_waypoint("right", wp)
        t += 0.01
        out = streamer.step(t)
    assert streamer.leash_events == 0
    assert out["right"][0] - start["right"][0] > 0.7

    live = cli.make_streamer(cfg, node, sides, adapters)  # a live streamer still judges the real arm
    live.start(start, thread=False)
    t = 0.0
    for k in range(1, 401):
        if k % 5 == 1:
            live.set_waypoint("right", start["right"] + (target - start["right"]) * min(1.0, k / 200))
        t += 0.01
        live.step(t)
    assert live.leash_events > 0
