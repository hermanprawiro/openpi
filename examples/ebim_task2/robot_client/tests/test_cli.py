"""The CLI with monkeypatched ROS factories (fake node, controller managers, motion backend, console), the real
Streamer / Session / Runner and the stub policy server. No ROS."""

import json
import os
from pathlib import Path
import signal
import time
from typing import ClassVar

import numpy as np
import pytest
from robot_client import checks
from robot_client import cli
from robot_client.keys import ENTER
from robot_client.prepare import PrepareReport
from robot_client.prepare import StepResult
from robot_client.preprocess import Snapshot
from robot_client.runner import EpisodeOutcome

MUNICH_YAML = Path(__file__).resolve().parents[1] / "robot_site.munich.yaml"
CTRL = "joint_impedance_controller"


class ScriptedConsole:
    """`lines` answer read_line() in order (None = EOF once exhausted); `keys` answer poll() in order."""

    def __init__(self, lines=(), keys=()):
        self.lines = list(lines)
        self.keys = list(keys)
        self.prompts = []
        self.polls = 0
        self.closed = False

    def read_line(self, prompt):
        self.prompts.append(prompt)
        return self.lines.pop(0) if self.lines else None

    def poll(self):
        self.polls += 1
        assert self.polls < 2000, "the CLI kept polling for a key that the script never sends"
        return self.keys.pop(0) if self.keys else None

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

    def list_controllers(self):
        return {CTRL: self.state, "joint_state_broadcaster": "active"}

    def switch(self, activate, deactivate, timeout_s):
        if deactivate and self.on_deactivate is not None:
            self.on_deactivate()
        self.world.log.append(("switch", self.side, tuple(activate), tuple(deactivate)))
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
    assert "ROS_DOMAIN_ID is 7" in capsys.readouterr().err
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
