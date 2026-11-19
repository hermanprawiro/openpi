"""End-to-end tests of the robot client against `fake_robot` and the stub policy server (plan Task 19).

Every test starts `python -m robot_client.fake_robot` and `python -m robot_client.stub_server` as subprocesses and
then the client CLI itself (`python -m robot_client ...`), all on `ROS_DOMAIN_ID=42` with the shipped UDP-only Fast
DDS profile, exactly as the operator would run it. The client's gates are answered through its stdin when its prompt
appears on stdout (never typed ahead: once an episode runs, the runner reads a stdin line per tick as a key).

Evidence comes from the fake robot's events file (every command, switch and action with its monotonic receive time;
commands also carry the sender's header stamp) and from the client's run directory (`events.log`, episode files).
Runs inside the client container (`ros` marker); each test takes ~15-30 s.
"""

from collections.abc import Callable
import contextlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.ros

rclpy = pytest.importorskip("rclpy")

PKG_PARENT = Path(__file__).resolve().parents[2]  # the directory that holds robot_client/
SITE = PKG_PARENT / "robot_client" / "robot_site.munich.yaml"
UDP_PROFILE = PKG_PARENT / "robot_client" / "fastdds_udp_only.xml"
DOMAIN_ID = 42
CONTROLLER = "joint_impedance_controller"
SIDES = ("left", "right")
RIGHT_J7 = 6  # right arm joint 7 (index in the arm); the stub sweeps it (action dim 14)

START_TIMEOUT_S = 30.0  # a subprocess becomes ready
PROMPT_TIMEOUT_S = 60.0  # a CLI prompt appears
EXIT_TIMEOUT_S = 30.0  # the CLI exits after its last answer
TERMINATE_GRACE_S = 3.0  # SIGTERM -> SIGKILL at teardown

RESET_PROMPT = r"\[Enter\] move both arms to the start pose / \[k\] keep current pose"
START_PROMPT = "Press Enter to start the policy"
OUTCOME_PROMPT = r"Outcome\? \[s\]uccess"
NOTE_PROMPT = r"Note \(optional"


# ----------------------------------------------------------------------------------------------------
# Subprocess plumbing
# ----------------------------------------------------------------------------------------------------


def ros_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("POLICY_SERVER_")}
    env.update(
        ROS_DOMAIN_ID=str(DOMAIN_ID),
        RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
        FASTRTPS_DEFAULT_PROFILES_FILE=str(UDP_PROFILE),
        PYTHONPATH=os.pathsep.join(p for p in (str(PKG_PARENT), env.get("PYTHONPATH", "")) if p),
        PYTHONUNBUFFERED="1",
    )
    env.update(extra)
    return env


class Proc:
    """A subprocess in its own process group with stdout+stderr captured (and copied to `log_path`)."""

    def __init__(self, name: str, args: list[str], env: dict[str, str], log_path: Path, *, stdin: bool = False):
        self.name = name
        self.args = args
        self._log = open(log_path, "wb")  # noqa: SIM115 - closed by the reader thread
        self.p = subprocess.Popen(
            args,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._text = ""
        self._pos = 0
        self._cond = threading.Condition()
        self._reader = threading.Thread(target=self._read, name=f"read-{name}", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        try:
            while True:
                data = os.read(self.p.stdout.fileno(), 65536)
                if not data:
                    break
                self._log.write(data)
                self._log.flush()
                with self._cond:
                    self._text += data.decode(errors="replace")
                    self._cond.notify_all()
        finally:
            self._log.close()
            with self._cond:
                self._cond.notify_all()

    @property
    def output(self) -> str:
        with self._cond:
            return self._text

    def tail(self, n: int = 4000) -> str:
        return f"\n----- {self.name} output (last {n} chars) -----\n{self.output[-n:]}"

    def expect(self, pattern: str, timeout: float = PROMPT_TIMEOUT_S) -> re.Match:
        """Wait until `pattern` (regex) appears after the previous match; fail with the output on timeout or exit."""
        regex = re.compile(pattern)
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                match = regex.search(self._text, self._pos)
                if match:
                    self._pos = match.end()
                    return match
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (self.p.poll() is not None and not self._reader.is_alive()):
                    break
                self._cond.wait(min(remaining, 0.1))
        why = f"exited with {self.p.returncode}" if self.p.poll() is not None else f"timed out after {timeout:g} s"
        pytest.fail(f"{self.name}: {why} waiting for {pattern!r}{self.tail()}")

    def send(self, line: str) -> None:
        self.p.stdin.write((line + "\n").encode())
        self.p.stdin.flush()

    def wait(self, timeout: float = EXIT_TIMEOUT_S) -> int:
        try:
            rc = self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"{self.name}: still running after {timeout:g} s{self.tail()}")
        self._reader.join(5.0)
        return rc

    def alive(self) -> bool:
        return self.p.poll() is None

    def kill(self, sig: int = signal.SIGKILL) -> None:
        if self.alive():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.p.pid, sig)

    def stop(self) -> None:
        """SIGTERM the process group, SIGKILL after TERMINATE_GRACE_S; idempotent."""
        if self.alive():
            self.kill(signal.SIGTERM)
            try:
                self.p.wait(TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                self.kill(signal.SIGKILL)
                self.p.wait(TERMINATE_GRACE_S)
        if self.p.stdin is not None:
            with contextlib.suppress(OSError):
                self.p.stdin.close()
        self._reader.join(5.0)


def wait_until(pred: Callable[[], object], timeout: float, what: str, proc: Proc | None = None):
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if proc is not None and not proc.alive():
            pytest.fail(f"{proc.name} exited with {proc.p.returncode} while waiting for {what}{proc.tail()}")
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout:g} s waiting for {what}" + (proc.tail() if proc else ""))
        time.sleep(0.05)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:  # a line still being written
            break
    return records


class FakeRobot:
    def __init__(self, proc: Proc, events_path: Path):
        self.proc = proc
        self.events_path = events_path

    def events(self, kind: str | None = None, **match) -> list[dict]:
        return [
            e
            for e in read_jsonl(self.events_path)
            if (kind is None or e["kind"] == kind) and all(e.get(k) == v for k, v in match.items())
        ]


class Harness:
    """Starts the fake robot, the stub server and the client; tears every process group down at the end."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.out = tmp_path / "out"
        self.procs: list[Proc] = []
        self.port: int | None = None

    def _start(self, name: str, args: list[str], *, stdin: bool = False, **env: str) -> Proc:
        proc = Proc(name, args, ros_env(**env), self.tmp / f"{name}_{len(self.procs)}.log", stdin=stdin)
        self.procs.append(proc)
        return proc

    def fake_robot(self, *args: str) -> FakeRobot:
        events = self.tmp / "fake_robot_events.jsonl"
        events.unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "robot_client.fake_robot", "--site", str(SITE), "--events", str(events), *args]
        robot = FakeRobot(self._start("fake_robot", cmd), events)
        wait_until(lambda: robot.events("ready"), START_TIMEOUT_S, "the fake robot to be ready", robot.proc)
        return robot

    def stub(self, *args: str, port: int | None = None) -> Proc:
        if port is None:
            port = self.port or free_port()
        self.port = port
        cmd = [sys.executable, "-m", "robot_client.stub_server", "--host", "127.0.0.1", "--port", str(port), *args]
        proc = self._start("stub_server", cmd)
        proc.expect("listening on", START_TIMEOUT_S)
        return proc

    def client(self, command: str, *args: str, run_name: str = "t") -> Proc:
        cmd = [
            sys.executable,
            "-m",
            "robot_client",
            command,
            "--site",
            str(SITE),
            "--set",
            f"ros.domain_id={DOMAIN_ID}",
            "--set",
            f"out_dir={self.out}",
            "--run-name",
            run_name,
            *args,
        ]
        env = {"POLICY_SERVER_HOST": "127.0.0.1"}
        if self.port is not None:
            env["POLICY_SERVER_PORT"] = str(self.port)
        return self._start(f"client_{command}", cmd, stdin=True, **env)

    def run_dir(self, run_name: str = "t") -> Path:
        return self.out / run_name

    def client_events(self, run_name: str = "t") -> list[dict]:
        return read_jsonl(self.run_dir(run_name) / "events.log")

    def episode(self, idx: int = 0, run_name: str = "t") -> dict:
        ep = self.run_dir(run_name) / f"episode_{idx:03d}"
        return {
            "result": json.loads((ep / "result.json").read_text()),
            "stream_stats": json.loads((ep / "stream_stats.json").read_text()),
            "steps": read_jsonl(ep / "steps.jsonl"),
        }

    def close(self) -> None:
        for proc in reversed(self.procs):
            proc.stop()


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    try:
        yield h
    finally:
        h.close()


# ----------------------------------------------------------------------------------------------------
# Driving the CLI
# ----------------------------------------------------------------------------------------------------


def press_start(client: Proc) -> None:
    """Answer the start gate with Enter once it accepts it (an Enter while the streams settle is refused and
    consumed, so retry until the policy runs)."""
    client.expect(START_PROMPT)
    deadline = time.monotonic() + PROMPT_TIMEOUT_S
    while True:
        client.send("")
        match = client.expect(r"Policy running|Not starting: [^\n]*")
        if match.group(0).startswith("Policy running"):
            return
        if time.monotonic() > deadline:
            pytest.fail(f"the start gate never accepted Enter{client.tail()}")
        time.sleep(0.5)


def run_one_episode(client: Proc, *, prepare: bool) -> int:
    """`run --episodes 1` until the episode timeout: reset choice, start, outcome `s`, empty note; the exit code."""
    client.expect(RESET_PROMPT)
    client.send("" if prepare else "k")
    press_start(client)
    client.expect(OUTCOME_PROMPT)
    client.send("s")
    client.expect(NOTE_PROMPT)
    client.send("")
    return client.wait()


def commands(robot: FakeRobot, side: str) -> list[dict]:
    """Commands a robot client sent to `side` (frame_id `robot_client:<process id>`), in arrival order."""
    return [e for e in robot.events("command", side=side) if e["frame_id"].startswith("robot_client:")]


def switches(robot: FakeRobot, side: str, action: str) -> list[dict]:
    """Successful switch_controller calls of `side` that activate / deactivate the impedance controller."""
    return [e for e in robot.events("switch", side=side) if e["ok"] and CONTROLLER in e[action]]


def max_dev(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


class RosPeer:
    """A ROS node inside the test process (own rclpy context on DOMAIN_ID) for listening and foreign publishing."""

    def __init__(self):
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.signals import SignalHandlerOptions

        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=DOMAIN_ID, signal_handler_options=SignalHandlerOptions.NO)
        self.node = Node("integration_test_peer", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True, name="integration-peer")
        self.thread.start()
        self.counts: dict[str, int] = {}

    def count(self, msg_type, topic: str) -> None:
        from rclpy.qos import qos_profile_sensor_data

        self.counts[topic] = 0

        def on_msg(_msg, topic=topic):
            self.counts[topic] += 1

        self.node.create_subscription(msg_type, topic, on_msg, qos_profile_sensor_data)

    def publish_periodically(self, topic: str, make_msg: Callable[[], object], hz: float) -> None:
        msg = make_msg()
        pub = self.node.create_publisher(type(msg), topic, 10)
        self.node.create_timer(1.0 / hz, lambda: pub.publish(make_msg()))

    def switch(self, side: str, *, activate: tuple[str, ...] = (), deactivate: tuple[str, ...] = ()) -> bool:
        """The organizers' tooling: call `/<side>/controller_manager/switch_controller` (STRICT) and wait."""
        from controller_manager_msgs.srv import SwitchController

        client = self.node.create_client(SwitchController, f"/{side}/controller_manager/switch_controller")
        try:
            assert client.wait_for_service(timeout_sec=START_TIMEOUT_S), f"no switch_controller for {side}"
            request = SwitchController.Request(
                activate_controllers=list(activate),
                deactivate_controllers=list(deactivate),
                strictness=SwitchController.Request.STRICT,
            )
            future = client.call_async(request)
            wait_until(future.done, START_TIMEOUT_S, f"the {side} switch_controller answer")
            return bool(future.result().ok)
        finally:
            self.node.destroy_client(client)

    def close(self) -> None:
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(2.0)
        self.node.destroy_node()
        rclpy.try_shutdown(context=self.context)


@pytest.fixture
def peer():
    p = RosPeer()
    try:
        yield p
    finally:
        p.close()


# ----------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------


def test_takeover_ordering_and_continuous_stream(harness):
    robot = harness.fake_robot()
    harness.stub("--latency-s", "0.2")
    client = harness.client("run", "--episodes", "1", "--set", "safety.episode_timeout_s=5")
    rc = run_one_episode(client, prepare=True)
    assert rc == 0, client.tail()
    assert harness.episode()["result"]["stats"]["ended_by"] == "timeout"

    released = [e for e in harness.client_events() if e["kind"] == "released"]
    assert len(released) == 1, harness.client_events()
    (episode_start,) = [e for e in harness.client_events() if e["kind"] == "episode_start"]
    for side in SIDES:
        cmds = commands(robot, side)
        activates = switches(robot, side, "activate")
        deactivates = switches(robot, side, "deactivate")
        assert len(activates) == 1, robot.events("switch", side=side)
        activate = activates[0]
        # prepare ran before the takeover: PTP of this arm and the spine finished before the activation
        ptp_done = robot.events("ptp_result", side=side)
        assert ptp_done, f"{side}: prepare sent no PTP goal"
        assert ptp_done[-1]["reached"]
        assert ptp_done[-1]["mono"] < activate["mono"]
        assert robot.events("spine_result")[-1]["mono"] < activate["mono"]

        # >= 20 hold commands, each equal to measured, immediately before the activation. Judged by the sender's
        # stamp (published before the robot handled the switch), not by which of two concurrent callbacks ran first.
        before = [c for c in cmds if c["stamp"] < activate["wall"]]
        assert len(before) >= 20, f"{side}: only {len(before)} commands before the activation"
        devs = [max_dev(c["position"], c["measured"]) for c in before[-20:]]
        assert max(devs) <= 0.02, f"{side}: the last 20 hold commands deviate up to {max(devs):.4f} rad"

        # activation window: from the activation until the policy starts, every command is the arm's own pose
        window = [c for c in cmds if c["active"] and c["mono"] < episode_start["mono"]]
        assert len(window) >= 5, f"{side}: only {len(window)} commands between activation and the policy start"
        dev = max(max_dev(c["position"], c["measured"]) for c in window)
        assert dev <= 1e-3, f"{side}: a command before the first policy waypoint is {dev:.5f} rad from measured"

        # the impedance controller was deactivated (by the release) after the activation, while still streaming
        deactivate = [d for d in deactivates if d["mono"] > activate["mono"]]
        assert deactivate, robot.events("switch", side=side)
        deactivate = deactivate[0]
        after = [c for c in cmds if c["mono"] > deactivate["mono"]]
        assert len(after) >= 5, f"{side}: only {len(after)} command(s) after the deactivation"

        # continuous stream while active: no gap of 50 ms, boundaries included
        times = [activate["mono"]] + [c["mono"] for c in cmds if activate["mono"] < c["mono"] < deactivate["mono"]]
        times.append(deactivate["mono"])
        assert len(times) > 100, f"{side}: only {len(times) - 2} commands while active"
        gap = float(np.max(np.diff(times)))
        assert gap < 0.05, f"{side}: largest command gap while active is {gap * 1000:.1f} ms"

        # nothing published after the client stopped its stream (header stamps vs the client's `released` event)
        late = [c for c in cmds if c["stamp"] > released[0]["wall"]]
        assert not late, f"{side}: {len(late)} command(s) stamped after the stream stopped"


def test_dry_run_never_moves_anything(harness, peer):
    from sensor_msgs.msg import JointState

    robot = harness.fake_robot()
    harness.stub("--latency-s", "0.2")
    for side in SIDES:
        peer.count(JointState, f"/{side}/robot_client_preview/joint_states")
    client = harness.client("run", "--dry-run", "--episodes", "1", "--set", "safety.episode_timeout_s=3")
    press_start(client)  # a dry run has no reset prompt (nothing is moved)
    client.expect(OUTCOME_PROMPT)
    client.send("s")
    client.expect(NOTE_PROMPT)
    client.send("")
    assert client.wait() == 0, client.tail()

    episode = harness.episode()
    assert episode["result"]["stats"]["dry_run"] is True
    assert any(step["mode"] == "run" for step in episode["steps"]), "the policy never produced a waypoint"
    for kind in ("switch", "ptp_goal", "spine_goal", "gripper", "command"):
        assert not robot.events(kind), f"dry run caused {kind} events: {robot.events(kind)[:3]}"
    for side in SIDES:
        topic = f"/{side}/robot_client_preview/joint_states"
        assert peer.counts[topic] > 0, f"nothing received on {topic}"


def track_errors(steps: list[dict], side: str, joint: int) -> tuple[list[float], float]:
    """(|measured - last waypoint| of `joint` at each tick where the arm waited for the next chunk and had been
    resting there for >= 0.25 s, the joint's range of motion)."""
    errors, measured, rested, last_wp = [], [], 0, None
    for step in steps:
        q = step["measured"][side][joint]
        measured.append(q)
        if step.get("waypoints"):
            last_wp, rested = step["waypoints"][side][joint], 0
        elif step["mode"] == "wait" and last_wp is not None:
            rested += 1
            if rested >= 5:
                errors.append(abs(q - last_wp))
    return errors, max(measured) - min(measured)


def test_relative_mode_end_to_end(harness):
    # The fake controller reads commands relative to activation with joint 7 reversed; the stub sweeps right joint 7.
    robot = harness.fake_robot("--command-mode", "relative", "--signs", "1,1,1,1,1,1,-1")
    harness.stub("--latency-s", "0.5", "--amplitude", "0.1")
    signs = "[1, 1, 1, 1, 1, 1, -1]"
    relative = ["--set", "command_mode=relative_to_activation"]
    client = harness.client(
        "run",
        "--episodes",
        "1",
        *relative,
        "--set",
        f"joint_signs.left={signs}",
        "--set",
        f"joint_signs.right={signs}",
        "--set",
        "safety.episode_timeout_s=6",
    )
    assert run_one_episode(client, prepare=False) == 0, client.tail()
    episode = harness.episode()
    assert episode["result"]["stats"]["ended_by"] == "timeout"
    errors, moved = track_errors(episode["steps"], "right", RIGHT_J7)
    assert moved >= 0.03, f"right joint 7 moved only {moved:.4f} rad: the test cannot tell the direction"
    assert len(errors) >= 3, "too few settled ticks to judge the tracking"
    assert max(errors) <= 0.02, f"measured right joint 7 is up to {max(errors):.4f} rad from the policy target"
    # The fake controller really saw reversed commands on joint 7: command and measured moved in opposite directions.
    cmds = commands(robot, "right")
    d_cmd = cmds[-1]["position"][RIGHT_J7] - cmds[0]["position"][RIGHT_J7]
    d_meas = cmds[-1]["measured"][RIGHT_J7] - cmds[0]["measured"][RIGHT_J7]
    assert abs(d_meas) > 0.01, (d_cmd, d_meas)
    assert d_cmd * d_meas < 0, (d_cmd, d_meas)

    # Mismatched signs (the site forgot the -1): check-motion reports joint 7 as reversed and fails.
    check = harness.client("check-motion", "--arm", "right", "--joint", "7", *relative, run_name="motion")
    check.expect("Workspace clear")
    check.send("")
    assert check.wait(EXIT_TIMEOUT_S + 20) == 2, check.tail()
    report = json.loads((harness.run_dir("motion") / "check_motion_right.json").read_text())
    verdicts = {j["joint"]: j["verdict"] for j in report["data"]["joints"]}
    assert verdicts == {7: "reversed"}, report["lines"]
    assert report["data"]["joint_signs"] == [1, 1, 1, 1, 1, 1, -1]


def test_slow_controller_engages_leash_without_jump(harness):
    robot = harness.fake_robot("--tau", "0.4")
    # A fast 0.8 rad sweep that a tau = 0.4 s controller cannot follow within max_track_lag_rad. The stub's first
    # row is anywhere on the sweep, so the start-offset check is relaxed for this test only.
    harness.stub("--latency-s", "0.2", "--amplitude", "0.8")
    client = harness.client(
        "run",
        "--episodes",
        "1",
        "--set",
        "safety.episode_timeout_s=6",
        "--set",
        "safety.max_start_offset_rad=1.5",
    )
    assert run_one_episode(client, prepare=False) == 0, client.tail()
    episode = harness.episode()
    assert episode["result"]["stats"]["ended_by"] == "timeout"
    assert episode["stream_stats"]["streamer"]["leash_events"] > 0
    assert any(step["mode"] == "leash" for step in episode["steps"])
    max_step = 0.10  # safety.max_step_rad
    for side in SIDES:
        positions = np.array([c["position"] for c in commands(robot, side)])
        steps = np.max(np.abs(np.diff(positions, axis=0)), axis=1)
        assert steps.max() <= max_step, f"{side}: command step {steps.max():.4f} rad > max_step_rad {max_step}"


def test_missing_left_gripper_state_runs(harness):
    robot = harness.fake_robot("--left-gripper-state", "off")
    harness.stub("--latency-s", "0.2")
    check = harness.client("check-obs", "--seconds", "3", run_name="obs")
    assert check.wait() == 0, check.tail()
    report = json.loads((harness.run_dir("obs") / "check_obs.json").read_text())
    assert report["passed"] is True
    assert report["data"]["streams"]["grip/left"]["count"] == 0
    assert report["data"]["streams"]["grip/right"]["count"] > 0

    client = harness.client("run", "--episodes", "1", "--set", "safety.episode_timeout_s=3")
    assert run_one_episode(client, prepare=False) == 0, client.tail()
    episode = harness.episode()
    assert episode["result"]["stats"]["ended_by"] == "timeout"
    assert any(step["mode"] == "run" for step in episode["steps"])
    assert robot.events("gripper", side="right"), "the policy's gripper widths never reached the right gripper"


def test_refuses_when_foreign_publisher(harness, peer):
    from sensor_msgs.msg import JointState

    robot = harness.fake_robot()
    harness.stub("--latency-s", "0.2")
    topic = "/right/gello/joint_states"
    names = [f"right_fr3v2_joint{i}" for i in range(1, 8)]

    def gello() -> JointState:
        msg = JointState(name=names, position=[0.0] * 7)
        msg.header.frame_id = "gello"
        return msg

    peer.publish_periodically(topic, gello, hz=20.0)
    wait_until(lambda: robot.events("command", frame_id="gello"), START_TIMEOUT_S, "GELLO messages at the robot")

    client = harness.client("run", "--episodes", "1")
    client.expect(RESET_PROMPT)
    client.send("k")
    assert client.wait() == 2, client.tail()
    refusal = re.search(r"REFUSED: [^\n]*", client.output)
    assert refusal, client.tail()
    assert topic in refusal.group(0), client.tail()
    assert not robot.events("switch"), "a controller was switched despite the foreign publisher"
    for side in SIDES:
        assert not commands(robot, side), f"the client published {side} commands"


def test_checkpoint_swap_same_image(harness):
    harness.fake_robot()
    stub = harness.stub("--latency-s", "0.1")
    first = harness.client("check-infer", "--n", "2", run_name="munich16")
    assert first.wait() == 0, first.tail()
    stub.stop()
    harness.stub("--layout", "sim20")  # same port: the client config is unchanged
    second = harness.client("check-infer", "--n", "2", run_name="sim20")
    assert second.wait() == 2, second.tail()
    report = json.loads((harness.run_dir("sim20") / "check_infer.json").read_text())
    assert report["passed"] is False
    assert any("munich16" in line for line in report["lines"]), report["lines"]


def test_server_killed_mid_episode_releases(harness):
    robot = harness.fake_robot()
    stub = harness.stub("--latency-s", "0.2")
    client = harness.client("run", "--episodes", "1", "--set", "safety.episode_timeout_s=30")
    client.expect(RESET_PROMPT)
    client.send("k")
    press_start(client)
    time.sleep(2.0)
    killed = time.monotonic()
    stub.kill()
    client.expect(r"Episode 0 ended by freeze")
    client.expect(NOTE_PROMPT)
    client.send("")
    assert client.wait() == 3, client.tail()
    episode = harness.episode()
    assert episode["result"]["outcome"] == "freeze"
    (episode_start,) = [e for e in harness.client_events() if e["kind"] == "episode_start"]
    # The freeze ends the episode at the tick after the last recorded step (the runner's t0 is just after the event).
    freeze_mono = episode_start["mono"] + episode["steps"][-1]["t"] + 0.1
    for side in SIDES:
        deactivate = [d for d in switches(robot, side, "deactivate") if d["mono"] > killed]
        assert deactivate, f"{side}: not deactivated after the server died: {robot.events('switch', side=side)}"
        assert deactivate[0]["mono"] - killed < 15.0
        assert deactivate[0]["mono"] < commands(robot, side)[-1]["mono"], f"{side}: stream stopped before release"
        # the freeze is a hold: from the freeze until the deactivation every command is the same pose
        held = [c for c in commands(robot, side) if freeze_mono <= c["mono"] <= deactivate[0]["mono"]]
        assert len(held) >= 40, f"{side}: only {len(held)} commands between the freeze and the deactivation"
        spread = max(max_dev(c["position"], held[0]["position"]) for c in held)
        assert spread <= 1e-5, f"{side}: the commands moved {spread:.6f} rad after the freeze"


# ---- off-nominal controller states (final review C2, I1, I2) ----


def test_refuses_an_already_active_controller(harness):
    """An active controller (left on by bring-up / a failed release) would act on our first message: `run` and
    `check-motion` refuse before publishing anything and never switch it."""
    robot = harness.fake_robot("--controller-state", "active")
    harness.stub("--latency-s", "0.2")
    client = harness.client("run", "--episodes", "1")
    client.expect(RESET_PROMPT)
    client.send("k")
    assert client.wait() == 2, client.tail()
    refusal = re.search(r"REFUSED: [^\n]*", client.output)
    assert refusal, client.tail()
    assert "already active" in refusal.group(0), client.tail()
    assert "release --arm left" in refusal.group(0), client.tail()

    check = harness.client("check-motion", "--arm", "right", run_name="motion")
    check.expect("Workspace clear")
    check.send("")
    assert check.wait() == 2, check.tail()
    assert "already active" in check.output, check.tail()

    assert not robot.events("switch"), robot.events("switch")
    for side in SIDES:
        assert not commands(robot, side), f"the client published {side} commands into an active controller"


def test_refused_deactivation_keeps_holding_until_the_operator_confirms(harness):
    robot = harness.fake_robot("--refuse-deactivate", "left")
    harness.stub("--latency-s", "0.2")
    client = harness.client("run", "--episodes", "1", "--set", "safety.episode_timeout_s=3")
    client.expect(RESET_PROMPT)
    client.send("k")
    press_start(client)
    client.expect(r"CONTROLLER STILL ACTIVE on /left/controller_manager \(joint_impedance_controller\)")
    refused = [e for e in robot.events("switch", side="left") if not e["ok"]]
    assert len(refused) > 1, f"the deactivation was not retried: {refused}"
    n = len(commands(robot, "left"))
    time.sleep(1.0)
    assert len(commands(robot, "left")) >= n + 50, "the hold stopped streaming while the controller was active"
    client.send("")  # the operator handled it
    assert client.wait() == 5, client.tail()
    assert "Released." not in client.output, client.tail()
    assert "has been released" not in client.output, client.tail()

    left = commands(robot, "left")
    assert left[-1]["mono"] - refused[0]["mono"] >= 5.0, "the hold did not outlast the deactivation retries"
    gaps = np.diff([c["mono"] for c in left if c["mono"] >= refused[0]["mono"]])
    assert gaps.max() < 0.05, f"gap of {gaps.max() * 1000:.0f} ms in the hold while the controller was active"
    assert switches(robot, "right", "deactivate"), "the right arm was not released"
    released = [e for e in harness.client_events() if e["kind"] == "released"]
    assert released[-1]["still_active"] == ["left"], released
    assert harness.episode()["result"]["outcome"] == "freeze"


def test_external_mode_streams_until_the_operator_deactivates(harness, peer):
    robot = harness.fake_robot()
    harness.stub("--latency-s", "0.2")
    client = harness.client(
        "run", "--episodes", "1", "--set", "takeover.mode=external", "--set", "safety.episode_timeout_s=3"
    )
    client.expect(RESET_PROMPT)
    client.send("k")
    for side in SIDES:  # the organizers' tooling activates when the client asks
        client.expect(rf"\[{side}\] activate {CONTROLLER} on /{side}/controller_manager now")
        assert peer.switch(side, activate=(CONTROLLER,))
    press_start(client)
    for side in SIDES:
        client.expect(rf"\[{side}\] deactivate {CONTROLLER} on /{side}/controller_manager now")
    asked = time.monotonic()
    time.sleep(1.5)  # the operator takes a moment
    for side in SIDES:
        assert peer.switch(side, deactivate=(CONTROLLER,))
    client.expect(OUTCOME_PROMPT)
    client.send("s")
    client.expect(NOTE_PROMPT)
    client.send("")
    assert client.wait() == 0, client.tail()

    for side in SIDES:
        (deactivate,) = switches(robot, side, "deactivate")
        assert len(switches(robot, side, "activate")) == 1  # the client never switched anything itself
        cmds = commands(robot, side)
        waiting = [c["mono"] for c in cmds if asked <= c["mono"] <= deactivate["mono"]]
        assert len(waiting) >= 100, f"{side}: only {len(waiting)} commands while waiting for the operator"
        gaps = np.diff([asked, *waiting, deactivate["mono"]])
        assert gaps.max() < 0.05, f"{side}: {gaps.max() * 1000:.0f} ms gap while the controller was active"
        after = [c for c in cmds if c["mono"] > deactivate["mono"]]
        assert after, f"{side}: the stream ended before the controller read inactive"
