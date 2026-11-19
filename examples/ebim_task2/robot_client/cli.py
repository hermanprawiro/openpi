"""The operator's command line (spec §6.3-6.7): `python -m robot_client <subcommand> --site PATH [--set k=v ...]
[--run-name NAME]`.

    check-obs [--seconds S] [--norm-stats PATH]    streams, rates, shapes, joint names (no publishers)
    check-infer [--n N]                            policy server metadata, chunk shape, latency
    check-motion --arm SIDE [--joint all|1..7] [--delta RAD]
                                                   one arm follows small joint moves (prints joint_signs)
    prepare                                        controllers off -> PTP to start pose -> spine -> grippers open
    run [--episodes N] [--dry-run] [--arms SIDE ...]
                                                   episodes: prepare -> takeover -> policy -> release -> outcome
    release [--arm SIDE ...]                       deactivate the configured controller (recovery tool)

Exit codes: 0 ok, 2 check failed / refused, 3 an episode ended in FREEZE, 4 configuration or usage error, 5 a
controller could not be confirmed inactive after the release (CONTROLLER STILL ACTIVE; the operator confirmed with
Enter), 130 interrupted (Ctrl-C / SIGTERM / SIGHUP; everything we activated was released first).

Nothing moves without an operator Enter: every prepare, takeover and episode start waits for one. SIGINT, SIGTERM and
SIGHUP (a closed terminal; python is PID 1 in the container, where an unhandled SIGHUP is ignored) raise
KeyboardInterrupt, so the Session's release runs; while a release runs, and after the first interrupt, further
signals are ignored ("releasing — wait, or use the E-stop"). `run --dry-run` never builds a controller manager, motion
backend or Session and uses a node in `preview` mode (no command-topic or gripper publishers).

The ROS pieces are built by `make_node` / `make_controller_managers` / `make_backend` (imported lazily, so the
module loads without rclpy and tests replace the factories).
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import io
import logging
import math
import os
from pathlib import Path
import re
import select
import signal
import sys
import time

from eval_client.inference import InferenceWorker
import numpy as np

from robot_client import checks
from robot_client import safety
from robot_client.command import CommandAdapter
from robot_client.config import ARM_NAMES
from robot_client.config import ENV_HOST
from robot_client.config import N_JOINTS
from robot_client.config import ConfigError
from robot_client.config import SiteConfig
from robot_client.config import check_ros_env
from robot_client.config import load_site_config
from robot_client.keys import ENTER
from robot_client.keys import TerminalKeys
from robot_client.layout import LayoutMismatchError
from robot_client.policy_link import PolicyUnreachableError
from robot_client.policy_link import ensure_worker
from robot_client.policy_link import make_policy_client
from robot_client.prepare import prepare
from robot_client.recorder import RunRecorder
from robot_client.runner import Runner
from robot_client.runner import required_stream_keys
from robot_client.smoother import make_smoother
from robot_client.streamer import Streamer
from robot_client.takeover import ControllerStillActive
from robot_client.takeover import Session
from robot_client.takeover import TakeoverError
from robot_client.takeover import session_sides

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_FREEZE = 3
EXIT_CONFIG = 4
EXIT_STILL_ACTIVE = 5
EXIT_INTERRUPTED = 130

DEFAULT_SITE = "/config/robot_site.yaml"
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RELEASING_MESSAGE = "releasing — wait, or use the E-stop"
GATE_POLL_S = 0.05  # start gate: key / stream health polling period
FIRST_HOLD_WAIT_S = 1.0  # dry run: how long to wait for the streamer's first hold command
LIVE_OBS_WAIT_S = 3.0  # check-infer: how long to wait for a complete live snapshot

RESET_PROMPT = (
    "Reset the scene. Workspace clear? [Enter] move both arms to the start pose / [k] keep current pose / [q] quit: "
)
START_PROMPT = "Press Enter to start the policy (q to release and quit)"
OUTCOME_PROMPT = "Outcome? [s]uccess / [f]ail / [x] skip: "
NOTE_PROMPT = "Note (optional, Enter to skip): "


# ----------------------------------------------------------------------------------------------------
# Factories (the only places that touch ROS; tests replace them)
# ----------------------------------------------------------------------------------------------------


def make_node(cfg: SiteConfig, command_mode: str):
    """A started `RobotNode` (`live`, `preview` or `none`). rclpy is initialized by the node without its signal
    handlers, so Ctrl-C reaches us as KeyboardInterrupt and the ROS context survives for the release."""
    from robot_client.ros_io import RobotNode

    node = RobotNode(cfg, command_mode=command_mode)
    node.start()
    return node


def make_controller_managers(node, cfg: SiteConfig) -> dict:
    """One controller manager per arm, at `arms.<side>.controller_manager`."""
    from robot_client.ros_io import RosControllerManager

    return {
        side: RosControllerManager(node, arm.controller_manager, cfg.takeover.service_timeout_s)
        for side, arm in cfg.arms.items()
    }


def make_backend(node, cfg: SiteConfig):
    from robot_client.ros_io import RosMotionBackend

    return RosMotionBackend(node, cfg)


def make_console() -> "Console":
    return Console()


def make_streamer(
    cfg: SiteConfig,
    node,
    sides: Sequence[str],
    adapters: Mapping[str, CommandAdapter],
    *,
    preview: bool = False,
) -> Streamer:
    """A fresh streamer for one Session / episode (`start()` works once and faults are sticky). `preview` (dry run):
    the robot does not follow the preview, so the tracking leash judges the streamer's own last output instead of
    the real measured joints (else the preview would stall after max_track_lag_rad)."""
    limits = {side: tuple(np.asarray(x, dtype=np.float64) for x in cfg.safety.joint_limits[side]) for side in sides}
    streamer: Streamer | None = None

    def own_output(side: str) -> np.ndarray:
        return streamer.last_command(side)  # adapters are identity in a dry run (never activated)

    streamer = Streamer(
        sides,
        publish=node.publish_arm,
        measured=own_output if preview else node.measured,
        smoother_factory=lambda: make_smoother(cfg.smoother, dt=1.0 / cfg.stream_hz),
        adapters={side: adapters[side] for side in sides},
        limits=limits,
        safety=cfg.safety,
        hz=cfg.stream_hz,
        waypoint_dt=1.0 / cfg.policy.control_hz,
    )
    return streamer  # noqa: RET504 - own_output() reads the name


def make_adapters(cfg: SiteConfig, sides: Sequence[str]) -> dict[str, CommandAdapter]:
    return {
        side: CommandAdapter(cfg.command_mode, np.asarray(cfg.joint_signs[side], dtype=np.float64)) for side in sides
    }


# ----------------------------------------------------------------------------------------------------
# Signals and the guarded Session
# ----------------------------------------------------------------------------------------------------


class SignalGuard:
    """SIGINT / SIGTERM / SIGHUP -> KeyboardInterrupt, once. After that first interrupt, and while `releasing()` is active,
    further signals only print RELEASING_MESSAGE: nothing may cut a release short."""

    def __init__(self):
        self._releasing = 0
        self._interrupted = False
        self._previous: dict = {}

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self._previous[sig] = signal.signal(sig, self._handle)

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()

    def _handle(self, signum, frame) -> None:
        if self._releasing or self._interrupted:
            with contextlib.suppress(OSError):  # a hung-up terminal (EIO) must not abort the release
                os.write(2, f"\n{RELEASING_MESSAGE}\n".encode())  # no print(): not re-entrant in a handler
            return
        self._interrupted = True
        raise KeyboardInterrupt(signal.Signals(signum).name)

    @contextlib.contextmanager
    def releasing(self):
        self._releasing += 1
        try:
            yield
        finally:
            self._releasing -= 1


class GuardedSession(Session):
    """A `Session` whose release (hold, deactivate, stop the stream) runs with further signals ignored. An interrupt
    landing before the deferral is entered re-runs the release (`Session._close_guarded`)."""

    def __init__(self, *args, guard: SignalGuard, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = guard

    def _close(self) -> bool:
        with self._guard.releasing():
            return super()._close()


def _operator_ack(console) -> Callable[[], bool]:
    """The release's CONTROLLER STILL ACTIVE wait: True once the operator pressed Enter after the message was shown
    (the first call discards keys typed before it, so a type-ahead Enter never confirms). After a True the next call
    starts over: a later message for another arm needs its own Enter."""
    shown = False

    def ack() -> bool:
        nonlocal shown
        if not shown:
            shown = True
            console.discard_pending()
            return False
        if console.poll() == ENTER:
            shown = False
            return True
        return False

    return ack


# ----------------------------------------------------------------------------------------------------
# Operator console
# ----------------------------------------------------------------------------------------------------


class Console:
    """Operator input on stdin: `poll()` (a `KeySource`: the next key, never blocks) and `read_line(prompt)` (a whole
    line, blocking; None at end of input).

    On a TTY, keys are single key presses (`TerminalKeys`, cbreak); `read_line` closes it (normal echo and line
    editing), discards whatever was typed before the prompt, reads, and reopens it. `discard_pending()` does the same
    discard for a key-polled gate (the start gate). Anything else (a pipe, a file) feeds both from one line buffer read straight from
    the file descriptor, so a scripted stdin works: `poll()` returns the first character of the next line (an empty
    line is Enter).
    """

    def __init__(self, stream=None, out=None):
        self._stream = sys.stdin if stream is None else stream
        self._out = sys.stdout if out is None else out
        try:
            self._fd: int | None = self._stream.fileno()
        except (AttributeError, OSError, io.UnsupportedOperation):
            self._fd = None
        self._tty = self._fd is not None and os.isatty(self._fd)
        self._keys: TerminalKeys | None = TerminalKeys(self._stream) if self._tty else None
        self._buf = ""
        self._eof = False

    def poll(self) -> str | None:
        if self._tty:
            return None if self._keys is None else self._keys.poll()
        line = self._next_line(block=False)
        if line is None:
            return None
        text = line.strip()
        return text[0].lower() if text else ENTER

    def discard_pending(self) -> None:
        """On a TTY, drop every key typed before now (an Enter pressed while the client was busy must never answer a
        gate shown later). Pipe / file input is a script and is kept."""
        if self._tty:
            import termios

            termios.tcflush(self._fd, termios.TCIFLUSH)

    def read_line(self, prompt: str) -> str | None:
        """Show `prompt` and read the answer; on a TTY, keys typed before the prompt was shown are discarded."""
        if self._tty:
            if self._keys is not None:
                self._keys.close()  # canonical mode back first: a flush before it could miss queued bytes
            self.discard_pending()
        self._out.write(prompt)
        self._out.flush()
        if self._tty:
            try:
                line = self._stream.readline()
            finally:
                self._keys = TerminalKeys(self._stream)
            return line.rstrip("\r\n") if line else None
        line = self._next_line(block=True)
        self._out.write(("<end of input>" if line is None else line) + "\n")  # echo: scripted runs stay readable
        self._out.flush()
        return line

    def close(self) -> None:
        if self._keys is not None:
            self._keys.close()
            self._keys = None

    def _next_line(self, *, block: bool) -> str | None:
        while "\n" not in self._buf and not self._eof:
            if self._fd is None:
                data = self._stream.readline()  # in-memory stream: never blocks
            else:
                ready, _, _ = select.select([self._fd], [], [], 0.1 if block else 0)
                if not ready:
                    if block:
                        continue
                    return None
                data = os.read(self._fd, 4096).decode(errors="replace")
            if not data:
                self._eof = True
            self._buf += data
        if "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            return line
        if self._buf:  # last line without a newline
            line, self._buf = self._buf, ""
            return line
        return None


def ask_choice(console, prompt: str, choices: Sequence[str], *, on_eof: str) -> str:
    """Ask until the answer (lower-cased, stripped; "" is Enter) is one of `choices`; `on_eof` at end of input."""
    while True:
        line = console.read_line(prompt)
        if line is None:
            return on_eof
        answer = line.strip().lower()
        if answer in choices:
            return answer
        print("Please answer " + " / ".join("Enter" if c == "" else c for c in choices) + ".", flush=True)


# ----------------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------------


class UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(f"{self.prog}: error: {message}")


def _joint_arg(text: str):
    if text == "all":
        return "all"
    try:
        joint = int(text)
    except ValueError:
        joint = 0
    if not 1 <= joint <= N_JOINTS:
        raise argparse.ArgumentTypeError(f"must be 'all' or a joint number 1..{N_JOINTS}, got {text!r}")
    return [joint]


def _delta_arg(text: str) -> float:
    try:
        delta = float(text)
    except ValueError:
        delta = math.nan
    if not 0 < delta <= checks.MAX_MOTION_DELTA:
        raise argparse.ArgumentTypeError(f"must be in (0, {checks.MAX_MOTION_DELTA}] rad, got {text!r}")
    return delta


def parse_args(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> argparse.Namespace:
    env = os.environ if env is None else env
    common = _Parser(add_help=False)
    common.add_argument(
        "--site",
        default=env.get("ROBOT_SITE") or DEFAULT_SITE,
        help="site YAML (default: $ROBOT_SITE or %(default)s)",
    )
    common.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a site key (repeatable)"
    )
    common.add_argument("--run-name", default=None, help="results go to <out_dir>/<run-name> (default: a timestamp)")

    parser = _Parser(prog="python -m robot_client", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True, metavar="SUBCOMMAND")
    p = sub.add_parser("check-obs", parents=[common], help="check the robot streams (no publishers)")
    p.add_argument("--seconds", type=float, default=8.0, help="how long to listen (default %(default)s)")
    p.add_argument("--norm-stats", default=None, help="training norm_stats.json: report state outside its range")
    p = sub.add_parser("check-infer", parents=[common], help="check the policy server")
    p.add_argument("--n", type=int, default=3, help="number of requests (default %(default)s)")
    p = sub.add_parser("check-motion", parents=[common], help="move one arm's joints slightly and back")
    p.add_argument("--arm", required=True, choices=ARM_NAMES)
    p.add_argument("--joint", type=_joint_arg, default="all", help="all or a joint number 1..7 (default all)")
    p.add_argument("--delta", type=_delta_arg, default=0.05, help="move in rad (default %(default)s)")
    sub.add_parser("prepare", parents=[common], help="move both arms to the start pose (after an Enter)")
    p = sub.add_parser("run", parents=[common], help="run policy episodes")
    p.add_argument("--episodes", type=int, default=None, help="stop after N episodes (default: until q)")
    p.add_argument("--dry-run", action="store_true", help="nothing moves: preview topics only")
    p.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=None, help="override active_arms")
    p = sub.add_parser("release", parents=[common], help="deactivate the configured controller(s)")
    p.add_argument("--arm", action="append", choices=ARM_NAMES, default=None, help="arm (repeatable; default both)")
    args = parser.parse_args(argv)
    if getattr(args, "episodes", None) is not None and args.episodes < 1:
        raise UsageError(f"{parser.prog} run: error: --episodes must be >= 1, got {args.episodes}")
    if getattr(args, "n", None) is not None and args.n < 1:
        raise UsageError(f"{parser.prog} check-infer: error: --n must be >= 1, got {args.n}")
    return args


# ----------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------


class Refused(Exception):  # noqa: N818 - an operator-facing outcome, not a bug
    """A pre-condition failed; the message tells the operator what to do. Exit code 2."""


@dataclasses.dataclass
class Context:
    args: argparse.Namespace
    cfg: SiteConfig
    env: Mapping[str, str]
    guard: SignalGuard
    run_name: str
    run_dir: Path
    _console: object = None

    @property
    def console(self):
        if self._console is None:
            self._console = make_console()
        return self._console

    def close(self) -> None:
        if self._console is not None:
            self._console.close()


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    guard = SignalGuard()
    guard.install()
    ctx = None
    try:
        args = parse_args(argv, env)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        ctx = _context(args, env, guard)
        return _COMMANDS[args.command](ctx)
    except UsageError as e:
        print(e, file=sys.stderr)
        return EXIT_CONFIG
    except SystemExit as e:  # --help
        return int(e.code or 0)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    except ImportError as e:  # make_node & co. outside the client container
        print(
            f"REFUSED: ROS 2 Python packages are not importable ({e}); run the client in its container "
            "(or source /opt/ros/humble/setup.bash and the franka message workspace)",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_REFUSED
    except ControllerStillActive as e:
        print(f"CONTROLLER STILL ACTIVE: {e}", file=sys.stderr, flush=True)
        return EXIT_STILL_ACTIVE
    except KeyboardInterrupt:
        print("\nInterrupted. Everything this client activated has been released (see above).", file=sys.stderr)
        return EXIT_INTERRUPTED
    finally:
        try:
            if ctx is not None:
                ctx.close()
        finally:
            guard.restore()


def _context(args: argparse.Namespace, env: Mapping[str, str], guard: SignalGuard) -> Context:
    overrides = list(args.set)
    if getattr(args, "arms", None):
        overrides.append(f"active_arms=[{', '.join(dict.fromkeys(args.arms))}]")
    cfg = load_site_config(args.site, overrides, env)
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    if not RUN_NAME_RE.match(run_name):
        raise ConfigError([f"--run-name must match {RUN_NAME_RE.pattern} (one path component), got {run_name!r}"])
    run_dir = Path(cfg.out_dir) / run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ConfigError(
            [f"cannot create the run directory {run_dir} ({e}); mount out_dir or set --set out_dir=..."]
        ) from e
    return Context(args, cfg, env, guard, run_name, run_dir)


def _ros_env_ok(ctx: Context, *, refuse: bool) -> bool:
    problems = check_ros_env(ctx.cfg, ctx.env)
    for problem in problems:
        label = "REFUSED" if refuse else "WARNING"
        hint = (
            " (the compose file passes ROS_DOMAIN_ID through: ROS_DOMAIN_ID=<n> docker compose ... run --rm "
            "robot-client ...)"
            if problem.startswith("ROS_DOMAIN_ID")
            else ""
        )
        print(
            f"{label}: {problem}; set the environment variable{hint} or the ros section of the site YAML",
            file=sys.stderr,
        )
    return not problems


def _server_hint(ctx: Context) -> None:
    if ctx.env.get(ENV_HOST):
        return
    server = ctx.cfg.server
    bar = "*" * 100
    print(
        f"{bar}\n* {ENV_HOST} is not set: connecting to {server.host}:{server.port}.\n"
        f"* If the policy server runs on another machine, set {ENV_HOST} (and POLICY_SERVER_PORT / "
        f"POLICY_SERVER_API_KEY) and rerun.\n{bar}",
        flush=True,
    )


def _finish_check(ctx: Context, report: checks.CheckReport) -> int:
    print(report.format(), flush=True)
    try:
        path = report.save(ctx.run_dir)
        print(f"saved {path}")
    except OSError as e:
        print(f"WARNING: could not save the report to {ctx.run_dir}: {e}", file=sys.stderr)
    return EXIT_OK if report.passed else EXIT_REFUSED


# ---- check-obs / check-infer / check-motion ----


def cmd_check_obs(ctx: Context) -> int:
    cfg, args = ctx.cfg, ctx.args
    _ros_env_ok(ctx, refuse=False)
    node = make_node(cfg, "none")
    try:
        backend = None
        if cfg.spine.enabled:
            try:
                backend = make_backend(node, cfg)
            except RuntimeError as e:
                print(f"WARNING: {e}; the spine position is not checked", file=sys.stderr)
        print(f"Listening to the robot streams for {args.seconds:g} s ...", flush=True)
        report = checks.check_obs(node, cfg, args.seconds, ctx.run_dir, args.norm_stats, backend)
    finally:
        node.shutdown()
    return _finish_check(ctx, report)


def cmd_check_infer(ctx: Context) -> int:
    _server_hint(ctx)
    snapshot = _live_snapshot(ctx)
    return _finish_check(ctx, checks.check_infer(ctx.cfg, ctx.args.n, snapshot))


def _live_snapshot(ctx: Context):
    """A complete live snapshot (every camera, both arms) if the robot streams arrive within LIVE_OBS_WAIT_S, else
    None (check-infer then sends zeros)."""
    cfg = ctx.cfg
    try:
        node = make_node(cfg, "none")
    except Exception as e:  # no rclpy (dev machine) or no ROS context
        print(f"ROS not available ({type(e).__name__}: {e}); sending zero observations", flush=True)
        return None
    _ros_env_ok(ctx, refuse=False)
    try:
        deadline = time.monotonic() + LIVE_OBS_WAIT_S
        while True:
            snap = node.snapshot()
            missing = [f"cam/{n}" for n in cfg.cameras if n not in snap.images]
            missing += [f"arm/{s}" for s in cfg.arms if s not in snap.arm_q]
            if not missing:
                return snap
            if time.monotonic() >= deadline:
                print(f"No live {', '.join(missing)} after {LIVE_OBS_WAIT_S:g} s; sending zero observations")
                return None
            time.sleep(0.1)
    finally:
        node.shutdown()


def cmd_check_motion(ctx: Context) -> int:
    cfg, args = ctx.cfg, ctx.args
    side = args.arm
    if not _ros_env_ok(ctx, refuse=True):
        return EXIT_REFUSED
    node = make_node(cfg, "live")
    try:
        cms = make_controller_managers(node, cfg)
        joints = "all" if args.joint == "all" else args.joint
        print(
            f"check-motion takes over the {side} arm ({cfg.arms[side].controller}) and moves joint(s) "
            f"{'1-7' if joints == 'all' else joints[0]} by {args.delta} rad and back, one at a time "
            f"(~{2 * (checks.MOTION_RAMP_S + checks.MOTION_HOLD_S):g} s per joint).",
            flush=True,
        )
        choice = ask_choice(
            ctx.console, "Workspace clear, E-stop at hand? [Enter] start / [q] quit: ", ("", "q"), on_eof="q"
        )
        if choice == "q":
            print("Nothing was moved.")
            return EXIT_OK
        _require_live(node)
        adapters = make_adapters(cfg, (side,))
        streamer = make_streamer(cfg, node, (side,), adapters)
        session = GuardedSession(
            cfg,
            (side,),
            {side: cms[side]},
            node,
            streamer,
            adapters,
            guard=ctx.guard,
            ack=_operator_ack(ctx.console),
        )
        report = checks.check_motion(
            side, joints, args.delta, cfg=cfg, session=session, streamer=streamer, measured=node.measured
        )
    finally:
        node.shutdown()
    return _finish_check(ctx, report)


def _require_live(node) -> None:
    if getattr(node, "command_mode", None) != "live":  # takeover counts our own command publisher
        raise RuntimeError(f"takeover needs a node in 'live' mode, got {getattr(node, 'command_mode', None)!r}")


# ---- prepare / release ----


def cmd_prepare(ctx: Context) -> int:
    cfg = ctx.cfg
    if not _ros_env_ok(ctx, refuse=True):
        return EXIT_REFUSED
    node = make_node(cfg, "live")  # the gripper command publishers
    try:
        cms = make_controller_managers(node, cfg)
        backend = _backend_or_refuse(node, cfg)
        choice = ask_choice(
            ctx.console, "Workspace clear? [Enter] move both arms to the start pose / [q] quit: ", ("", "q"), on_eof="q"
        )
        if choice == "q":
            print("Nothing was moved.")
            return EXIT_OK
        report = prepare(cfg, cms, backend, streamer_active=False)
        print(report.summary(), flush=True)
        if not report.ok:
            raise Refused(f"prepare stopped at '{report.failed_step}' (see above); fix the cause and rerun")
        return EXIT_OK
    finally:
        node.shutdown()


def _backend_or_refuse(node, cfg: SiteConfig):
    try:
        return make_backend(node, cfg)
    except RuntimeError as e:
        raise Refused(str(e)) from e


def cmd_release(ctx: Context) -> int:
    cfg = ctx.cfg
    _ros_env_ok(ctx, refuse=False)
    sides = list(dict.fromkeys(ctx.args.arm or cfg.arms))
    node = make_node(cfg, "none")
    try:
        cms = make_controller_managers(node, cfg)
        with ctx.guard.releasing():
            failed = [side for side in sides if not _release_side(cfg, side, cms[side])]
    finally:
        node.shutdown()
    if failed:
        raise Refused(f"release failed for {failed} (see above); use the robot's own tooling or the E-stop")
    return EXIT_OK


def _release_side(cfg: SiteConfig, side: str, cm) -> bool:
    arm = cfg.arms[side]
    where = f"{arm.controller} on {arm.controller_manager}"
    try:
        state = cm.list_controllers().get(arm.controller)
    except Exception as e:
        print(f"{side}: cannot read the state of {where}: {e}", file=sys.stderr)
        return False
    if state != "active":
        print(f"{side}: {where} is {state or 'not loaded'}; nothing to deactivate")
        return True
    try:
        ok = cm.switch([], [arm.controller], timeout_s=cfg.takeover.service_timeout_s)
        state = cm.list_controllers().get(arm.controller)
    except Exception as e:
        print(f"{side}: deactivating {where} failed: {e}", file=sys.stderr)
        return False
    if not ok or state == "active":
        print(f"{side}: switch_controller did not deactivate {where} (state: {state})", file=sys.stderr)
        return False
    print(f"{side}: {where} deactivated (now {state or 'not loaded'})")
    return True


# ---- run ----


def _is_layout_error(error: BaseException | None) -> bool:
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, LayoutMismatchError):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


def _layout_message(error: BaseException) -> str:
    return (
        f"the policy server is not the munich16 policy this client needs: {error}. Serve a munich16 checkpoint "
        "(POLICY_CONFIG) and rerun"
    )


def _unreachable_message(error: BaseException, cfg: SiteConfig) -> str:
    return (
        f"{error}. Is the policy server running and reachable at {cfg.server.host}:{cfg.server.port}? "
        "Set POLICY_SERVER_HOST / POLICY_SERVER_PORT"
    )


def _start_worker(cfg: SiteConfig) -> tuple[InferenceWorker, dict]:
    """Connect (metadata validated against munich16) and start the inference worker; Refused if impossible."""
    try:
        client = make_policy_client(cfg)
    except LayoutMismatchError as e:
        raise Refused(_layout_message(e)) from e
    except PolicyUnreachableError as e:
        raise Refused(_unreachable_message(e, cfg)) from e
    metadata = client.metadata()
    worker = InferenceWorker(client)
    worker.start()
    return worker, metadata


def _no_gripper(side: str, width: float) -> None:
    """Dry run: gripper commands are dropped."""


class _RunCommand:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.console = ctx.console
        self.dry = bool(ctx.args.dry_run)
        self.sides = session_sides(self.cfg, self.cfg.active_arms)
        self.required = required_stream_keys(self.cfg)
        self.worker: InferenceWorker | None = None
        self.recorder: RunRecorder | None = None
        self.node = None
        self.cms = None
        self.backend = None
        self._ep = None

    def execute(self) -> int:
        if not _ros_env_ok(self.ctx, refuse=True):
            return EXIT_REFUSED
        _server_hint(self.ctx)
        try:
            self.worker, metadata = _start_worker(self.cfg)
            print(f"Policy server OK: {metadata.get('layout')} / {metadata.get('config')}", flush=True)
            self.recorder = RunRecorder(Path(self.cfg.out_dir), self.ctx.run_name, self.cfg.redacted_dict(), metadata)
            self._event("run_start", dry_run=self.dry, sides=list(self.sides), active_arms=list(self.cfg.active_arms))
            self.node = make_node(self.cfg, "preview" if self.dry else "live")
            if not self.dry:
                _require_live(self.node)
                self.cms = make_controller_managers(self.node, self.cfg)
                self.backend = _backend_or_refuse(self.node, self.cfg)
            elif self.node.command_mode != "preview":
                raise RuntimeError(f"a dry run needs a node in 'preview' mode, got {self.node.command_mode!r}")
            return self._episodes()
        except Refused as e:
            self._event("refused", message=str(e))
            raise
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        if self.worker is not None:
            self.worker.stop(timeout=1.0)
        if self.node is not None:
            self.node.shutdown()
        if self.recorder is not None:
            self._event("run_end")
            self.recorder.close()

    def _event(self, kind: str, **fields) -> None:
        if self.recorder is not None:
            self.recorder.event(kind, **fields)

    def _episodes(self) -> int:
        limit = self.ctx.args.episodes
        done = 0
        while limit is None or done < limit:
            idx = self._next_index()
            status = self._dry_episode(idx) if self.dry else self._live_episode(idx)
            if status == "quit":
                break
            if status == "freeze":
                return EXIT_FREEZE
            done += 1
        print(f"Run finished: {done} episode(s) in {self.recorder.dir}", flush=True)
        return EXIT_OK

    def _next_index(self) -> int:
        numbers = [int(p.name[8:]) for p in self.recorder.dir.glob("episode_*") if p.name[8:].isdigit()]
        return max(numbers, default=-1) + 1

    # ---- one episode ----

    def _live_episode(self, idx: int) -> str:
        cfg = self.cfg
        print(f"\n=== episode {idx} ===", flush=True)
        choice = ask_choice(self.console, RESET_PROMPT, ("", "k", "q"), on_eof="q")
        if choice == "q":
            return "quit"
        if choice == "":
            report = prepare(cfg, self.cms, self.backend, streamer_active=False)
            print(report.summary(), flush=True)
            self._event("prepare", episode=idx, ok=report.ok, steps=[dataclasses.asdict(s) for s in report.steps])
            if not report.ok:
                raise Refused(f"prepare stopped at '{report.failed_step}' (see above); fix the cause and rerun")
        self._ensure_worker()
        adapters = make_adapters(cfg, self.sides)
        streamer = make_streamer(cfg, self.node, self.sides, adapters)
        cms = {side: self.cms[side] for side in self.sides}
        session = GuardedSession(
            cfg,
            self.sides,
            cms,
            self.node,
            streamer,
            adapters,
            recorder=self.recorder,
            guard=self.ctx.guard,
            ack=_operator_ack(self.console),
        )
        self._ep = None
        try:
            with session:
                result = self._gate_and_run(idx, streamer)
        except TakeoverError as e:
            raise Refused(f"takeover refused: {e}") from e
        except KeyboardInterrupt:
            self._abandon_episode(streamer)
            raise
        except ControllerStillActive:
            self._abandon_episode(streamer, "release: a controller could not be confirmed inactive")
            raise
        print("Released.", flush=True)
        return self._conclude(idx, result, streamer)

    def _dry_episode(self, idx: int) -> str:
        print(f"\n=== dry-run episode {idx} (nothing moves; preview topics only) ===", flush=True)
        self._ensure_worker()
        self._wait_discovery()
        adapters = make_adapters(self.cfg, self.sides)
        streamer = make_streamer(self.cfg, self.node, self.sides, adapters, preview=True)
        hold = self._measured_hold()
        try:
            streamer.start(hold)
        except ValueError as e:
            raise Refused(f"refusing to stream hold = measured: {e}") from e
        self._ep = None
        try:
            self._wait_first_hold(streamer)
            result = self._gate_and_run(idx, streamer)
        except KeyboardInterrupt:
            self._abandon_episode(streamer)
            raise
        finally:
            streamer.stop()
        return self._conclude(idx, result, streamer)

    def _wait_discovery(self) -> None:
        """As a Session does before a live episode: let `takeover.discovery_wait_s` pass since the node started, so
        the start gate does not read a camera whose first frame is still on its way as a dead stream."""
        wait = self.cfg.takeover.discovery_wait_s - self.node.uptime_s()
        if wait > 0:
            time.sleep(wait)

    def _wait_first_hold(self, streamer: Streamer) -> None:
        """Until every side has published its first hold command: the start gate reads "not published yet" as a
        stopped stream. Bounded; a stream that never starts is then reported by the gate."""
        deadline = time.monotonic() + FIRST_HOLD_WAIT_S
        while streamer.fault is None and time.monotonic() < deadline:
            if all(streamer.cycles_since_start(side) > 0 for side in self.sides):
                return
            time.sleep(0.005)

    def _measured_hold(self) -> dict:
        deadline = time.monotonic() + self.cfg.takeover.discovery_wait_s
        while True:
            hold = {side: self.node.measured(side) for side in self.sides}
            missing = [side for side, q in hold.items() if q is None]
            if not missing:
                return hold
            if time.monotonic() >= deadline:
                topics = ", ".join(self.cfg.arms[side].measured_topic for side in missing)
                raise Refused(
                    f"no measured joint states for {missing} ({topics}); is the robot state broadcaster running?"
                )
            time.sleep(GATE_POLL_S)

    def _ensure_worker(self) -> None:
        if self.worker is not None and _is_layout_error(self.worker.error):
            raise Refused(_layout_message(self.worker.error))
        try:
            fresh = ensure_worker(self.worker, self.cfg)
        except LayoutMismatchError as e:
            raise Refused(_layout_message(e)) from e
        except PolicyUnreachableError as e:
            raise Refused(_unreachable_message(e, self.cfg)) from e
        if fresh is not self.worker:
            self._event("worker_restarted", previous_error=repr(getattr(self.worker, "error", None)))
        self.worker = fresh

    def _gate_and_run(self, idx: int, streamer: Streamer) -> tuple[str, object]:
        verdict, why = self._wait_start(streamer)
        if verdict != "start":
            return verdict, why
        self._ep = self.recorder.episode(idx)
        generation = idx + 1  # distinct per episode: results of an earlier episode are never accepted
        self._event("episode_start", episode=idx, generation=generation, dry_run=self.dry)
        grippers = _no_gripper if self.dry else self.node.publish_gripper
        runner = Runner(self.cfg, self.node, self.worker, streamer, grippers, self._ep, self.console)
        print("Policy running: p pause, Enter resume, q end the episode, Ctrl-C emergency release.", flush=True)
        return "ran", runner.run_episode(generation=generation)

    def _wait_start(self, streamer: Streamer) -> tuple[str, str]:
        """Poll the operator's key while the hold streams; the stream health and the safety evaluation are checked
        every GATE_POLL_S. Returns ("start" | "quit" | "freeze", reason). Keys typed before the prompt (e.g. an Enter
        pressed during prepare or takeover) are discarded first."""
        self.console.discard_pending()
        print(START_PROMPT, flush=True)
        shown = None
        while True:
            key = self.console.poll()
            if not streamer.healthy():
                return "freeze", f"command streamer unhealthy: {streamer.fault or 'publishing stopped'}"
            snap = self.node.snapshot()
            ages = {k: snap.ages.get(k, math.inf) for k in self.required}
            worker = self.worker
            fatal = worker.error is not None or not worker.is_alive()
            mode, why = safety.evaluate(ages, 0.0, fatal, self.cfg.safety)
            if mode is safety.Mode.FREEZE:
                if fatal:
                    why += f" ({worker.error or 'inference worker thread is not running'})"
                return "freeze", why
            if key == "q":
                return "quit", "operator pressed q at the start prompt"
            if mode is safety.Mode.HOLD and why != shown:
                print(f"Waiting for the streams: {why}", flush=True)
                shown = why
            elif mode is safety.Mode.RUN:
                shown = None
            if key == ENTER:
                if mode is safety.Mode.RUN:
                    return "start", ""
                print(f"Not starting: {why}", flush=True)
            time.sleep(GATE_POLL_S)

    def _conclude(self, idx: int, result: tuple[str, object], streamer: Streamer) -> str:
        kind, payload = result
        if kind == "quit":
            self._event("quit", episode=idx, reason=payload)
            print("Quit.", flush=True)
            return "quit"
        if kind == "freeze":
            self._event("freeze_before_start", episode=idx, reason=payload)
            print(f"FREEZE before the start: {payload}", file=sys.stderr, flush=True)
            if _is_layout_error(self.worker.error):
                raise Refused(_layout_message(self.worker.error))
            return "freeze"
        outcome = payload
        stats = {**outcome.stats, "ended_by": outcome.ended_by, "reason": outcome.reason, "dry_run": self.dry}
        stream_stats = self._stream_stats(streamer)
        print(f"Episode {idx} ended by {outcome.ended_by} after {outcome.steps} steps: {outcome.reason}", flush=True)
        self._event("episode_end", episode=idx, ended_by=outcome.ended_by, reason=outcome.reason, steps=outcome.steps)
        if outcome.ended_by == "freeze":
            print(f"FREEZE: {outcome.reason}", file=sys.stderr, flush=True)
            note = self.console.read_line(NOTE_PROMPT) or ""
            self._ep.finish("freeze", note, stats, stream_stats)
            if _is_layout_error(self.worker.error):
                raise Refused(_layout_message(self.worker.error))
            return "freeze"
        choice = ask_choice(self.console, OUTCOME_PROMPT, ("s", "f", "x"), on_eof="x")
        note = self.console.read_line(NOTE_PROMPT) or ""
        self._ep.finish(choice, note, stats, stream_stats)
        self._event("outcome", episode=idx, outcome=choice)
        return "done"

    def _abandon_episode(self, streamer: Streamer, note: str = "interrupted (Ctrl-C / SIGTERM / SIGHUP)") -> None:
        """Ctrl-C mid-episode (or a release that could not confirm a controller inactive): close the episode's
        records as a freeze (the release already ran)."""
        if self._ep is not None:
            self._ep.finish("freeze", note, {"dry_run": self.dry}, self._stream_stats(streamer))
        self._event("interrupted", note=note)

    def _stream_stats(self, streamer: Streamer) -> dict:
        try:
            node_stats = self.node.stream_stats()
        except Exception as e:  # diagnostics only
            node_stats = {"error": repr(e)}
        return {
            "node": node_stats,
            "streamer": {
                "cycles": streamer.cycles,
                "max_gap_s": dict(streamer.max_gap_s),
                "leash_events": streamer.leash_events,
                "limit_clamps": streamer.limit_clamps,
                "hard_clamps": streamer.hard_clamps,
                "fault": streamer.fault,
            },
        }


def cmd_run(ctx: Context) -> int:
    return _RunCommand(ctx).execute()


_COMMANDS = {
    "check-obs": cmd_check_obs,
    "check-infer": cmd_check_infer,
    "check-motion": cmd_check_motion,
    "prepare": cmd_prepare,
    "run": cmd_run,
    "release": cmd_release,
}
