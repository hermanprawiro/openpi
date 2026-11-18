"""The operator's pre-flight checks (spec §6.7): `check_obs` (streams), `check_infer` (policy server), `check_motion`
(one arm follows small joint moves; disambiguates `command_mode` / `joint_signs`).

Each returns a `CheckReport`: rows of `status item detail` (status `ok`, `warn`, `FAIL` or `info`; any `FAIL` fails
the check), printed as a plain table and saved as `<run_dir>/<name>.json`. No ROS imports: the node, the motion
backend and the Session are passed in.
"""

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import json
from pathlib import Path
import time
from typing import Literal

from eval_client.chunking import AsyncChunkBuffer
import numpy as np
from PIL import Image

from robot_client import layout
from robot_client.config import N_JOINTS
from robot_client.config import SiteConfig
from robot_client.layout import LayoutMismatchError
from robot_client.policy_link import PolicyUnreachableError
from robot_client.policy_link import make_policy_client
from robot_client.preprocess import IMAGE_KEYS
from robot_client.preprocess import POLICY_IMAGE_SIZE
from robot_client.preprocess import Snapshot
from robot_client.preprocess import StateError
from robot_client.preprocess import build_observation
from robot_client.preprocess import build_state
from robot_client.preprocess import state_range_report
from robot_client.preprocess import to_policy_image
from robot_client.takeover import TakeoverError

OK, WARN, FAIL, INFO = "ok", "warn", "FAIL", "info"

# check_obs: expected stream rates; a stream below RATE_WARN_FRACTION of its rate is a warning.
EXPECTED_CAMERA_HZ = {"head": 20.0, "wrist_left": 30.0, "wrist_right": 30.0}
MIN_ARM_HZ = 100.0
RATE_WARN_FRACTION = 0.8

# check_motion: per joint, ramp to +delta over MOTION_RAMP_S, hold MOTION_HOLD_S, measure, ramp back, hold again.
MOTION_RAMP_S = 1.0
MOTION_HOLD_S = 1.0
MAX_MOTION_DELTA = 0.2  # rad; check_motion is a small-motion probe
ABORT_SLACK_RAD = 0.02  # abort when any joint is more than 2 * delta + this from where the check started
NO_MOTION_RATIO = 0.2  # |measured / commanded| below this: the joint did not move


@dataclasses.dataclass
class CheckReport:
    name: str
    passed: bool
    lines: list[str]
    data: dict

    def format(self) -> str:
        warnings = sum(1 for line in self.lines if line.startswith(WARN))
        head = f"{self.name}: {'PASS' if self.passed else 'FAIL'}"
        if warnings:
            head += f" ({warnings} warning{'s' if warnings > 1 else ''})"
        return "\n".join([head, *(f"  {line}" for line in self.lines)])

    def save(self, run_dir: Path | str) -> Path:
        path = Path(run_dir) / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "passed": self.passed, "lines": self.lines, "data": self.data}
        path.write_text(json.dumps(payload, indent=1, default=_json_default) + "\n", encoding="utf-8")
        return path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


class _Rows:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, item: str, detail: str) -> None:
        self.rows.append((status, item, detail))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)

    def lines(self) -> list[str]:
        width = max((len(item) for _, item, _ in self.rows), default=0)
        return [f"{status:<4}  {item:<{width}}  {detail}" for status, item, detail in self.rows]

    def report(self, name: str, data: dict) -> CheckReport:
        return CheckReport(name, not self.failed, self.lines(), data)


def _fmt_q(q) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in np.asarray(q, dtype=float)) + "]"


# ----------------------------------------------------------------------------------------------------
# check_obs
# ----------------------------------------------------------------------------------------------------


def check_obs(
    node,
    cfg: SiteConfig,
    seconds: float = 8.0,
    out_dir: Path | str | None = None,
    norm_stats_path: str | None = None,
    backend=None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> CheckReport:
    """Listen for `seconds`, then judge every configured stream from `node.stream_stats()` / `node.snapshot()`.

    Fails on a missing required stream (or one that stopped), a missing joint, an undecodable stream (unsupported
    encoding) or an image / state that cannot be preprocessed. Rates below 80 % of the expected rate, undecodable
    messages among good ones, state outside the training range (`norm_stats_path`) and a spine far from
    `spine.height` (`backend.spine_position`) are warnings. One preprocessed JPEG per camera goes to `out_dir`.
    """
    sleep(seconds)
    stats = node.stream_stats()
    snap = node.snapshot()
    rows = _Rows()
    fields = ("topic", "count", "rate_hz", "max_gap_s", "age_s", "decode_errors", "last_error", "raw")
    streams = {key: {k: st.get(k) for k in fields} for key, st in stats.items() if key != "commands"}
    data: dict = {"seconds": seconds, "streams": streams, "images": {}}
    out = None if out_dir is None else Path(out_dir)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    def usable(key: str, topic: str, *, required: bool, expected_hz: float | None = None, note: str = ""):
        return _judge_stream(
            rows, cfg, key, topic, stats.get(key), seconds, required=required, expected_hz=expected_hz, note=note
        )

    for name, cam in cfg.cameras.items():
        key = f"cam/{name}"
        st = usable(key, cam.topic, required=True, expected_hz=EXPECTED_CAMERA_HZ.get(name))
        if st is None:
            continue
        frame = snap.images.get(name)
        if frame is None:
            rows.add(FAIL, key, f"no image in the snapshot although {st['topic']} delivers")
            continue
        try:
            policy_image = to_policy_image(frame, cam.train_res, cam.crop)
        except ValueError as e:
            rows.add(FAIL, key, f"cannot preprocess {_raw_desc(st['raw'], frame)}: {e} (cameras.{name}.crop?)")
            continue
        shape = "x".join(str(v) for v in policy_image.shape)
        steps = []
        if cam.crop is not None:
            steps.append(f"crop {list(cam.crop)}")
        if cam.train_res is not None:
            live = (frame.shape[1], frame.shape[0]) if cam.crop is None else (cam.crop[2], cam.crop[3])
            if live != tuple(cam.train_res):
                steps.append(f"resize to {cam.train_res[0]}x{cam.train_res[1]}")
        steps.append(shape)
        detail = f"{_raw_desc(st['raw'], frame)} -> {' -> '.join(steps)}, {_rate(st)}"
        if policy_image.shape != (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE, 3):
            rows.add(FAIL, key, f"{detail}: expected {POLICY_IMAGE_SIZE}x{POLICY_IMAGE_SIZE}x3")
        else:
            rows.add(OK, key, detail)
        if out is not None:
            path = out / f"check_obs_{name}.jpg"
            try:
                Image.fromarray(policy_image).save(path, quality=90)
                data["images"][name] = str(path)
            except OSError as e:
                rows.add(WARN, key, f"could not save {path}: {e}")

    arms_ok = True
    for side, arm in cfg.arms.items():
        key = f"arm/{side}"
        if _missing_joints(rows, key, stats.get(key), arm.joint_names, f"arms.{side}.joint_names"):
            arms_ok = False
            continue
        st = usable(key, arm.measured_topic, required=True, expected_hz=MIN_ARM_HZ)
        q = snap.arm_q.get(side)
        if st is None or q is None:
            arms_ok = False
            if st is not None:
                rows.add(FAIL, key, f"no joint positions in the snapshot although {st['topic']} delivers")
            continue
        rows.add(OK, key, f"{len(arm.joint_names)} joints matched, {_rate(st)}, q = {_fmt_q(q)}")

    for side, arm in cfg.arms.items():
        grip = arm.gripper
        key = f"grip/{side}"
        optional_note = "" if grip.required else _optional_note(cfg, side)
        status = FAIL if grip.required else WARN
        if _missing_joints(rows, key, stats.get(key), (grip.joint,), f"arms.{side}.gripper.joint", status=status):
            continue
        st = usable(key, grip.state_topic, required=grip.required, note=optional_note)
        if st is None:
            continue
        value = snap.knuckle.get(side)
        if value is None:
            rows.add(FAIL if grip.required else WARN, key, f"no gripper value in the snapshot{optional_note}")
            continue
        rows.add(OK, key, f"{grip.joint} = {value:.4f}, {_rate(st)}")

    data["state"] = None
    data["state_range"] = []
    if arms_ok:
        try:
            state = build_state(snap, cfg.state.pin)
        except StateError as e:
            rows.add(FAIL, "state", str(e))
        else:
            data["state"] = state.tolist()
            pinned = f", pinned {dict(cfg.state.pin)}" if cfg.state.pin else ""
            rows.add(OK, "state", f"{layout.STATE_DIM} dims{pinned}")
            _check_state_range(rows, data, state, norm_stats_path)
    else:
        rows.add(INFO, "state", "not built (an arm stream is missing)")

    data["spine"] = _check_spine(rows, cfg, backend)
    return rows.report("check_obs", data)


def _judge_stream(
    rows: _Rows,
    cfg: SiteConfig,
    key: str,
    topic: str,
    st: dict | None,
    seconds: float,
    *,
    required: bool,
    expected_hz: float | None,
    note: str,
) -> dict | None:
    """Add the row(s) for a stream that is missing, undecodable, stopped or slow; return `st` if it is usable.
    A missing optional stream is `info`, an unusable one `warn`; `note` is appended to those rows."""
    missing = FAIL if required else INFO
    optional = "" if required else f" (optional{note})"
    if st is None or not st.get("count"):
        rows.add(missing, key, f"no messages on {topic} in {seconds:g} s{optional}")
        return None
    if st.get("age_s") is None:
        status = FAIL if required else WARN
        rows.add(status, key, f"{st['count']} message(s) on {topic}, none usable: {st.get('last_error')}{optional}")
        return None
    if st["age_s"] > cfg.safety.obs_stale_hold_s:
        rows.add(
            FAIL if required else WARN,
            key,
            f"last usable sample on {topic} is {st['age_s']:.1f} s old (> obs_stale_hold_s "
            f"{cfg.safety.obs_stale_hold_s}): the stream stopped{optional}",
        )
        return None
    if st.get("decode_errors"):
        rows.add(WARN, key, f"{st['decode_errors']} undecodable message(s) on {topic}: {st.get('last_error')}")
    rate = st.get("rate_hz")
    if expected_hz is not None and (rate is None or rate < RATE_WARN_FRACTION * expected_hz):
        shown = "unknown" if rate is None else f"{rate:.1f} Hz"
        rows.add(WARN, key, f"rate {shown} on {topic} is below 80 % of the expected {expected_hz:g} Hz")
    return st


def _missing_joints(
    rows: _Rows, key: str, st: dict | None, wanted: Sequence[str], knob: str, status: str = FAIL
) -> bool:
    names = (st or {}).get("raw", {}).get("names")
    if not st or not st.get("count") or names is None:
        return False
    missing = [n for n in wanted if n not in names]
    if missing:
        rows.add(status, key, f"joint(s) {missing} not on {st['topic']} (names seen: {list(names)}); check {knob}")
    return bool(missing)


def _optional_note(cfg: SiteConfig, side: str) -> str:
    dim = layout.GRIPPER_INDEX[side]
    if dim in cfg.state.pin:
        return f"; state dim {dim} is pinned to {cfg.state.pin[dim]}"
    return f"; but state dim {dim} is NOT pinned: set state.pin.{dim}"


def _raw_desc(raw: dict, frame: np.ndarray | None) -> str:
    if "encoding" in raw:
        return f"{raw.get('width')}x{raw.get('height')} {raw.get('encoding')}"
    if "format" in raw:
        size = "" if frame is None else f" {frame.shape[1]}x{frame.shape[0]}"
        return f"compressed {raw.get('format')}{size}"
    return "?" if frame is None else f"{frame.shape[1]}x{frame.shape[0]}"


def _rate(st: dict) -> str:
    rate = st.get("rate_hz")
    return "rate unknown" if rate is None else f"{rate:.1f} Hz"


def _check_state_range(rows: _Rows, data: dict, state: np.ndarray, norm_stats_path: str | None) -> None:
    if norm_stats_path is None:
        rows.add(INFO, "state range", "not checked (no norm stats given)")
        return
    try:
        norm_stats = json.loads(Path(norm_stats_path).read_text(encoding="utf-8"))
        report = state_range_report(state, norm_stats)
    except (OSError, ValueError, KeyError, TypeError) as e:
        rows.add(WARN, "state range", f"cannot check against {norm_stats_path}: {type(e).__name__}: {e}")
        return
    data["state_range"] = report
    if not report:
        rows.add(OK, "state range", "every dim inside the training range")
    for line in report:
        rows.add(WARN, "state range", line)


def _check_spine(rows: _Rows, cfg: SiteConfig, backend) -> dict:
    sp = cfg.spine
    if not sp.enabled:
        rows.add(INFO, "spine", "not checked (spine.enabled: false)")
        return {"enabled": False}
    if backend is None:
        rows.add(INFO, "spine", "position not checked (no motion backend)")
        return {"enabled": True, "position": None}
    try:
        position = backend.spine_position(timeout_s=cfg.takeover.service_timeout_s)
    except Exception as e:  # diagnostics only
        position = None
        rows.add(WARN, "spine", f"reading the position failed: {type(e).__name__}: {e}")
        return {"enabled": True, "position": None}
    if position is None:
        rows.add(WARN, "spine", f"position unavailable ({sp.get_position_service}); prepare would refuse")
        return {"enabled": True, "position": None}
    delta = abs(sp.height - position)
    detail = f"at {position:.3f} m, spine.height {sp.height} m (delta {delta:.3f} m)"
    if delta > sp.max_delta_m:
        rows.add(
            WARN,
            "spine",
            f"{detail} > spine.max_delta_m {sp.max_delta_m}: prepare would refuse; check the height and its unit",
        )
    else:
        rows.add(OK, "spine", detail)
    return {"enabled": True, "position": position, "height": sp.height, "delta": delta}


# ----------------------------------------------------------------------------------------------------
# check_infer
# ----------------------------------------------------------------------------------------------------


def zero_observation(prompt: str) -> dict:
    """An all-zero policy observation (black images, zero state)."""
    obs = {key: np.zeros((POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE, 3), dtype=np.uint8) for key in IMAGE_KEYS.values()}
    obs["observation/state"] = np.zeros(layout.STATE_DIM, dtype=np.float32)
    obs["prompt"] = prompt
    return obs


def rtt_budget(cfg: SiteConfig) -> tuple[float, str]:
    """(seconds, explanation) a request may take before it costs motion: sync waits once per execute_horizon steps;
    async must answer before the queue drains from the trigger length."""
    hz = cfg.policy.control_hz
    inf = cfg.inference
    if inf.mode == "async":
        a = inf.async_cfg
        trigger_len = AsyncChunkBuffer(
            inf.execute_horizon,
            a.trigger_threshold,
            a.merge,
            a.blend_steps,
            action_dim=layout.ACTION_DIM,
            arm_dims=layout.ARM_DIMS,
            blend_dims=layout.ARM_DIMS,
        ).trigger_len
        return trigger_len / hz, f"async trigger_len {trigger_len} / control_hz {hz}"
    return inf.execute_horizon / hz, f"sync execute_horizon {inf.execute_horizon} / control_hz {hz}"


def check_infer(
    cfg: SiteConfig,
    n: int = 3,
    snapshot: Snapshot | None = None,
    *,
    client_factory=make_policy_client,
    clock: Callable[[], float] = time.monotonic,
) -> CheckReport:
    """Connect (timeouts, metadata validated against munich16), send `n` requests built from `snapshot` (zeros if
    None) and report the chunk shape and the round-trip latency against the policy-step budget."""
    rows = _Rows()
    server = f"{cfg.server.host}:{cfg.server.port}"
    data: dict = {"server": server, "n": n, "metadata": None, "chunk_shape": None, "rtt_ms": []}
    try:
        client = client_factory(cfg)
    except LayoutMismatchError as e:
        rows.add(FAIL, "server", f"{server} refused: {e}")
        return rows.report("check_infer", data)
    except PolicyUnreachableError as e:
        rows.add(FAIL, "server", str(e))
        return rows.report("check_infer", data)
    try:
        meta = client.metadata()
        data["metadata"] = meta
        rows.add(
            OK,
            "server",
            f"{server}: layout {meta.get('layout')}, config {meta.get('config')}, "
            f"checkpoint {meta.get('checkpoint', '?')}",
        )
        obs = _infer_input(rows, cfg, snapshot)
        shape = None
        server_ms = []
        for i in range(n):
            t0 = clock()
            try:
                result = client.infer(obs)
            except Exception as e:
                rows.add(FAIL, "request", f"{i + 1}/{n} failed: {type(e).__name__}: {e}")
                break
            data["rtt_ms"].append((clock() - t0) * 1000.0)
            try:
                chunk = layout.validate_chunk(result.get("actions"))
            except ValueError as e:
                rows.add(FAIL, "chunk", f"request {i + 1}: {e}")
                break
            shape = chunk.shape
            timing = (result.get("server_timing") or {}).get("infer_ms")
            if timing is not None:
                server_ms.append(float(timing))
        if shape is not None:
            data["chunk_shape"] = list(shape)
            if shape[0] < cfg.inference.execute_horizon:
                rows.add(
                    WARN,
                    "chunk",
                    f"shape {shape}: fewer rows than inference.execute_horizon {cfg.inference.execute_horizon}",
                )
            else:
                rows.add(OK, "chunk", f"shape {shape} (horizon x {layout.ACTION_DIM} munich16 actions)")
        _latency_rows(rows, cfg, data, server_ms)
    finally:
        _close_client(client)
    return rows.report("check_infer", data)


def _infer_input(rows: _Rows, cfg: SiteConfig, snapshot: Snapshot | None) -> dict:
    if snapshot is not None:
        try:
            obs = build_observation(snapshot, cfg.cameras, cfg.policy.prompt, pin=cfg.state.pin)
        except ValueError as e:
            rows.add(WARN, "observation", f"live snapshot unusable ({e}); sending zeros")
        else:
            rows.add(INFO, "observation", "live snapshot from the robot streams")
            return obs
    else:
        rows.add(INFO, "observation", "zeros (no live observation)")
    return zero_observation(cfg.policy.prompt)


def _latency_rows(rows: _Rows, cfg: SiteConfig, data: dict, server_ms: list[float]) -> None:
    rtts = data["rtt_ms"]
    if not rtts:
        return
    mean, peak = float(np.mean(rtts)), float(np.max(rtts))
    data["rtt_mean_ms"], data["rtt_max_ms"] = mean, peak
    budget_s, why = rtt_budget(cfg)
    data["rtt_budget_ms"] = budget_s * 1000.0
    detail = f"round trip mean {mean:.0f} ms, max {peak:.0f} ms over {len(rtts)} request(s)"
    if server_ms:
        detail += f" (server infer mean {np.mean(server_ms):.0f} ms)"
    rows.add(OK, "latency", detail)
    if peak > budget_s * 1000.0:
        rows.add(
            WARN,
            "latency",
            f"max round trip {peak:.0f} ms exceeds the budget {budget_s * 1000.0:.0f} ms ({why}): the arm will wait "
            "for the policy",
        )


def _close_client(client) -> None:
    connection = getattr(client, "_client", None)  # PolicyClient keeps no close(); its connection has one
    close = getattr(connection, "close", None)
    if close is not None:
        with contextlib.suppress(Exception):  # best effort
            close()


# ----------------------------------------------------------------------------------------------------
# check_motion
# ----------------------------------------------------------------------------------------------------

Verdict = Literal["follows", "reversed", "no-motion", "unclear"]


def motion_verdict(commanded: float, moved: float) -> Verdict:
    """How a joint answered a commanded move: ratio moved / commanded in [0.5, 1.5] follows, in [-1.5, -0.5]
    reversed, |ratio| < 0.2 no-motion, anything else unclear."""
    if commanded == 0:
        raise ValueError("commanded move must be non-zero")
    ratio = moved / commanded
    if 0.5 <= ratio <= 1.5:
        return "follows"
    if -1.5 <= ratio <= -0.5:
        return "reversed"
    if abs(ratio) < NO_MOTION_RATIO:
        return "no-motion"
    return "unclear"


class _MotionAbort(Exception):  # noqa: N818 - control flow inside the Session
    pass


def check_motion(
    side: str,
    joints: Sequence[int] | Literal["all"] = "all",
    delta: float = 0.05,
    *,
    cfg: SiteConfig,
    session,
    streamer,
    measured: Callable[[str], np.ndarray | None],
    sleep: Callable[[float], None] = time.sleep,
) -> CheckReport:
    """Under `session` (a not-yet-entered Session over `side` only, streaming through `streamer`), move each joint
    (1-based numbers) by `delta` rad and back and judge how the measured joint follows.

    Per joint: ramp to base + delta (or - delta when the joint has no room towards its upper limit) in waypoints at
    `policy.control_hz` over MOTION_RAMP_S, hold MOTION_HOLD_S, measure, ramp back, hold. Any joint more than
    2 * delta + 0.02 rad from the start, a missing measurement or an unhealthy stream aborts: the Session releases.
    Passes only if every tested joint follows. Prints (in the report) the `joint_signs` line to paste.
    """
    numbers = list(range(1, N_JOINTS + 1)) if joints == "all" else [int(j) for j in joints]
    bad = [j for j in numbers if not 1 <= j <= N_JOINTS]
    if bad or not numbers:
        raise ValueError(f"joints must be 'all' or numbers in 1..{N_JOINTS}, got {joints!r}")
    if not 0 < delta <= MAX_MOTION_DELTA:
        raise ValueError(f"delta must be in (0, {MAX_MOTION_DELTA}] rad, got {delta}")
    rows = _Rows()
    name = f"check_motion_{side}"
    data: dict = {"side": side, "delta": delta, "command_mode": cfg.command_mode, "joints": []}
    q_min, q_max = (np.asarray(x, dtype=float) for x in cfg.safety.joint_limits[side])
    margin = cfg.safety.q_margin_rad
    hz = cfg.policy.control_hz
    abort_rad = 2.0 * delta + ABORT_SLACK_RAD

    def read() -> np.ndarray:
        q = measured(side)
        if q is None:
            raise _MotionAbort(f"no measured joint states for {side}")
        return np.asarray(q, dtype=float)

    def tick(base: np.ndarray) -> None:
        sleep(1.0 / hz)
        if not streamer.healthy():
            raise _MotionAbort(f"the command stream is unhealthy ({getattr(streamer, 'fault', None) or 'no publish'})")
        q = read()
        dev = np.abs(q - base)
        worst = int(np.argmax(dev))
        if not dev[worst] <= abort_rad:  # NaN aborts too
            raise _MotionAbort(
                f"joint {worst + 1} is {dev[worst]:.3f} rad from where the check started (> 2 * delta + "
                f"{ABORT_SLACK_RAD} = {abort_rad:.3f} rad)"
            )

    def ramp(base: np.ndarray, j: int, start: float, end: float) -> None:
        steps = max(1, round(MOTION_RAMP_S * hz))
        for k in range(1, steps + 1):
            q = base.copy()
            q[j] += start + (end - start) * k / steps
            streamer.set_waypoint(side, q)
            tick(base)

    def hold(base: np.ndarray) -> None:
        for _ in range(max(1, round(MOTION_HOLD_S * hz))):
            tick(base)

    try:
        with session:
            base = read()
            data["start"] = base.tolist()
            for number in numbers:
                j = number - 1
                commanded = delta if base[j] + delta <= q_max[j] - margin else -delta
                if base[j] + commanded < q_min[j] + margin:
                    rows.add(WARN, f"joint {number}", "no room to move inside the joint limits; skipped")
                    continue
                ramp(base, j, 0.0, commanded)
                hold(base)
                q = read()
                moved = float(q[j] - base[j])
                others = float(np.max(np.abs(np.delete(q - base, j))))
                ramp(base, j, commanded, 0.0)
                hold(base)
                verdict = motion_verdict(commanded, moved)
                data["joints"].append(
                    {
                        "joint": number,
                        "commanded": commanded,
                        "moved": moved,
                        "ratio": moved / commanded,
                        "verdict": verdict,
                        "others_max": others,
                    }
                )
                status = OK if verdict == "follows" else FAIL
                rows.add(
                    status,
                    f"joint {number}",
                    f"{verdict}: commanded {commanded:+.3f} rad, measured {moved:+.3f} rad (ratio "
                    f"{moved / commanded:+.2f}), other joints moved <= {others:.3f} rad",
                )
    except _MotionAbort as e:
        rows.add(FAIL, "aborted", f"{e}; the arm was released")
    except TakeoverError as e:
        rows.add(FAIL, "takeover", f"refused: {e}")
    _signs_rows(rows, cfg, side, data)
    return rows.report(name, data)


def _signs_rows(rows: _Rows, cfg: SiteConfig, side: str, data: dict) -> None:
    results = data["joints"]
    if not results:
        return
    signs = list(cfg.joint_signs[side])
    for r in results:
        if r["verdict"] == "reversed":
            signs[r["joint"] - 1] = -signs[r["joint"] - 1]
    data["joint_signs"] = signs
    verdicts = {r["verdict"] for r in results}
    line = f"joint_signs:  {side}: [{', '.join(str(s) for s in signs)}]"
    rows.add(INFO, "joint_signs", f"{line}   (or --set joint_signs.{side}=[{','.join(str(s) for s in signs)}])")
    if "reversed" in verdicts and cfg.command_mode == "absolute":
        rows.add(
            INFO,
            "hint",
            "joint_signs only apply with command_mode relative_to_activation; a reversed joint in absolute mode "
            "suggests the controller reads commands relative to activation: retry with "
            "--set command_mode=relative_to_activation and the signs above",
        )
    if verdicts <= {"no-motion"}:
        rows.add(
            INFO,
            "hint",
            "no joint moved: is the controller really following the command topic? Check arms."
            f"{side}.command_topic, takeover.mode and command_mode",
        )
    if "unclear" in verdicts:
        rows.add(INFO, "hint", "unclear joints moved too little or too much; rerun with a larger --delta")
