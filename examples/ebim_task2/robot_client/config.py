"""Site configuration of the real-robot client.

The organizers' operator adapts the client to their site only through a YAML file (see
`robot_site.munich.yaml`), `--set dotted.key=yaml_value` overrides and environment variables, so loading
is strict: unknown keys, wrong types and out-of-range values are all reported together as one
`ConfigError`, each naming the exact dotted key.

Precedence: dataclass defaults < YAML file < `--set` overrides. The policy-server connection
(`POLICY_SERVER_HOST` / `POLICY_SERVER_PORT` / `POLICY_SERVER_API_KEY`) comes from the environment only and
is rejected in the YAML, so a shared site file never carries it; `redacted_dict()` masks the key.

Pure Python (+ pyyaml); no ROS imports.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import difflib
import math
import os
from pathlib import Path
import types
import typing
from typing import Any, Literal

import yaml

from robot_client import layout

ARM_NAMES = ("left", "right")
CAMERA_NAMES = ("head", "wrist_left", "wrist_right")
N_JOINTS = 7

# FR3 joint limits (rad), identical for both arms.
_FR3_Q_MIN = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159)
_FR3_Q_MAX = (2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159)

ENV_HOST = "POLICY_SERVER_HOST"
ENV_PORT = "POLICY_SERVER_PORT"
ENV_API_KEY = "POLICY_SERVER_API_KEY"


class ConfigError(ValueError):
    """The site configuration is invalid; `problems` lists every issue found."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__(
            f"{len(self.problems)} problem(s) in the site configuration:\n"
            + "\n".join(f"  - {p}" for p in self.problems)
        )


def _yaml_key(name: str) -> dict:
    return {"yaml_key": name}


@dataclasses.dataclass(frozen=True)
class RosConfig:
    rmw: str = "rmw_fastrtps_cpp"
    domain_id: int = 0


@dataclasses.dataclass(frozen=True)
class PolicyConfig:
    prompt: str = layout.PROMPT
    control_hz: int = layout.CONTROL_HZ


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    topic: str
    type: Literal["raw", "compressed"] = "raw"
    # (W, H) the frame is resized to (only if its live size differs) before letterboxing to 224x224.
    train_res: tuple[int, int] | None = None
    # (x, y, w, h) crop applied before everything else (site knob; the training data is never cropped).
    crop: tuple[int, int, int, int] | None = None


@dataclasses.dataclass(frozen=True)
class GripperConfig:
    state_topic: str
    joint: str
    command_topic: str
    # False: the state stream may be absent (Munich left gripper); the state dim is then pinned.
    required: bool = True


@dataclasses.dataclass(frozen=True)
class StateConfig:
    # state dim -> constant replacing the live value before inference (Munich: left knuckle, dim 7 -> 0.0).
    pin: dict[int, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TakeoverConfig:
    # "switch": we call the controller_manager. "external": the organizers activate/deactivate the
    # controller with their own tooling; we stream hold, wait for `active`, and never call switch.
    mode: Literal["switch", "external"] = "switch"
    discovery_wait_s: float = 3.0
    warm_cycles: int = 20
    tol_rad: float = 0.02
    service_timeout_s: float = 5.0
    hold_inactive_arms: bool = True
    # The newest measured joint sample must be at most this old whenever takeover builds the hold or checks the
    # tolerance (a dead joint state stream would otherwise compare a stale pose with itself).
    max_measured_age_s: float = 0.3
    # An impedance controller that is already active when the client starts is refused (it would act on our very
    # first message). true: stream into it anyway (only if you know it follows absolute targets).
    accept_active: bool = False


@dataclasses.dataclass(frozen=True)
class ArmConfig:
    measured_topic: str
    command_topic: str
    joint_names: tuple[str, ...]
    controller_manager: str
    controller: str
    gripper: GripperConfig
    ptp_action: str
    start_pose: tuple[float, ...]
    # Joint names on the command topic if they differ from the measured ones (default: joint_names).
    command_joint_names: tuple[str, ...] | None = None


@dataclasses.dataclass(frozen=True)
class SpineConfig:
    action: str
    height: float  # metres
    velocity: float = 0.1
    acceleration: float = 0.2
    deceleration: float = 0.2
    enabled: bool = True
    get_position_service: str = "/franka_spine_node/get_position"
    max_delta_m: float = 0.2


@dataclasses.dataclass(frozen=True)
class AsyncConfig:
    trigger_threshold: float = 0.5
    merge: str = "blend_favor_old"
    blend_steps: int = 10


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
    mode: Literal["sync", "async"] = "sync"
    execute_horizon: int = 30
    async_cfg: AsyncConfig = dataclasses.field(default_factory=AsyncConfig, metadata=_yaml_key("async"))


def _default_joint_limits() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    return dict.fromkeys(ARM_NAMES, (_FR3_Q_MIN, _FR3_Q_MAX))


@dataclasses.dataclass(frozen=True)
class SafetyConfig:
    max_step_rad: float = 0.10
    max_track_lag_rad: float = 0.5
    obs_stale_hold_s: float = 1.0
    obs_stale_freeze_s: float = 5.0
    measured_dead_s: float = 1.5
    infer_dry_freeze_s: float = 5.0
    max_start_offset_rad: float = 0.3
    episode_timeout_s: float = 120.0
    q_margin_rad: float = 0.05
    gripper_deadband: float = 0.02
    # Longest the 100 Hz command stream may go without publishing on a side before it counts as unhealthy.
    stream_gap_freeze_s: float = 0.1
    # arm -> (q_min, q_max), 7 joints each.
    joint_limits: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = dataclasses.field(
        default_factory=_default_joint_limits
    )


@dataclasses.dataclass(frozen=True)
class SmootherConfig:
    impl: Literal["ruckig", "clamp"] = "ruckig"
    velocity_scale: float = 0.5
    v_max: tuple[float, ...] = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)
    a_max: tuple[float, ...] = (10.0,) * N_JOINTS
    j_max: tuple[float, ...] = (5000.0,) * N_JOINTS


@dataclasses.dataclass(frozen=True)
class PrepareConfig:
    ptp_max_joint_velocities: tuple[float, ...] = (0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4)
    ptp_goal_tolerance: float = 0.01
    timeout_s: float = 60.0


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    # host / port / api_key are taken from the environment only (never from the YAML).
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = dataclasses.field(default=None, repr=False)  # never printed
    connect_timeout_s: float = 10.0
    infer_timeout_s: float = 10.0


_ENV_ONLY_SERVER_KEYS = {"host": ENV_HOST, "port": ENV_PORT, "api_key": ENV_API_KEY}


def _default_joint_signs() -> dict[str, tuple[int, ...]]:
    return dict.fromkeys(ARM_NAMES, (1,) * N_JOINTS)


@dataclasses.dataclass(frozen=True)
class SiteConfig:
    cameras: dict[str, CameraConfig]
    arms: dict[str, ArmConfig]
    spine: SpineConfig
    ros: RosConfig = dataclasses.field(default_factory=RosConfig)
    policy: PolicyConfig = dataclasses.field(default_factory=PolicyConfig)
    state: StateConfig = dataclasses.field(default_factory=StateConfig)
    takeover: TakeoverConfig = dataclasses.field(default_factory=TakeoverConfig)
    active_arms: tuple[str, ...] = ARM_NAMES
    # "absolute": command = target joint positions. "relative_to_activation": command = target - the
    # measured pose at activation, as a GELLO-style teleop interface would publish.
    command_mode: Literal["absolute", "relative_to_activation"] = "absolute"
    joint_signs: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=_default_joint_signs)
    inference: InferenceConfig = dataclasses.field(default_factory=InferenceConfig)
    safety: SafetyConfig = dataclasses.field(default_factory=SafetyConfig)
    smoother: SmootherConfig = dataclasses.field(default_factory=SmootherConfig)
    prepare: PrepareConfig = dataclasses.field(default_factory=PrepareConfig)
    stream_hz: int = 100
    out_dir: str = "/out"
    server: ServerConfig = dataclasses.field(default_factory=ServerConfig)

    def redacted_dict(self) -> dict:
        """The config as plain data (YAML key names) with the API key masked; safe to write into results."""
        data = _to_plain(self)
        if data["server"]["api_key"]:
            data["server"]["api_key"] = "***"
        return data


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {_field_key(f): _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [_to_plain(v) for v in obj]
    return obj


def _field_key(f: dataclasses.Field) -> str:
    return f.metadata.get("yaml_key", f.name)


# ----------------------------------------------------------------------------------------------------
# Generic strict builder: raw YAML data + a dataclass type -> instance, collecting every problem.
# ----------------------------------------------------------------------------------------------------


class _Bad:
    """Marker for a value that failed to build (its problems are already recorded)."""


_BAD = _Bad()


def _describe(value: Any) -> str:
    return "null" if value is None else f"{value!r} ({type(value).__name__})"


def _build(tp: Any, value: Any, path: str, errs: list[str]) -> Any:
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _build(inner[0], value, path, errs)
    if origin is Literal:
        choices = typing.get_args(tp)
        if value not in choices:
            errs.append(f"{path}: must be one of {list(choices)}, got {_describe(value)}")
            return _BAD
        return value
    if dataclasses.is_dataclass(tp):
        return _build_dataclass(tp, value, path, errs)
    if origin is dict:
        return _build_dict(tp, value, path, errs)
    if origin is tuple:
        return _build_tuple(tp, value, path, errs)
    return _build_scalar(tp, value, path, errs)


def _build_scalar(tp: type, value: Any, path: str, errs: list[str]) -> Any:
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif tp is float:
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    else:  # pragma: no cover - a schema bug, not an operator error
        raise TypeError(f"unsupported config type {tp!r} at {path}")
    errs.append(f"{path}: expected {tp.__name__}, got {_describe(value)}")
    return _BAD


def _build_tuple(tp: Any, value: Any, path: str, errs: list[str]) -> Any:
    if not isinstance(value, list | tuple):
        errs.append(f"{path}: expected a list, got {_describe(value)}")
        return _BAD
    args = typing.get_args(tp)
    if len(args) == 2 and args[1] is Ellipsis:
        item_types = [args[0]] * len(value)
    else:
        if len(value) != len(args):
            errs.append(f"{path}: expected {len(args)} values, got {len(value)}")
            return _BAD
        item_types = list(args)
    items = [_build(t, v, f"{path}[{i}]", errs) for i, (t, v) in enumerate(zip(item_types, value, strict=True))]
    return _BAD if any(i is _BAD for i in items) else tuple(items)


def _build_dict(tp: Any, value: Any, path: str, errs: list[str]) -> Any:
    if not isinstance(value, dict):
        errs.append(f"{path}: expected a mapping, got {_describe(value)}")
        return _BAD
    key_type, value_type = typing.get_args(tp)
    out = {}
    bad = False
    for raw_key, raw_value in value.items():
        key = raw_key
        if key_type is int and isinstance(raw_key, str) and raw_key.lstrip("-").isdigit():
            key = int(raw_key)  # `--set state.pin.7=0.0` arrives with a string key
        sub = f"{path}.{raw_key}"
        if isinstance(key, bool) or not isinstance(key, key_type):
            errs.append(f"{sub}: key must be {key_type.__name__}, got {_describe(raw_key)}")
            bad = True
            continue
        built = _build(value_type, raw_value, sub, errs)
        if built is _BAD:
            bad = True
        else:
            out[key] = built
    return _BAD if bad else out


def _build_dataclass(cls: type, value: Any, path: str, errs: list[str]) -> Any:
    if value is None:
        value = {}  # an empty section keeps all defaults
    if not isinstance(value, dict):
        errs.append(f"{path or '<root>'}: expected a mapping, got {_describe(value)}")
        return _BAD
    start = len(errs)
    hints = typing.get_type_hints(cls)
    fields_by_key = {_field_key(f): f for f in dataclasses.fields(cls)}

    def sub_path(key: Any) -> str:
        return f"{path}.{key}" if path else str(key)

    for key in value:
        if key not in fields_by_key:
            hint = difflib.get_close_matches(str(key), fields_by_key, n=1)
            suffix = f" (did you mean '{hint[0]}'?)" if hint else f" (valid keys: {', '.join(fields_by_key)})"
            errs.append(f"{sub_path(key)}: unknown key{suffix}")

    built: dict[str, Any] = {}
    for key, f in fields_by_key.items():
        if key in value:
            result = _build(hints[f.name], value[key], sub_path(key), errs)
            if result is not _BAD:
                built[f.name] = result
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            errs.append(f"{sub_path(key)}: required key is missing")

    validator = _VALIDATORS.get(cls)
    if validator is not None:
        validator(_Ctx(path, built, value, errs))
    return _BAD if len(errs) > start else cls(**built)


# ----------------------------------------------------------------------------------------------------
# Semantic validators (ranges, lengths, cross-references). They see only the fields that built
# successfully, so one bad field never hides another problem.
# ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _Ctx:
    path: str
    built: dict[str, Any]
    raw: dict
    errs: list[str]

    def key(self, name: str) -> str:
        return f"{self.path}.{name}" if self.path else name

    def get(self, name: str) -> Any:
        return self.built.get(name)

    def fail(self, name: str, message: str) -> None:
        self.errs.append(f"{self.key(name)}: {message}")

    def positive(self, *names: str) -> None:
        for name in names:
            if name in self.built and not self.built[name] > 0:
                self.fail(name, f"must be > 0, got {self.built[name]}")

    def non_negative(self, *names: str) -> None:
        for name in names:
            if name in self.built and not self.built[name] >= 0:
                self.fail(name, f"must be >= 0, got {self.built[name]}")

    def non_empty(self, *names: str) -> None:
        for name in names:
            if name in self.built and not self.built[name].strip():
                self.fail(name, "must not be empty")

    def length(self, name: str, n: int = N_JOINTS) -> None:
        value = self.built.get(name)
        if value is None:
            return
        if len(value) != n:
            self.fail(name, f"expected {n} values, got {len(value)}")

    def exact_keys(self, name: str, expected: Sequence[str]) -> None:
        raw = self.raw.get(name)
        if not isinstance(raw, dict):
            return
        got = {str(k) for k in raw}
        missing = [k for k in expected if k not in got]
        extra = sorted(got - set(expected))
        if missing or extra:
            parts = [f"missing {missing}"] if missing else []
            parts += [f"unexpected {extra}"] if extra else []
            self.fail(name, f"keys must be exactly {list(expected)} ({'; '.join(parts)})")


def _validate_ros(c: _Ctx) -> None:
    c.non_empty("rmw")
    domain = c.get("domain_id")
    if domain is not None and not 0 <= domain <= 232:
        c.fail("domain_id", f"must be in [0, 232], got {domain}")


def _validate_policy(c: _Ctx) -> None:
    prompt = c.get("prompt")
    if prompt is not None and prompt != layout.PROMPT:
        c.fail(
            "prompt",
            f"must equal the trained prompt {layout.PROMPT!r} (got {prompt!r}); the policy server would refuse it",
        )
    hz = c.get("control_hz")
    if hz is not None and hz != layout.CONTROL_HZ:
        c.fail("control_hz", f"must equal the trained policy rate {layout.CONTROL_HZ} Hz, got {hz}")


def _validate_camera(c: _Ctx) -> None:
    c.non_empty("topic")
    res = c.get("train_res")
    if res is not None and not all(v > 0 for v in res):
        c.fail("train_res", f"(W, H) must be positive, got {list(res)}")
    crop = c.get("crop")
    if crop is not None:
        x, y, w, h = crop
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            c.fail("crop", f"(x, y, w, h) needs x, y >= 0 and w, h > 0, got {list(crop)}")


def _validate_gripper(c: _Ctx) -> None:
    c.non_empty("state_topic", "joint", "command_topic")


def _validate_state(c: _Ctx) -> None:
    for dim in c.get("pin") or {}:
        if not 0 <= dim < layout.STATE_DIM:
            c.errs.append(f"{c.key('pin')}.{dim}: state dim must be in [0, {layout.STATE_DIM - 1}]")


def _validate_takeover(c: _Ctx) -> None:
    c.positive("discovery_wait_s", "tol_rad", "service_timeout_s", "max_measured_age_s")
    c.non_negative("warm_cycles")


def _validate_arm(c: _Ctx) -> None:
    c.non_empty("measured_topic", "command_topic", "controller_manager", "controller", "ptp_action")
    for name in ("joint_names", "command_joint_names", "start_pose"):
        c.length(name)
    for name in ("joint_names", "command_joint_names"):
        names = c.get(name)
        if names and len(set(names)) != len(names):
            c.fail(name, "joint names must be unique")


def _validate_spine(c: _Ctx) -> None:
    c.non_empty("action", "get_position_service")
    height = c.get("height")
    if height is not None and not 0.0 <= height <= 1.0:
        c.fail("height", f"must be in [0.0, 1.0] metres, got {height} (a value in mm?)")
    c.positive("velocity", "acceleration", "deceleration", "max_delta_m")


def _validate_async(c: _Ctx) -> None:
    from eval_client.chunking import CHUNK_MERGE_STRATEGIES  # shared with the sim client

    merge = c.get("merge")
    if merge is not None and merge not in CHUNK_MERGE_STRATEGIES:
        c.fail("merge", f"must be one of {sorted(CHUNK_MERGE_STRATEGIES)}, got {merge!r}")
    threshold = c.get("trigger_threshold")
    if threshold is not None and not 0.0 <= threshold < 1.0:
        c.fail("trigger_threshold", f"must be in [0.0, 1.0), got {threshold}")
    blend = c.get("blend_steps")
    if blend is not None and blend < 1:
        c.fail("blend_steps", f"must be >= 1, got {blend}")


def _validate_inference(c: _Ctx) -> None:
    horizon = c.get("execute_horizon")
    if horizon is not None and not 1 <= horizon <= layout.ACTION_HORIZON:
        c.fail("execute_horizon", f"must be in [1, {layout.ACTION_HORIZON}], got {horizon}")


def _validate_safety(c: _Ctx) -> None:
    c.positive(
        "max_step_rad",
        "max_track_lag_rad",
        "obs_stale_hold_s",
        "obs_stale_freeze_s",
        "measured_dead_s",
        "infer_dry_freeze_s",
        "max_start_offset_rad",
        "episode_timeout_s",
        "stream_gap_freeze_s",
    )
    c.non_negative("q_margin_rad", "gripper_deadband")
    hold, freeze = c.get("obs_stale_hold_s"), c.get("obs_stale_freeze_s")
    if hold is not None and freeze is not None and freeze < hold:
        c.fail("obs_stale_freeze_s", f"must be >= obs_stale_hold_s ({hold}), got {freeze}")
    c.exact_keys("joint_limits", ARM_NAMES)
    for arm, (q_min, q_max) in (c.get("joint_limits") or {}).items():
        base = f"{c.key('joint_limits')}.{arm}"
        if len(q_min) != N_JOINTS or len(q_max) != N_JOINTS:
            c.errs.append(f"{base}: q_min and q_max need {N_JOINTS} values each, got {len(q_min)} and {len(q_max)}")
        elif any(lo >= hi for lo, hi in zip(q_min, q_max, strict=True)):
            c.errs.append(f"{base}: every q_min must be below its q_max")


def _validate_smoother(c: _Ctx) -> None:
    scale = c.get("velocity_scale")
    if scale is not None and not 0.0 < scale <= 1.0:
        c.fail("velocity_scale", f"must be in (0.0, 1.0], got {scale}")
    for name in ("v_max", "a_max", "j_max"):
        c.length(name)
        if name in c.built and not all(v > 0 for v in c.built[name]):
            c.fail(name, "all limits must be > 0")


def _validate_prepare(c: _Ctx) -> None:
    c.length("ptp_max_joint_velocities")
    vel = c.get("ptp_max_joint_velocities")
    if vel is not None and not all(v > 0 for v in vel):
        c.fail("ptp_max_joint_velocities", "all velocities must be > 0")
    c.positive("ptp_goal_tolerance", "timeout_s")


def _validate_server(c: _Ctx) -> None:
    c.non_empty("host")
    port = c.get("port")
    if port is not None and not 1 <= port <= 65535:
        c.fail("port", f"must be in [1, 65535], got {port}")
    c.positive("connect_timeout_s", "infer_timeout_s")


def _validate_site(c: _Ctx) -> None:
    c.exact_keys("cameras", CAMERA_NAMES)
    c.exact_keys("arms", ARM_NAMES)
    c.exact_keys("joint_signs", ARM_NAMES)
    c.positive("stream_hz")
    c.non_empty("out_dir")

    active = c.get("active_arms")
    if active is not None:
        arm_keys = {str(k) for k in c.raw.get("arms", {})} if isinstance(c.raw.get("arms"), dict) else set(ARM_NAMES)
        if not active:
            c.fail("active_arms", "must not be empty")
        unknown = [a for a in active if a not in arm_keys]
        if unknown:
            c.fail("active_arms", f"{unknown} not in arms {sorted(arm_keys)}")
        if len(set(active)) != len(active):
            c.fail("active_arms", f"duplicate entries in {list(active)}")

    for arm, signs in (c.get("joint_signs") or {}).items():
        if len(signs) != N_JOINTS:
            c.errs.append(f"{c.key('joint_signs')}.{arm}: expected {N_JOINTS} values, got {len(signs)}")
        elif any(s not in (-1, 1) for s in signs):
            c.errs.append(f"{c.key('joint_signs')}.{arm}: every sign must be 1 or -1, got {list(signs)}")

    # The streamer is unhealthy after a publish gap of stream_gap_freeze_s: allow at least two periods.
    hz, safety = c.get("stream_hz"), c.get("safety")
    if isinstance(hz, int | float) and hz > 0 and safety is not None:
        gap = safety.stream_gap_freeze_s
        if gap > 0 and gap < 2.0 / hz:
            c.errs.append(
                f"{c.key('safety')}.stream_gap_freeze_s: {gap} must be >= 2 / stream_hz "
                f"(= {2.0 / hz:.4g} s at stream_hz {hz})"
            )

    # start_pose must lie inside the joint limits (catches typos and unit slips before any motion).
    arms, limits = c.get("arms") or {}, (c.get("safety") or SafetyConfig()).joint_limits
    for arm, arm_cfg in arms.items():
        if arm not in limits or len(arm_cfg.start_pose) != N_JOINTS:
            continue
        q_min, q_max = limits[arm]
        for i, q in enumerate(arm_cfg.start_pose):
            if not q_min[i] <= q <= q_max[i]:
                c.errs.append(
                    f"{c.key('arms')}.{arm}.start_pose[{i}]: {q} is outside the joint limits "
                    f"[{q_min[i]}, {q_max[i]}] (safety.joint_limits.{arm})"
                )


_VALIDATORS = {
    RosConfig: _validate_ros,
    PolicyConfig: _validate_policy,
    CameraConfig: _validate_camera,
    GripperConfig: _validate_gripper,
    StateConfig: _validate_state,
    TakeoverConfig: _validate_takeover,
    ArmConfig: _validate_arm,
    SpineConfig: _validate_spine,
    AsyncConfig: _validate_async,
    InferenceConfig: _validate_inference,
    SafetyConfig: _validate_safety,
    SmootherConfig: _validate_smoother,
    PrepareConfig: _validate_prepare,
    ServerConfig: _validate_server,
    SiteConfig: _validate_site,
}


# ----------------------------------------------------------------------------------------------------
# Loading: YAML + --set overrides + environment.
# ----------------------------------------------------------------------------------------------------


def _apply_override(raw: dict, override: str, errs: list[str]) -> None:
    key, sep, text = override.partition("=")
    key = key.strip()
    if not sep or not key or any(not part for part in key.split(".")):
        errs.append(f"override {override!r}: expected dotted.key=yaml_value")
        return
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as e:
        errs.append(f"override {key}: value {text!r} is not valid YAML ({e})")
        return
    *parents, leaf = key.split(".")
    node = raw
    for i, part in enumerate(parents):
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        if not isinstance(child, dict):
            errs.append(f"override {key}: '{'.'.join(parents[: i + 1])}' is not a section (it is {_describe(child)})")
            return
        node = child
    node[leaf] = value


def _inject_env(raw: dict, env: Mapping[str, str], errs: list[str]) -> None:
    server = raw.get("server")
    if server is None:
        server = raw["server"] = {}
    if not isinstance(server, dict):
        return  # reported by the builder
    for key, env_name in _ENV_ONLY_SERVER_KEYS.items():
        if key in server:
            errs.append(f"server.{key}: not allowed in the YAML or --set; set the environment variable {env_name}")
    host, port, api_key = env.get(ENV_HOST), env.get(ENV_PORT), env.get(ENV_API_KEY)
    if host:
        server["host"] = host
    if port:
        try:
            server["port"] = int(port)
        except ValueError:
            errs.append(f"{ENV_PORT}: expected an integer, got {port!r}")
    if api_key:
        server["api_key"] = api_key


def load_site_config(
    path: str | Path, overrides: Sequence[str] = (), env: Mapping[str, str] = os.environ
) -> SiteConfig:
    """Load the site YAML, apply `dotted.key=yaml_value` overrides and the server env; raise ConfigError
    listing every problem."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError([f"site config file not found: {path}"])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError([f"{path}: not valid YAML ({e})"]) from e
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError([f"{path}: the top level must be a mapping, got {_describe(raw)}"])

    errs: list[str] = []
    for override in overrides:
        _apply_override(raw, override, errs)
    _inject_env(raw, env, errs)
    cfg = _build_dataclass(SiteConfig, raw, "", errs)
    if errs:
        raise ConfigError(errs)
    return cfg


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse a minimal KEY=VALUE .env file (comments, blank lines, optional `export `, quotes). A missing
    file gives {}. Callers merge it under the real environment: `{**load_env_file(p), **os.environ}`."""
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key:
            out[key] = value.strip().strip("'\"")
    return out


def check_ros_env(cfg: SiteConfig, env: Mapping[str, str]) -> list[str]:
    """Mismatches between the YAML `ros` section and the process environment (empty list: consistent).
    An unset variable counts as the ROS default (rmw_fastrtps_cpp on Humble, domain 0)."""
    problems = []
    rmw = env.get("RMW_IMPLEMENTATION") or "rmw_fastrtps_cpp"
    if rmw != cfg.ros.rmw:
        problems.append(f"RMW_IMPLEMENTATION is {rmw!r} but ros.rmw is {cfg.ros.rmw!r}")
    domain = env.get("ROS_DOMAIN_ID") or "0"
    try:
        domain_id = int(domain)
    except ValueError:
        problems.append(f"ROS_DOMAIN_ID {domain!r} is not an integer (ros.domain_id is {cfg.ros.domain_id})")
    else:
        if domain_id != cfg.ros.domain_id:
            problems.append(f"ROS_DOMAIN_ID is {domain_id} but ros.domain_id is {cfg.ros.domain_id}")
    return problems
