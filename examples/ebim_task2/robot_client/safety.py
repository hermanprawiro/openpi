"""Safety decisions of the real-robot client: stream staleness -> RUN / HOLD / FREEZE, and joint clamps.

Pure numpy; no ROS imports. Every check fails closed: a never-seen stream (age `inf`), a NaN age and
a NaN joint value all count as unsafe, never as "fine because the comparison was False".
"""

from collections.abc import Mapping
import enum
import math

import numpy as np

from robot_client.config import SafetyConfig


class Mode(enum.Enum):
    RUN = "run"
    HOLD = "hold"  # stop advancing, keep streaming the held pose; resumes when the cause clears
    FREEZE = "freeze"  # end the episode and release the controllers


_STREAM_KINDS = ("cam", "arm", "grip")


def _exceeds(value: float, threshold: float) -> bool:
    return not value <= threshold  # NaN exceeds every threshold


def _format_age(key: str, age: float) -> str:
    if math.isinf(age):
        return f"{key} never seen"
    return f"{key} age invalid (NaN)" if math.isnan(age) else f"{key} age {age:.2f} s"


def _badness(value: float, threshold: float) -> float:
    return math.inf if math.isnan(value) else value / threshold


def evaluate(
    ages: Mapping[str, float],
    infer_dry_s: float,
    fatal_inference: bool,  # noqa: FBT001 - positional by the runner's contract
    cfg: SafetyConfig,
) -> tuple[Mode, str]:
    """Decide the mode from stream ages (s since the last sample; `inf` if never seen), keyed
    `cam/<name>`, `arm/<side>`, `grip/<side>`, and the time the inference queue has been dry.

    FREEZE: fatal inference error, an arm older than `measured_dead_s`, any stream older than
    `obs_stale_freeze_s`, or inference dry longer than `infer_dry_freeze_s`. HOLD: any stream older than
    `obs_stale_hold_s`. The reason lists the offending streams, worst first (by age / threshold).
    """
    freeze: list[tuple[float, str]] = []
    hold: list[tuple[float, str]] = []
    if fatal_inference:
        freeze.append((math.inf, "fatal inference error"))
    if _exceeds(infer_dry_s, cfg.infer_dry_freeze_s):
        freeze.append(
            (
                _badness(infer_dry_s, cfg.infer_dry_freeze_s),
                f"inference dry for {infer_dry_s:.2f} s > infer_dry_freeze_s {cfg.infer_dry_freeze_s}",
            )
        )
    for key, age in ages.items():
        kind = key.partition("/")[0]
        if kind not in _STREAM_KINDS:
            raise ValueError(f"unknown stream key {key!r}; expected one of {[k + '/<name>' for k in _STREAM_KINDS]}")
        freeze_s, freeze_name = cfg.obs_stale_freeze_s, "obs_stale_freeze_s"
        if kind == "arm" and cfg.measured_dead_s < freeze_s:
            freeze_s, freeze_name = cfg.measured_dead_s, "measured_dead_s"
        if _exceeds(age, freeze_s):
            freeze.append((_badness(age, freeze_s), f"{_format_age(key, age)} > {freeze_name} {freeze_s}"))
        elif _exceeds(age, cfg.obs_stale_hold_s):
            hold.append(
                (
                    _badness(age, cfg.obs_stale_hold_s),
                    f"{_format_age(key, age)} > obs_stale_hold_s {cfg.obs_stale_hold_s}",
                )
            )

    for mode, causes in ((Mode.FREEZE, freeze), (Mode.HOLD, hold)):
        if causes:
            causes.sort(key=lambda c: c[0], reverse=True)  # stable: fatal stays first among infinities
            return mode, "; ".join(text for _, text in causes)
    return Mode.RUN, "ok"


def _as_checked(name: str, value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name}: shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: not finite: {arr.tolist()}")
    return arr


def clamp_step(prev: np.ndarray, target: np.ndarray, max_step: float) -> tuple[np.ndarray, bool]:
    """Limit each joint of `target` to within `max_step` of `prev`; the flag says whether any joint was cut."""
    prev = _as_checked("prev", prev)
    target = _as_checked("target", target, prev.shape)
    out = np.clip(target, prev - max_step, prev + max_step)
    return out, not np.array_equal(out, target)


def clamp_limits(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray, margin: float) -> tuple[np.ndarray, bool]:
    """Clamp `q` into [q_min + margin, q_max - margin]; the flag says whether any joint was clamped."""
    q = _as_checked("q", q)
    lo = _as_checked("q_min", q_min, q.shape) + margin
    hi = _as_checked("q_max", q_max, q.shape) - margin
    if np.any(lo > hi):
        raise ValueError(f"margin {margin} leaves no room inside the joint limits")
    out = np.clip(q, lo, hi)
    return out, not np.array_equal(out, q)


def lag_exceeded(cmd: np.ndarray, measured: np.ndarray, max_lag: float) -> bool:
    """True if any joint's |cmd - measured| exceeds `max_lag` (or is NaN)."""
    diff = np.abs(np.asarray(cmd, dtype=np.float64) - np.asarray(measured, dtype=np.float64))
    return not bool(np.all(diff <= max_lag))


def start_offset(first_target: np.ndarray, measured: np.ndarray) -> float:
    """Largest |first_target - measured| over the joints; `inf` if any value is non-finite."""
    first_target = np.asarray(first_target, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if first_target.shape != measured.shape:
        raise ValueError(f"first_target shape {first_target.shape} does not match measured {measured.shape}")
    offset = float(np.max(np.abs(first_target - measured)))
    return offset if math.isfinite(offset) else math.inf
