"""Action-chunk bookkeeping for sync and async inference (pure, main-thread).

Actions are indexed by control step: a chunk inferred from the observation
at step k covers steps k..k+chunk_len-1, so rows before the next step to
publish are stale and skipped on arrival.

Merge strategies reconcile the usable part of a fresh chunk with the
not-yet-executed queue (background and trade-offs in
docs/chunk_merge_strategies.md). Only BLEND_ACTION_DIMS (arm joints +
spine) are ever interpolated; base twist and gripper dims always take the
fresh chunk's value.
"""

from collections import deque
import math

import numpy as np

from eval_client.constants import ACTION_DIM
from eval_client.constants import ARM_ACTION_DIMS
from eval_client.constants import BLEND_ACTION_DIMS

_ARM = np.asarray(ARM_ACTION_DIMS)
_BLEND = np.asarray(BLEND_ACTION_DIMS)


def _validate_chunk(chunk) -> np.ndarray:
    chunk = np.asarray(chunk)
    if chunk.ndim != 2 or chunk.shape[1] < ACTION_DIM:
        raise ValueError(f"Bad action chunk shape: {chunk.shape}")
    return chunk


class ChunkExecutor:
    """Sync mode: execute the first replan_steps of each chunk."""

    def __init__(self, replan_steps: int):
        self.replan_steps = replan_steps
        self._queue = deque()

    @property
    def needs_infer(self) -> bool:
        return not self._queue

    def load(self, chunk) -> None:
        chunk = _validate_chunk(chunk)
        self._queue.extend(chunk[: self.replan_steps, :ACTION_DIM])

    def next_action(self) -> np.ndarray:
        return self._queue.popleft()


# Merge strategies share one signature: merge(queue, fresh, last_action,
# blend_steps). `fresh` is the usable (non-stale) slice of the new chunk,
# step-aligned with queue[0]; `last_action` is the last published action
# (None before the first chunk) and `blend_steps` the hand-off window.


def _queue_array(queue: deque, like: np.ndarray) -> np.ndarray:
    """Queue contents as a 2-D array (an empty queue stays 2-D so blends
    broadcast)."""
    return np.asarray(list(queue), dtype=like.dtype).reshape(-1, like.shape[1])


def _mix(old: np.ndarray, new: np.ndarray, w_old, w_new) -> np.ndarray:
    """Rows of `new` whose blend dims are replaced by w_old*old + w_new*new.
    Weights are scalars or (rows, 1) columns, so they may vary per row."""
    out = new.copy()
    out[:, _BLEND] = w_old * old[:, _BLEND] + w_new * new[:, _BLEND]
    return out


def _merge_replace(queue: deque, fresh: np.ndarray, last_action, blend_steps: int) -> None:
    queue.clear()
    queue.extend(fresh)


def _make_blend(old_weight: float, new_weight: float):
    """Average the step-aligned overlap of the remaining queue and the fresh
    chunk with fixed weights, then keep the tail of whichever plan extends
    further (a short fresh chunk never shortens the plan). Follows
    LeRobot's aggregate_fn idea."""

    def merge(queue: deque, fresh: np.ndarray, last_action, blend_steps: int) -> None:
        old = _queue_array(queue, fresh)
        queue.clear()
        overlap = min(len(old), len(fresh))
        queue.extend(_mix(old[:overlap], fresh[:overlap], old_weight, new_weight))
        queue.extend(old[overlap:] if len(old) > overlap else fresh[overlap:])

    return merge


def _merge_crossfade(queue: deque, fresh: np.ndarray, last_action, blend_steps: int) -> None:
    """Ramp from the old plan into the fresh one over B = min(blend_steps,
    overlap) rows with alpha_t = (t + 1) / (B + 1), pure fresh afterwards;
    tail rule as blend_*. Continuous at both ends of the window, unlike a
    fixed-weight blend. On a dry queue there is nothing to ramp from and
    this degrades to replace (offset_decay covers that case)."""
    old = _queue_array(queue, fresh)
    queue.clear()
    overlap = min(len(old), len(fresh))
    window = min(blend_steps, overlap)
    alpha = ((np.arange(window) + 1) / (window + 1))[:, None]
    queue.extend(_mix(old[:window], fresh[:window], 1.0 - alpha, alpha))
    queue.extend(fresh[window:overlap])
    queue.extend(old[overlap:] if len(old) > overlap else fresh[overlap:])


def _merge_offset_decay(queue: deque, fresh: np.ndarray, last_action, blend_steps: int) -> None:
    """Adopt the fresh plan wholesale (replace's reactivity) but shift it so
    its first row matches what would have been published next -- queue[0],
    or the held last action when the queue is dry -- and bleed that shift
    off linearly over D = min(blend_steps, len(fresh)) rows (weights 1,
    1 - 1/D, ..., 1/D). Preserves the fresh chunk's shape instead of
    averaging two plans. The first chunk ever is a pass-through."""
    anchor = queue[0] if queue else last_action
    queue.clear()
    out = fresh.copy()
    if anchor is not None:
        decay_len = min(blend_steps, len(fresh))
        delta = np.asarray(anchor)[_BLEND] - fresh[0, _BLEND]
        weights = ((decay_len - np.arange(decay_len)) / decay_len)[:, None]
        out[:decay_len, _BLEND] += weights * delta
    queue.extend(out)


CHUNK_MERGE_STRATEGIES = {
    "replace": _merge_replace,
    "blend_uniform": _make_blend(0.5, 0.5),
    "blend_favor_new": _make_blend(0.3, 0.7),
    "blend_favor_old": _make_blend(0.7, 0.3),
    "offset_decay": _merge_offset_decay,
    "crossfade": _merge_crossfade,
}


def _summary(values: list, ndigits: int) -> tuple:
    if not values:
        return None, None
    return round(float(np.mean(values)), ndigits), round(float(np.max(values)), ndigits)


class AsyncChunkBuffer:
    """Main-thread bookkeeping for async inference (no locks: only the
    control loop touches it; worker traffic arrives via queue.Queue)."""

    def __init__(self, execute_horizon: int, trigger_threshold: float, merge: str = "replace", blend_steps: int = 10):
        if execute_horizon < 1:
            raise ValueError(f"execute_horizon must be >= 1, got {execute_horizon}")
        if not 0.0 <= trigger_threshold < 1.0:
            raise ValueError(f"trigger_threshold must be in [0.0, 1.0), got {trigger_threshold}")
        if merge not in CHUNK_MERGE_STRATEGIES:
            raise ValueError(f"Unknown chunk merge strategy {merge!r}; valid: {sorted(CHUNK_MERGE_STRATEGIES)}")
        if blend_steps < 1:
            raise ValueError(f"blend_steps must be >= 1, got {blend_steps}")
        self.execute_horizon = execute_horizon
        self.blend_steps = blend_steps
        # Request a fresh chunk while this many actions remain; >= 1 so
        # threshold 0.0 still requests one step before running dry.
        self.trigger_len = max(1, math.ceil(trigger_threshold * execute_horizon))
        self._merge = CHUNK_MERGE_STRATEGIES[merge]
        self._queue = deque()
        self.in_flight = False
        self.last_action = None
        self.depth_at_pop = []
        self.latency_steps = []
        self.stale_skipped = 0
        self.dry_steps = 0
        self.dry_events = 0
        self.chunks_dropped = 0
        self._dry_run = 0
        # Max |delta| over arm joints between consecutive published actions
        # (rad); switch_jumps is the subset at the first pop after a merge.
        self.step_jumps = []
        self.switch_jumps = []
        self._pending_switch = False

    def __len__(self) -> int:
        return len(self._queue)

    def should_request(self) -> bool:
        return not self.in_flight and len(self._queue) <= self.trigger_len

    def on_request_sent(self) -> None:
        self.in_flight = True

    def on_chunk(self, obs_step: int, chunk, next_step: int) -> None:
        chunk = _validate_chunk(chunk)
        self.in_flight = False
        skip = next_step - obs_step
        self.latency_steps.append(skip)
        usable = chunk[skip : skip + self.execute_horizon, :ACTION_DIM]
        if len(usable) == 0:
            # Whole chunk predates next_step (very late arrival);
            # should_request() re-fires on the next step.
            self.chunks_dropped += 1
            return
        self.stale_skipped += skip
        self._merge(self._queue, usable, self.last_action, self.blend_steps)
        self._pending_switch = True

    def pop(self):
        """Next action, or the held last action while the queue is dry
        (None only before the first chunk ever arrives)."""
        if self._queue:
            self.depth_at_pop.append(len(self._queue))
            action = self._queue.popleft()
            if self.last_action is not None:
                jump = float(np.max(np.abs(action[_ARM] - self.last_action[_ARM])))
                self.step_jumps.append(jump)
                if self._pending_switch:
                    self.switch_jumps.append(jump)
            self._pending_switch = False
            self.last_action = action
            self._dry_run = 0
        else:
            if self._dry_run == 0:
                self.dry_events += 1
            self._dry_run += 1
            self.dry_steps += 1
        return self.last_action

    @property
    def consecutive_dry(self) -> int:
        return self._dry_run

    def stats(self) -> dict:
        switch_mean, switch_max = _summary(self.switch_jumps, 4)
        step_mean, step_max = _summary(self.step_jumps, 4)
        return {
            "chunk_latency_steps_mean": round(float(np.mean(self.latency_steps)), 2) if self.latency_steps else None,
            "chunk_latency_steps_max": int(np.max(self.latency_steps)) if self.latency_steps else None,
            "stale_steps_skipped": self.stale_skipped,
            "queue_dry_steps": self.dry_steps,
            "queue_dry_events": self.dry_events,
            "queue_depth_at_pop_mean": round(float(np.mean(self.depth_at_pop)), 1) if self.depth_at_pop else None,
            "queue_depth_at_pop_min": int(np.min(self.depth_at_pop)) if self.depth_at_pop else None,
            "chunks_dropped": self.chunks_dropped,
            "switch_jump_mean": switch_mean,
            "switch_jump_max": switch_max,
            "step_jump_mean": step_mean,
            "step_jump_max": step_max,
        }
