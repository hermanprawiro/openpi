"""Action-chunk bookkeeping for sync and async inference (pure, main-thread).

Actions are indexed by control step: a chunk inferred from the observation
at step k covers steps k..k+chunk_len-1, so rows before the next step to
publish are stale and skipped on arrival.
"""

from collections import deque
import math

import numpy as np

from eval_client.constants import ACTION_DIM


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


def _merge_replace(queue: deque, fresh: np.ndarray) -> None:
    queue.clear()
    queue.extend(fresh)


def _make_blend(old_weight: float, new_weight: float):
    """Average the step-aligned overlap of the remaining queue and the fresh
    chunk as old_weight * old + new_weight * new, then keep the tail of
    whichever plan extends further (a short fresh chunk never shortens the
    plan). Follows LeRobot's aggregate_fn idea with fixed weights."""

    def merge(queue: deque, fresh: np.ndarray) -> None:
        # reshape keeps an empty queue 2-D so the blend broadcasts
        old = np.asarray(list(queue), dtype=fresh.dtype).reshape(-1, fresh.shape[1])
        queue.clear()
        overlap = min(len(old), len(fresh))
        queue.extend(old_weight * old[:overlap] + new_weight * fresh[:overlap])
        queue.extend(old[overlap:] if len(old) > overlap else fresh[overlap:])

    return merge


CHUNK_MERGE_STRATEGIES = {
    "replace": _merge_replace,
    "blend_uniform": _make_blend(0.5, 0.5),
    "blend_favor_new": _make_blend(0.3, 0.7),
    "blend_favor_old": _make_blend(0.7, 0.3),
}


class AsyncChunkBuffer:
    """Main-thread bookkeeping for async inference (no locks: only the
    control loop touches it; worker traffic arrives via queue.Queue)."""

    def __init__(self, execute_horizon: int, trigger_threshold: float, merge: str = "replace"):
        if execute_horizon < 1:
            raise ValueError(f"execute_horizon must be >= 1, got {execute_horizon}")
        if not 0.0 <= trigger_threshold < 1.0:
            raise ValueError(f"trigger_threshold must be in [0.0, 1.0), got {trigger_threshold}")
        if merge not in CHUNK_MERGE_STRATEGIES:
            raise ValueError(f"Unknown chunk merge strategy {merge!r}; valid: {sorted(CHUNK_MERGE_STRATEGIES)}")
        self.execute_horizon = execute_horizon
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
        self._merge(self._queue, usable)

    def pop(self):
        """Next action, or the held last action while the queue is dry
        (None only before the first chunk ever arrives)."""
        if self._queue:
            self.depth_at_pop.append(len(self._queue))
            self.last_action = self._queue.popleft()
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
        return {
            "chunk_latency_steps_mean": round(float(np.mean(self.latency_steps)), 2) if self.latency_steps else None,
            "chunk_latency_steps_max": int(np.max(self.latency_steps)) if self.latency_steps else None,
            "stale_steps_skipped": self.stale_skipped,
            "queue_dry_steps": self.dry_steps,
            "queue_dry_events": self.dry_events,
            "queue_depth_at_pop_mean": round(float(np.mean(self.depth_at_pop)), 1) if self.depth_at_pop else None,
            "queue_depth_at_pop_min": int(np.min(self.depth_at_pop)) if self.depth_at_pop else None,
            "chunks_dropped": self.chunks_dropped,
        }
