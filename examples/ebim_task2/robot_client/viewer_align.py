"""Chunk alignment for the viewer: which row of which action chunk produced each executed control step.

Built only from the recorded columns (`steps.jsonl` records, `chunks.npz` arrays), never assuming
`step == plan_step`. It mirrors `Runner._tick`'s order: HOLD entry (queue dropped, epoch bumped) ->
`_handle_results` (chunk recorded, start check on a first chunk, `queue.on_chunk`) -> pop if `run` ->
`_record` -> `plan_step += 1` unless the async leash holds plan time. So `latency_steps[k]` is the receipt
tick's `plan_step` minus `obs_step[k]`.

- Receipt of chunk k: the first record with `plan_step == obs_step[k] + latency_steps[k]` (approximate in
  async mode, where leash ticks repeat a plan step). A first chunk that fails the start check FREEZEs the
  episode before `_record`: its receipt is the tick after the last record, flagged not recorded.
- Sync: the `_SyncQueue` is simulated (rows 0..execute_horizon-1 enqueued at receipt, one popped per `run`
  tick, dropped on `hold` / `paused`) and every executed action is checked against its row.
- Async: mapping by plan step. A first chunk (after start / HOLD) is loaded without skip, so row r runs at
  receipt plan step + r; a later chunk skips `latency_steps` rows, so row r runs at `obs_step + r`. A `run`
  tick belongs to the most recently received chunk (same epoch) that covers its plan step. Executed values may
  be blended, so no value check.
"""

from collections import deque
import dataclasses

import numpy as np

from robot_client import layout

ACTION_ATOL = 1e-5  # chunks.npz is float32, actions are float64 in steps.jsonl
CLAMP_TOL = 1e-9  # clamp_step returns the target exactly when it does not cut

Entry = tuple[int | None, int | None, bool]


@dataclasses.dataclass
class Alignment:
    per_step: list[Entry]  # per step record, in order: (chunk_k, row, clamped)
    chunks: list[dict]  # per chunk k, see `align`
    mismatches: int | None  # sync: run ticks whose action != the assigned row (or unassigned); async: None

    def to_json(self) -> dict:
        return {
            "per_step": [[k, r, bool(c)] for k, r, c in self.per_step],
            "chunks": [_chunk_json(c) for c in self.chunks],
            "mismatches": self.mismatches,
        }


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _chunk_json(info: dict) -> dict:
    out = dict(info)
    out["executed"] = None if info["executed"] is None else [int(v) for v in info["executed"]]
    out["start_offset"] = {side: _finite_or_none(v) for side, v in info["start_offset"].items()}
    return out


def _start_offset(row0: np.ndarray, record: dict | None, side: str) -> float | None:
    measured = (record or {}).get("measured") or {}
    if side not in measured or measured[side] is None:
        return None
    try:
        q = np.asarray(measured[side], dtype=np.float64)
        return float(np.max(np.abs(row0[layout.ARM_SLICES[side]].astype(np.float64) - q)))
    except (TypeError, ValueError):
        return None


def _clamped(record: dict, active_arms: list[str]) -> bool:
    action = record.get("action")
    waypoints = record.get("waypoints") or {}
    if action is None:
        return False
    try:
        action = np.asarray(action, dtype=np.float64)
        for side in active_arms:
            wp = waypoints.get(side)
            if wp is None:
                continue
            diff = np.abs(np.asarray(wp, dtype=np.float64) - action[layout.ARM_SLICES[side]])
            if not np.all(diff <= CLAMP_TOL):
                return True
    except (TypeError, ValueError, IndexError):
        return False
    return False


def _matches(action, row: np.ndarray) -> bool:
    if action is None:
        return False
    try:
        return bool(np.allclose(np.asarray(action, dtype=np.float64), row, rtol=0.0, atol=ACTION_ATOL))
    except (TypeError, ValueError):
        return False


def align(
    steps: list[dict], chunks: dict | None, *, mode: str, execute_horizon: int, active_arms: list[str]
) -> Alignment:
    """Reconstruct the chunk row behind every step record.

    `Alignment.chunks[k]`: `k`, `receipt_step` (None if not received yet), `receipt_recorded`, `receipt_approx`,
    `first` (first chunk of its epoch; None without a receipt), `stale_rows` (rows skipped on load),
    `executed` ([first_step, last_step] or None), `n_executed`, `n_clamped`, `start_offset` {side: max |row 0 -
    measured at receipt| or None}.
    """
    is_async = mode == "async"
    arr = None if chunks is None else np.asarray(chunks["chunks"])
    n_chunks = 0 if arr is None else int(arr.shape[0])
    n_rows = 0 if arr is None else int(arr.shape[1])
    horizon = max(0, min(int(execute_horizon), n_rows))

    first_index: dict = {}  # plan_step -> index of the first record with it
    for i, s in enumerate(steps):
        first_index.setdefault(s.get("plan_step"), i)
    last = steps[-1] if steps else None

    # ---- receipts ----
    infos: list[dict] = []
    receipt_index: list[int | None] = []  # record index of the receipt tick (None: unrecorded / not yet)
    chunk_epoch: list = []
    seen_epochs: set = set()
    for k in range(n_chunks):
        obs = int(chunks["obs_step"][k])
        latency = int(chunks["latency_steps"][k])
        target = obs + latency
        idx = first_index.get(target)
        ref = None
        if idx is not None:
            ref = steps[idx]
            receipt_step, recorded = ref.get("step"), True
        elif last is not None and target == last.get("plan_step", float("nan")) + 1:
            ref = last  # FREEZE'd on that tick (start check): epoch / measured are the last record's
            receipt_step, recorded = last.get("step") + 1, False
        else:
            receipt_step, recorded = None, False
        if ref is None:
            first = None
            epoch = None
        else:
            epoch = ref.get("epoch")
            first = epoch not in seen_epochs
            seen_epochs.add(epoch)
        stale = latency if (is_async and first is False) else 0
        infos.append(
            {
                "k": k,
                "receipt_step": receipt_step,
                "receipt_recorded": recorded,
                "receipt_approx": is_async and receipt_step is not None,
                "first": first,
                "stale_rows": stale,
                "executed": None,
                "n_executed": 0,
                "n_clamped": 0,
                "start_offset": {
                    side: None if ref is None else _start_offset(arr[k, 0], ref, side) for side in active_arms
                },
            }
        )
        receipt_index.append(idx)
        chunk_epoch.append(epoch)

    # ---- per step ----
    per_step: list[Entry] = []
    mismatches = 0
    if is_async:
        # Row window per chunk: row r runs at plan step base + r for lo <= r < hi.
        windows = []
        for k, info in enumerate(infos):
            latency = int(chunks["latency_steps"][k])
            if info["first"] is None:
                windows.append(None)
            elif info["first"]:
                windows.append((int(chunks["obs_step"][k]) + latency, 0, horizon))
            else:
                windows.append((int(chunks["obs_step"][k]), latency, min(latency + horizon, n_rows)))
        received: list[int] = []  # chunks received so far, in receipt order
        pending = sorted((info["receipt_step"], k) for k, info in enumerate(infos) if info["receipt_step"] is not None)
        p = 0
        for s in steps:
            while p < len(pending) and pending[p][0] <= s.get("step"):
                received.append(pending[p][1])
                p += 1
            if s.get("mode") != "run":
                per_step.append((None, None, False))
                continue
            k_hit = r_hit = None
            plan = s.get("plan_step")
            for k in reversed(received):
                if chunk_epoch[k] != s.get("epoch"):
                    break  # a HOLD dropped every earlier chunk
                base, lo, hi = windows[k]
                r = plan - base
                if lo <= r < hi:
                    k_hit, r_hit = k, r
                    break
            per_step.append((k_hit, r_hit, _clamped(s, active_arms)))
    else:
        arrivals: dict[int, list[int]] = {}
        for k, idx in enumerate(receipt_index):
            if idx is not None:
                arrivals.setdefault(idx, []).append(k)
        queue: deque = deque()
        for i, s in enumerate(steps):
            m = s.get("mode")
            if m in ("hold", "paused"):
                queue.clear()
            for k in arrivals.get(i, ()):
                queue.extend((k, r) for r in range(horizon))
            if m != "run":
                per_step.append((None, None, False))
                continue
            k, r = queue.popleft() if queue else (None, None)
            if k is None or not _matches(s.get("action"), arr[k, r]):
                mismatches += 1
            per_step.append((k, r, _clamped(s, active_arms)))

    for s, (k, _, clamped) in zip(steps, per_step, strict=True):
        if k is None:
            continue
        info = infos[k]
        step = s.get("step")
        info["executed"] = [step, step] if info["executed"] is None else [info["executed"][0], step]
        info["n_executed"] += 1
        info["n_clamped"] += int(clamped)

    return Alignment(per_step=per_step, chunks=infos, mismatches=None if is_async else mismatches)
