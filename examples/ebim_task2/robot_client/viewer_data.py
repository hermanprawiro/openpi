"""Reads a robot-client run folder for the viewer (read-only, incremental).

A run folder is what `recorder.py` writes: `config.yaml`, `events.log`, `episode_NNN/{steps.jsonl, chunks.npz,
images/, result.json}` and, for check-only runs, `check_obs_<camera>.jpg`. Files of a live run are still being
written: `steps.jsonl` grows line by line (the last line may be partial) and `chunks.npz` is replaced atomically.

This module deliberately imports neither `robot_client.recorder` nor `robot_client.config` (no rclpy / yaml /
eval_client at import time): `yaml` is imported lazily inside `config_summary`.
"""

import json
import math
from pathlib import Path
import re
import time
import zipfile
import zlib

import numpy as np

LIVE_STALE_S = 7200  # the client is silent for minutes at the start gate; a crashed client leaves no run_end
RUN_EVENT_KINDS = (
    "takeover",
    "freeze_before_start",
    "refused",
    "interrupted",
    "worker_restarted",
    "quit",
    "release_errors",
    "prepare",
)
_EPISODE_DIR = re.compile(r"episode_(\d+)$")
_IMAGE_FILE = re.compile(r"step_(\d+)\.jpg$")
_CHUNK_KEYS = ("obs_step", "t_request", "t_receive", "latency_steps", "chunks")
_RETRY_PAUSE_S = 0.02


def sanitize(obj):
    """JSON-safe copy: numpy -> Python, non-finite floats -> None (recursive)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ----------------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------------


def _episode_dirs(run_dir: Path) -> list[Path]:
    try:
        entries = [p for p in run_dir.iterdir() if p.is_dir() and _EPISODE_DIR.match(p.name)]
    except OSError:
        return []
    return sorted(entries, key=lambda p: p.name)


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _last_run_start(events: list[dict]) -> int | None:
    """Position of the last `run_start` in `events`: a run folder reused by `run --run-name NAME` holds one
    run_start ... run_end pair per run (events.log is append-only), and only the last one can still be running."""
    for pos in range(len(events) - 1, -1, -1):
        if events[pos].get("kind") == "run_start":
            return pos
    return None


def _newest_activity(run_dir: Path) -> float | None:
    """mtime of events.log or of the last episode's steps.jsonl, whichever is newer."""
    episodes = _episode_dirs(run_dir)
    stamps = [_mtime(run_dir / "events.log")]
    if episodes:
        stamps.append(_mtime(episodes[-1] / "steps.jsonl"))
    return max((s for s in stamps if s is not None), default=None)


def run_state(run_dir: Path, events: list[dict] | None = None, now: float | None = None) -> dict:
    """`live`: the last `run_start` has no `run_end` after it, and events.log (or the last episode's steps.jsonl)
    was written recently. `started`: wall time of the last run_start (None without one). `age_s`: seconds since
    that newest write (None if neither file exists)."""
    now = time.time() if now is None else now
    events = read_events(run_dir) if events is None else events
    start = _last_run_start(events)
    newest = _newest_activity(Path(run_dir))
    age = None if newest is None else max(0.0, now - newest)
    open_run = start is not None and not any(e.get("kind") == "run_end" for e in events[start + 1 :])
    started = events[start].get("wall") if start is not None else None
    return {
        "live": open_run and age is not None and age <= LIVE_STALE_S,
        "started": started if isinstance(started, int | float) else None,
        "age_s": age,
    }


def run_is_live(run_dir: Path, now: float | None = None, events: list[dict] | None = None) -> bool:
    """See `run_state`."""
    return run_state(run_dir, events, now)["live"]


def list_runs(root: Path, now: float | None = None) -> list[dict]:
    """Run directories under `root`, newest first: `{"name", "kind", "mtime", "live", "started", "age_s"}`."""
    now = time.time() if now is None else now
    try:
        dirs = [p for p in Path(root).iterdir() if p.is_dir()]
    except OSError:
        return []
    runs = []
    for run_dir in dirs:
        mtime = _mtime(run_dir)
        if mtime is None:
            continue
        if _episode_dirs(run_dir):
            kind = "episodes"
        elif check_obs_cameras(run_dir):
            kind = "check-obs"
        else:
            kind = "other"
        runs.append({"name": run_dir.name, "kind": kind, "mtime": mtime, **run_state(run_dir, now=now)})
    runs.sort(key=lambda r: (r["mtime"], r["name"]), reverse=True)
    return runs


def read_events(run_dir: Path) -> list[dict]:
    """The run-level events in file order; unparsable (e.g. half-written) lines are skipped."""
    try:
        text = (Path(run_dir) / "events.log").read_text(errors="replace")
    except OSError:
        return []
    events = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            events.append(record)
    return events


def config_summary(run_dir: Path) -> dict | None:
    """The config fields the page shows; `state.pin` keys (strings on disk) become ints. None if unavailable."""
    path = Path(run_dir) / "config.yaml"
    try:  # the whole body: a missing yaml, an unreadable file, a parse error or an odd structure all give None
        import yaml

        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            return None
        inference = data.get("inference") or {}
        pin = ((data.get("state") or {}).get("pin")) or {}
        return {
            "active_arms": data.get("active_arms"),
            "mode": inference.get("mode"),
            "execute_horizon": inference.get("execute_horizon"),
            "pin": {int(k): v for k, v in pin.items()},
            "safety": data.get("safety") or {},
        }
    except Exception:
        return None


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def episode_list(run_dir: Path, events: list[dict], run_live: bool) -> list[dict]:  # noqa: FBT001
    """One entry per `episode_NNN/` directory. Ending from the `episode_end` event (or a following `interrupted`
    event), outcome from result.json (else the `outcome` event). Only an episode of the last run (started after the
    last `run_start`) can be live: an earlier run's episode without an ending was cut off by a killed client."""
    last_start = _last_run_start(events)
    last_start = -1 if last_start is None else last_start
    started_before = [e.get("episode") for e in events[: last_start + 1] if e.get("kind") == "episode_start"]
    newest_before = max((i for i in started_before if isinstance(i, int)), default=-1)
    entries = []
    for ep_dir in _episode_dirs(Path(run_dir)):
        index = int(_EPISODE_DIR.match(ep_dir.name).group(1))
        start_pos = None
        dry_run = False
        end = None
        outcome_event = None
        for pos, event in enumerate(events):
            if event.get("episode") != index:
                continue
            kind = event.get("kind")
            if kind == "episode_start":
                start_pos, dry_run = pos, bool(event.get("dry_run"))
            elif kind == "episode_end":
                end = event
            elif kind == "outcome":
                outcome_event = event
        ended_by = reason = None
        if end is not None:
            ended_by, reason = end.get("ended_by"), end.get("reason")
        elif start_pos is not None:
            for event in events[start_pos + 1 :]:
                if event.get("kind") in ("episode_start", "run_start"):
                    break
                if event.get("kind") == "interrupted":
                    ended_by, reason = "interrupted", event.get("note")
                    break
        # no episode_start yet (the directory is created just before it): new iff numbered after the earlier runs'
        current = start_pos > last_start if start_pos is not None else index > newest_before
        result = _read_json(ep_dir / "result.json")
        outcome = result.get("outcome") if result else None
        if outcome is None and outcome_event is not None:
            outcome = outcome_event.get("outcome")
        entries.append(
            {
                "name": ep_dir.name,
                "index": index,
                "dry_run": dry_run,
                "ended_by": ended_by,
                "reason": reason,
                "outcome": outcome,
                "live": ended_by is None and run_live and current,
                "has_result": result is not None,
            }
        )
    return entries


def run_events(events: list[dict]) -> list[dict]:
    """The gate-level events the page lists."""
    return [e for e in events if e.get("kind") in RUN_EVENT_KINDS]


# ----------------------------------------------------------------------------------------------------
# Episode files
# ----------------------------------------------------------------------------------------------------


class StepsTail:
    """Incremental reader of a growing `steps.jsonl`: reads only appended bytes and holds back a partial last line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: list[dict] = []
        self._offset = 0  # bytes consumed (complete lines only)

    def read(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return list(self._records)
        if size < self._offset:  # truncated or replaced: start over
            self._records, self._offset = [], 0
        if size > self._offset:
            try:
                with open(self.path, "rb") as f:
                    f.seek(self._offset)
                    data = f.read(size - self._offset)
            except OSError:
                return list(self._records)
            end = data.rfind(b"\n") + 1  # complete lines only
            for line in data[:end].splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    self._records.append(record)
            self._offset += end
        return list(self._records)


class ChunksCache:
    """`chunks.npz` loaded as arrays; reloaded only when (mtime_ns, size) changes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._sig: tuple[int, int] | None = None
        self._value: dict | None = None

    def read(self) -> dict | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._sig:
            return self._value
        for attempt in range(2):
            try:
                value = self._load()
            except (OSError, ValueError, KeyError, EOFError, zlib.error, zipfile.BadZipFile):
                if attempt == 0:
                    time.sleep(_RETRY_PAUSE_S)
                continue
            self._sig, self._value = sig, value
            return value
        return self._value  # still unreadable (mid-write?): keep the previous value, try again next time

    def _load(self) -> dict | None:
        with np.load(self.path) as z:
            data = {key: np.array(z[key]) for key in _CHUNK_KEYS}
        if data["chunks"].ndim != 3 or data["chunks"].shape[0] == 0:
            return None
        return data


def image_index(ep_dir: Path) -> dict[str, list[int]]:
    """camera -> sorted steps that have a recorded image."""
    index: dict[str, list[int]] = {}
    try:
        cameras = sorted(p for p in (Path(ep_dir) / "images").iterdir() if p.is_dir())
    except OSError:
        return index
    for cam_dir in cameras:
        steps = [int(m.group(1)) for p in cam_dir.iterdir() if (m := _IMAGE_FILE.match(p.name))]
        index[cam_dir.name] = sorted(steps)
    return index


def check_obs_cameras(run_dir: Path) -> list[str]:
    """Cameras that have a `check_obs_<camera>.jpg` in the run dir."""
    try:
        names = [p.name for p in Path(run_dir).iterdir() if p.is_file()]
    except OSError:
        return []
    return sorted(
        n[len("check_obs_") : -len(".jpg")] for n in names if n.startswith("check_obs_") and n.endswith(".jpg")
    )
