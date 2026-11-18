"""Diagnostics recorder: the only thing the organizers hand back to us after a run.

Layout of one run:

    <out_dir>/<run_name>/
        config.yaml            redacted site config
        server_metadata.json   what the policy server advertised
        events.log             JSON lines, one per `RunRecorder.event`
        episode_NNN/
            steps.jsonl        one JSON line per control step (line-buffered)
            chunks.npz         every received action chunk (rewritten after each chunk)
            images/<camera>/step_NNNNNN.jpg   policy inputs at 2 Hz
            result.json, stream_stats.json    written by `finish`

Everything is flushed as it is written so a crash mid-episode still leaves usable files. The recorder is
diagnostics only: `event`, `step`, `chunk` and `images` never raise into the control loop. The first error of a
sink is logged and that sink is disabled; `finish` still attempts every final write and records what failed.
"""

import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Literal

import numpy as np
from PIL import Image
import yaml

from robot_client import layout

logger = logging.getLogger(__name__)

Outcome = Literal["s", "f", "x", "freeze"]
OUTCOMES = ("s", "f", "x", "freeze")
JPEG_QUALITY = 90


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _dumps(data) -> str:
    return json.dumps(data, default=_json_default)


def _redact(obj):
    """Defence in depth: the caller passes a redacted config, but never let a raw api_key reach the disk."""
    if isinstance(obj, dict):
        return {k: ("***" if k == "api_key" and v else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_redact(v) for v in obj]
    return obj


def _write_atomic(path: Path, write, *, sync: bool = True) -> None:
    """Write via a temp file + os.replace (survives a process crash); fsync too unless `sync` is False."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        if sync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_json(path: Path, data) -> None:
    _write_atomic(path, lambda f: f.write(_dumps(data).encode()))


class RunRecorder:
    def __init__(self, out_dir: Path, run_name: str, config: dict, server_metadata: dict | None):
        self.dir = Path(out_dir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(
            self.dir / "config.yaml",
            lambda f: f.write(yaml.safe_dump(_redact(json.loads(_dumps(config))), sort_keys=False).encode()),
        )
        _write_json(self.dir / "server_metadata.json", server_metadata or {})
        self._lock = threading.Lock()
        self._events_ok = True
        self._events = open(self.dir / "events.log", "a", buffering=1)  # noqa: SIM115

    def event(self, kind: str, **fields) -> None:
        with self._lock:
            if not self._events_ok:
                return
            try:
                record = {"wall": time.time(), "mono": time.monotonic(), "kind": kind, **fields}
                self._events.write(_dumps(record) + "\n")
            except Exception:
                self._events_ok = False
                logger.exception("recorder: events.log write failed; event logging disabled")

    def episode(self, idx: int) -> "EpisodeRecorder":
        return EpisodeRecorder(self.dir / f"episode_{idx:03d}", control_hz=layout.CONTROL_HZ)

    def close(self) -> None:
        with self._lock:
            self._events.close()


class EpisodeRecorder:
    def __init__(self, ep_dir: Path, control_hz: int = layout.CONTROL_HZ):
        self.dir = Path(ep_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.image_every = max(1, control_hz // 2)
        self._lock = threading.Lock()
        self._finished = False
        self._disabled: set[str] = set()  # sinks ("steps", "chunks", "images") turned off after an error
        self._errors: list[str] = []
        self._steps = None
        try:
            self._steps = open(self.dir / "steps.jsonl", "a", buffering=1)  # noqa: SIM115
        except Exception as e:
            self._fail("steps", e)
        self._obs_step: list[int] = []
        self._t_request: list[float] = []
        self._t_receive: list[float] = []
        self._latency_steps: list[int] = []
        self._chunks: list[np.ndarray] = []

    def _fail(self, sink: str, exc: Exception) -> None:
        if sink not in self._disabled:
            self._disabled.add(sink)
            logger.error("recorder: %s write failed (%r); %s recording disabled", sink, exc, sink)
        self._errors.append(f"{sink}: {exc!r}")

    def step(self, record: dict) -> None:
        with self._lock:
            if self._finished or "steps" in self._disabled or self._steps is None:
                return
            try:
                self._steps.write(_dumps(record) + "\n")
            except Exception as e:
                self._fail("steps", e)

    def chunk(self, obs_step: int, t_request: float, t_receive: float, latency_steps: int, chunk: np.ndarray) -> None:
        with self._lock:
            if self._finished:
                return
            try:
                arr = np.array(chunk, dtype=np.float32)
                if arr.ndim != 2 or (self._chunks and arr.shape != self._chunks[0].shape):
                    expected = self._chunks[0].shape if self._chunks else "(horizon, dim)"
                    logger.error("recorder: rejecting chunk of shape %s (expected %s)", arr.shape, expected)
                    self._errors.append(f"chunks: rejected shape {arr.shape} at obs_step {obs_step}")
                    return
                row = (int(obs_step), float(t_request), float(t_receive), int(latency_steps))
            except Exception as e:
                self._fail("chunks", e)
                return
            self._obs_step.append(row[0])
            self._t_request.append(row[1])
            self._t_receive.append(row[2])
            self._latency_steps.append(row[3])
            self._chunks.append(arr)
            if "chunks" in self._disabled:
                return
            try:
                # No fsync here: os.replace already survives a process crash and the control loop must not
                # wait on disk. `finish` does the fsync'd final write.
                self._save_chunks(sync=False)
            except Exception as e:
                self._fail("chunks", e)

    def images(self, step: int, images: dict[str, np.ndarray]) -> None:
        if step % self.image_every != 0:
            return
        with self._lock:
            if self._finished or "images" in self._disabled:
                return
            try:
                for camera, arr in images.items():
                    cam_dir = self.dir / "images" / camera
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    frame = Image.fromarray(np.asarray(arr, dtype=np.uint8))
                    frame.save(cam_dir / f"step_{step:06d}.jpg", quality=JPEG_QUALITY)
            except Exception as e:
                self._fail("images", e)

    def finish(self, outcome: Outcome, note: str, stats: dict, stream_stats: dict) -> None:
        """Write result.json and stream_stats.json; every write is attempted even if another fails."""
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        with self._lock:
            if self._finished:
                return
            try:
                if self._steps is not None:
                    try:
                        self._steps.close()
                    except Exception as e:
                        self._errors.append(f"steps: close {e!r}")
                try:
                    self._save_chunks(sync=True)
                except Exception as e:
                    self._errors.append(f"chunks: final save {e!r}")
                    logger.exception("recorder: final chunks.npz write failed")
                try:
                    _write_json(self.dir / "stream_stats.json", stream_stats)
                except Exception as e:
                    self._errors.append(f"stream_stats: {e!r}")
                    logger.exception("recorder: stream_stats.json write failed")
                result = {
                    "outcome": outcome,
                    "note": note,
                    "stats": stats,
                    "n_chunks": len(self._chunks),
                    "finished_wall": time.time(),
                    "recorder_errors": self._errors,
                }
                try:
                    _write_json(self.dir / "result.json", result)
                except Exception:
                    logger.exception("recorder: result.json write failed")
            finally:
                self._finished = True

    def _save_chunks(self, *, sync: bool) -> None:
        """Rewrite chunks.npz atomically (caller holds the lock)."""
        chunks = np.stack(self._chunks) if self._chunks else np.zeros((0, 0, 0), dtype=np.float32)
        arrays = {
            "obs_step": np.asarray(self._obs_step, dtype=np.int64),
            "t_request": np.asarray(self._t_request, dtype=np.float64),
            "t_receive": np.asarray(self._t_receive, dtype=np.float64),
            "latency_steps": np.asarray(self._latency_steps, dtype=np.int64),
            "chunks": chunks,
        }
        _write_atomic(self.dir / "chunks.npz", lambda f: np.savez(f, **arrays), sync=sync)
