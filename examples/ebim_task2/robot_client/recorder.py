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

Everything is flushed as it is written so a crash mid-episode still leaves usable files.
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


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
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
        self._events = open(self.dir / "events.log", "a", buffering=1)  # noqa: SIM115

    def event(self, kind: str, **fields) -> None:
        record = {"wall": time.time(), "mono": time.monotonic(), "kind": kind, **fields}
        line = _dumps(record)
        with self._lock:
            self._events.write(line + "\n")

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
        self._steps = open(self.dir / "steps.jsonl", "a", buffering=1)  # noqa: SIM115
        self._obs_step: list[int] = []
        self._t_request: list[float] = []
        self._t_receive: list[float] = []
        self._latency_steps: list[int] = []
        self._chunks: list[np.ndarray] = []

    def step(self, record: dict) -> None:
        line = _dumps(record)
        with self._lock:
            if not self._finished:
                self._steps.write(line + "\n")

    def chunk(self, obs_step: int, t_request: float, t_receive: float, latency_steps: int, chunk: np.ndarray) -> None:
        with self._lock:
            if self._finished:
                return
            self._obs_step.append(int(obs_step))
            self._t_request.append(float(t_request))
            self._t_receive.append(float(t_receive))
            self._latency_steps.append(int(latency_steps))
            self._chunks.append(np.asarray(chunk, dtype=np.float32).copy())
            self._save_chunks()

    def images(self, step: int, images: dict[str, np.ndarray]) -> None:
        if step % self.image_every != 0:
            return
        with self._lock:
            if self._finished:
                return
        for camera, arr in images.items():
            cam_dir = self.dir / "images" / camera
            cam_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(
                cam_dir / f"step_{step:06d}.jpg", quality=JPEG_QUALITY
            )

    def finish(self, outcome: Outcome, note: str, stats: dict, stream_stats: dict) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._steps.close()
            self._save_chunks()
            _write_json(self.dir / "stream_stats.json", stream_stats)
            _write_json(
                self.dir / "result.json",
                {
                    "outcome": outcome,
                    "note": note,
                    "stats": stats,
                    "n_chunks": len(self._chunks),
                    "finished_wall": time.time(),
                },
            )

    def _save_chunks(self) -> None:
        """Rewrite chunks.npz atomically (caller holds the lock); small enough to do after every chunk."""
        chunks = np.stack(self._chunks) if self._chunks else np.zeros((0, 0, 0), dtype=np.float32)
        arrays = {
            "obs_step": np.asarray(self._obs_step, dtype=np.int64),
            "t_request": np.asarray(self._t_request, dtype=np.float64),
            "t_receive": np.asarray(self._t_receive, dtype=np.float64),
            "latency_steps": np.asarray(self._latency_steps, dtype=np.int64),
            "chunks": chunks,
        }
        _write_atomic(self.dir / "chunks.npz", lambda f: np.savez(f, **arrays))
