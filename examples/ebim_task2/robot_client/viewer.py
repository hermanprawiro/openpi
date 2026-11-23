"""Read-only web viewer for the robot client's run folders (live while a run is in progress, and for replay).

    python3 -m robot_client.viewer --runs <dir> [--host 127.0.0.1] [--port 8765]

Serves a static page from `viewer_static/` and a small JSON API over `<runs>/<run>/episode_NNN/...`. GET only: the
server never creates, modifies or deletes anything. Files of a live run may be half-written; every reader here
tolerates that (see `viewer_data`). Imports stdlib, numpy and the viewer modules only: no rclpy, no recorder/config.
"""

import argparse
import contextlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import sys
import threading
from urllib.parse import parse_qs
from urllib.parse import unquote
from urllib.parse import urlsplit

import numpy as np

from robot_client import labels
from robot_client import viewer_align
from robot_client import viewer_data

logger = logging.getLogger("robot_client.viewer")

STATIC_DIR = Path(__file__).resolve().parent / "viewer_static"
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # the CLI's --run-name rule
EPISODE_RE = re.compile(r"^episode_\d{3}$")
STEP_RE = re.compile(r"^\d{1,9}$")
STATIC_RE = re.compile(r"^[a-z_]+\.(html|css|js)$")
CAMERAS = ("head", "wrist_left", "wrist_right")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}
JPEG_EOI = b"\xff\xd9"
DEFAULT_ALIGN = {"mode": "sync", "execute_horizon": 30, "active_arms": ["left", "right"]}


class _NotFoundError(Exception):
    pass


class _Episode:
    """The incremental readers of one episode, plus the last alignment (recomputed only when its inputs change)."""

    def __init__(self, ep_dir: Path):
        self.lock = threading.Lock()
        self.steps = viewer_data.StepsTail(ep_dir / "steps.jsonl")
        self.chunks = viewer_data.ChunksCache(ep_dir / "chunks.npz")
        self._aligned_for: tuple[int, dict | None, dict] | None = None  # (n steps, chunks, align args)
        self._alignment: viewer_align.Alignment | None = None

    def read(self, align_args: dict) -> tuple[list[dict], dict | None, viewer_align.Alignment]:
        """steps, chunks, alignment. `steps.jsonl` is read before `chunks.npz`: chunks are at least as new."""
        with self.lock:
            steps = self.steps.read()
            chunks = self.chunks.read()
            same = self._aligned_for
            if same is None or same[0] != len(steps) or same[1] is not chunks or same[2] != align_args:
                self._alignment = viewer_align.align(steps, chunks, **align_args)
                self._aligned_for = (len(steps), chunks, dict(align_args))
            return steps, chunks, self._alignment


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, runs: Path):
        super().__init__(address, handler)
        self.runs = Path(runs)
        self.verbose = False
        self._episodes: dict[Path, _Episode] = {}
        self._episodes_lock = threading.Lock()

    def episode(self, ep_dir: Path) -> _Episode:
        with self._episodes_lock:
            if ep_dir not in self._episodes:
                self._episodes[ep_dir] = _Episode(ep_dir)
            return self._episodes[ep_dir]


def _json_bytes(obj) -> bytes:
    return json.dumps(viewer_data.sanitize(obj), allow_nan=False, separators=(",", ":")).encode()


def _from_param(query: dict) -> int:
    try:
        return max(0, int(query.get("from", ["0"])[0]))
    except ValueError:
        return 0


def _read_json_file(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _scalar_safety(safety) -> dict:
    """Only the scalar thresholds: `joint_limits` (nested, large) stays out of the response."""
    if not isinstance(safety, dict):
        return {}
    return {k: v for k, v in safety.items() if not isinstance(v, dict | list | tuple)}


class Handler(BaseHTTPRequestHandler):
    server: ViewerServer
    server_version = "EbimRunViewer"

    # ---- plumbing ----

    def log_message(self, format, *args):
        if self.server.verbose:
            super().log_message(format, *args)

    def _send(self, status: int, content_type: str, body: bytes, *, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status: int = 200) -> None:
        self._send(status, "application/json", _json_bytes(obj))

    def _not_found(self) -> None:
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _method_not_allowed(self) -> None:
        body = _json_bytes({"error": "method not allowed"})
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _method_not_allowed  # noqa: N815

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route()
        except _NotFoundError:
            self._not_found()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.exception("viewer: unexpected error handling GET %s", self.path)
            with contextlib.suppress(OSError):
                self._send_json({"error": "internal"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    # ---- routing ----

    def _route(self) -> None:
        split = urlsplit(self.path)
        parts = [unquote(p) for p in split.path.split("/")]
        query = parse_qs(split.query)
        if parts[:1] != [""]:
            raise _NotFoundError
        parts = parts[1:]
        if parts == [""]:
            return self._static("index.html")
        if len(parts) == 2 and parts[0] == "static":
            return self._static(parts[1])
        if parts[:1] == ["api"]:
            return self._api(parts[1:], query)
        if parts[:1] == ["img"]:
            return self._image(parts[1:])
        raise _NotFoundError

    def _static(self, name: str) -> None:
        if not STATIC_RE.match(name):
            raise _NotFoundError
        path = (STATIC_DIR / name).resolve()
        if not path.is_relative_to(STATIC_DIR.resolve()) or not path.is_file():
            raise _NotFoundError
        try:
            body = path.read_bytes()
        except OSError:
            raise _NotFoundError from None
        self._send(200, CONTENT_TYPES[path.suffix], body)

    # ---- path validation ----

    def _run_dir(self, run: str) -> Path:
        if not RUN_NAME_RE.match(run) or run in (".", ".."):
            raise _NotFoundError
        root = self.server.runs.resolve()
        run_dir = (root / run).resolve()
        if not run_dir.is_relative_to(root) or run_dir == root or not run_dir.is_dir():
            raise _NotFoundError
        return run_dir

    def _episode_dir(self, run: str, ep: str) -> Path:
        run_dir = self._run_dir(run)
        if not EPISODE_RE.match(ep):
            raise _NotFoundError
        ep_dir = (run_dir / ep).resolve()
        if not ep_dir.is_relative_to(self.server.runs.resolve()) or not ep_dir.is_dir():
            raise _NotFoundError
        return ep_dir

    # ---- API ----

    def _api(self, parts: list[str], query: dict) -> None:
        if parts == ["labels"]:
            return self._send_json(labels.as_json())
        if parts == ["runs"]:
            return self._send_json({"runs": viewer_data.list_runs(self.server.runs)})
        if len(parts) == 2 and parts[0] == "runs":
            return self._send_json(self._run_info(self._run_dir(parts[1])))
        if len(parts) == 4 and parts[0] == "runs":
            ep_dir = self._episode_dir(parts[1], parts[2])
            run_dir = ep_dir.parent
            if parts[3] == "steps":
                return self._send_json(self._steps(run_dir, ep_dir, _from_param(query)))
            if parts[3] == "chunks":
                return self._send_json(self._chunks(run_dir, ep_dir, _from_param(query)))
            if parts[3] == "summary":
                return self._send_json(self._summary(run_dir, ep_dir))
        raise _NotFoundError

    @staticmethod
    def _align_args(run_dir: Path) -> dict:
        cfg = viewer_data.config_summary(run_dir)
        if cfg is None:
            return dict(DEFAULT_ALIGN)
        return {
            "mode": cfg["mode"] or DEFAULT_ALIGN["mode"],
            "execute_horizon": cfg["execute_horizon"] or DEFAULT_ALIGN["execute_horizon"],
            "active_arms": list(cfg["active_arms"] or DEFAULT_ALIGN["active_arms"]),
        }

    def _run_info(self, run_dir: Path) -> dict:
        events = viewer_data.read_events(run_dir)
        live = viewer_data.run_is_live(run_dir)
        config = viewer_data.config_summary(run_dir)
        if config is not None:
            config = {**config, "safety": _scalar_safety(config["safety"])}
        server_meta = _read_json_file(run_dir / "server_metadata.json")
        return {
            "name": run_dir.name,
            "config": config,
            "server": server_meta if isinstance(server_meta, dict) else {},
            "run_events": viewer_data.run_events(events),
            "episodes": viewer_data.episode_list(run_dir, events, live),
            "check_obs": viewer_data.check_obs_cameras(run_dir),
            "live": live,
        }

    def _read_episode(self, run_dir: Path, ep_dir: Path):
        return self.server.episode(ep_dir).read(self._align_args(run_dir))

    def _steps(self, run_dir: Path, ep_dir: Path, first: int) -> dict:
        steps, _, alignment = self._read_episode(run_dir, ep_dir)
        out = []
        for record, (chunk_k, row, clamped) in zip(steps, alignment.per_step, strict=True):
            step = record.get("step")
            if isinstance(step, int) and step >= first:
                out.append({**record, "chunk_k": chunk_k, "row": row, "clamped": bool(clamped)})
        return {"steps": out}

    def _chunks(self, run_dir: Path, ep_dir: Path, first: int) -> dict:
        _, chunks, _ = self._read_episode(run_dir, ep_dir)
        out = []
        if chunks is not None:
            t_request = np.asarray(chunks["t_request"], dtype=np.float64)
            t_receive = np.asarray(chunks["t_receive"], dtype=np.float64)
            for k in range(first, int(chunks["chunks"].shape[0])):
                latency_ms = (t_receive[k] - t_request[k]) * 1000.0  # NaN if either is NaN -> null
                out.append(
                    {
                        "k": k,
                        "obs_step": chunks["obs_step"][k],
                        "t_request": t_request[k],
                        "t_receive": t_receive[k],
                        "latency_ms": latency_ms,
                        "latency_steps": chunks["latency_steps"][k],
                        "rows": chunks["chunks"][k],
                    }
                )
        return {"chunks": out}

    def _summary(self, run_dir: Path, ep_dir: Path) -> dict:
        _, _, alignment = self._read_episode(run_dir, ep_dir)
        events = viewer_data.read_events(run_dir)
        episodes = viewer_data.episode_list(run_dir, events, viewer_data.run_is_live(run_dir))
        entry = next((e for e in episodes if e["name"] == ep_dir.name), None)
        return {
            "episode": entry,
            "alignment": alignment.to_json(),
            "images": viewer_data.image_index(ep_dir),
            "result": _read_json_file(ep_dir / "result.json"),
            "stream_stats": _read_json_file(ep_dir / "stream_stats.json"),
        }

    # ---- images ----

    def _image(self, parts: list[str]) -> None:
        if len(parts) == 3 and parts[1] == "check_obs":
            run, camera = parts[0], parts[2]
            if camera not in CAMERAS:
                raise _NotFoundError
            return self._send_jpeg(self._run_dir(run) / f"check_obs_{camera}.jpg")
        if len(parts) == 4:
            run, ep, camera, step = parts
            if camera not in CAMERAS or not STEP_RE.match(step):
                raise _NotFoundError
            ep_dir = self._episode_dir(run, ep)
            return self._send_jpeg(ep_dir / "images" / camera / f"step_{int(step):06d}.jpg")
        raise _NotFoundError

    def _send_jpeg(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.server.runs.resolve()):
            raise _NotFoundError
        try:
            body = resolved.read_bytes()
        except OSError:
            raise _NotFoundError from None
        if not body.endswith(JPEG_EOI):  # still being written (the recorder saves non-atomically): retry later
            raise _NotFoundError
        self._send(200, "image/jpeg", body, cache="no-cache")


def make_server(runs: Path, host: str, port: int) -> ThreadingHTTPServer:
    """A viewer server for the run folders under `runs` (which need not exist yet). Port 0 picks a free port."""
    return ViewerServer((host, port), Handler, Path(runs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m robot_client.viewer", description="Read-only web viewer for robot client run folders."
    )
    parser.add_argument("--runs", type=Path, required=True, help="directory holding the run folders")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8765, help="TCP port (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        server = make_server(args.runs, args.host, args.port)
    except OSError as e:
        print(f"cannot listen on {args.host}:{args.port}: {e}", file=sys.stderr)
        return 1
    server.verbose = args.verbose
    host, port = server.server_address[:2]
    print(f"Viewer on http://{host}:{port}/ reading {args.runs} (read-only; Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
