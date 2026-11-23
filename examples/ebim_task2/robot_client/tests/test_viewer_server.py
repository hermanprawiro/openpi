import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from robot_client import viewer
from robot_client import viewer_data
from viewer_fixture import make_run

STATIC = Path(viewer.__file__).parent / "viewer_static"


class Client:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def raw(self, path, method="GET"):
        req = urllib.request.Request(self.base + path, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, dict(e.headers), e.read()

    def get(self, path):
        status, _, body = self.raw(path)
        return status, json.loads(body)


@pytest.fixture
def serve(tmp_path):
    started = []

    def _serve(root=None):
        server = viewer.make_server(tmp_path if root is None else root, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, thread))
        return Client(server.server_address[1])

    yield _serve
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_api_runs_and_run(tmp_path, serve):
    make_run(tmp_path)
    client = serve()

    status, body = client.get("/api/runs")
    assert status == 200
    assert [r["name"] for r in body["runs"]] == ["run_test"]

    status, run = client.get("/api/runs/run_test")
    assert status == 200
    assert run["name"] == "run_test"
    assert run["config"]["mode"] == "sync"
    assert run["server"]["checkpoint"] == "test/0"
    assert [e["name"] for e in run["episodes"]] == ["episode_000"]
    assert run["check_obs"] == []
    assert run["live"] is False
    assert isinstance(run["run_events"], list)
    # only scalar safety thresholds, not the large joint_limits table
    assert "joint_limits" not in run["config"]["safety"]
    assert run["config"]["safety"]["max_step_rad"] == 0.10
    assert all(not isinstance(v, dict | list) for v in run["config"]["safety"].values())


def test_api_labels(serve):
    status, labels = serve().get("/api/labels")
    assert status == 200
    assert len(labels) == 16
    assert labels[0]["index"] == 0


def test_api_steps_incremental(tmp_path, serve):
    make_run(tmp_path)
    client = serve()
    status, body = client.get("/api/runs/run_test/episode_000/steps?from=0")
    assert status == 200
    steps = body["steps"]
    assert len(steps) > 60
    assert all({"chunk_k", "row", "clamped"} <= s.keys() for s in steps)
    assert any(s["chunk_k"] is not None for s in steps)

    _, tail = client.get("/api/runs/run_test/episode_000/steps?from=50")
    assert tail["steps"][0]["step"] == 50
    assert tail["steps"] == [s for s in steps if s["step"] >= 50]
    _, none = client.get("/api/runs/run_test/episode_000/steps")
    assert none["steps"] == steps
    status, _ = client.get("/api/runs/run_test/episode_000/steps?from=abc")
    assert status == 200  # a bad from is treated as 0


def test_api_chunks_and_summary(tmp_path, serve):
    make_run(tmp_path)
    make_run(tmp_path, "run_nan", nan_t_request=True)
    client = serve()

    _, body = client.get("/api/runs/run_test/episode_000/chunks?from=0")
    chunks = body["chunks"]
    assert len(chunks) >= 2
    c0 = chunks[0]
    assert set(c0) == {"k", "obs_step", "t_request", "t_receive", "latency_ms", "latency_steps", "rows"}
    assert np.asarray(c0["rows"]).shape == (50, 16)
    _, later = client.get("/api/runs/run_test/episode_000/chunks?from=1")
    assert [c["k"] for c in later["chunks"]] == [c["k"] for c in chunks if c["k"] >= 1]

    status, summary = client.get("/api/runs/run_test/episode_000/summary")
    assert status == 200
    assert summary["alignment"]["mismatches"] == 0
    assert summary["episode"]["name"] == "episode_000"
    assert "head" in summary["images"]
    assert summary["result"] is not None
    assert "streamer" in summary["stream_stats"]

    status, headers, raw = client.raw("/api/runs/run_nan/episode_000/chunks?from=0")
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    assert b"NaN" not in raw
    first = json.loads(raw)["chunks"][0]
    assert first["t_request"] is None
    assert first["latency_ms"] is None
    assert b"NaN" not in client.raw("/api/runs/run_nan/episode_000/summary")[2]


def test_api_empty_episode(tmp_path, serve):
    ep = tmp_path / "run_empty" / "episode_000"
    ep.mkdir(parents=True)
    (ep / "steps.jsonl").write_text("")
    (ep / "chunks.npz").write_bytes(b"")
    client = serve()
    assert client.get("/api/runs/run_empty/episode_000/steps?from=0") == (200, {"steps": []})
    assert client.get("/api/runs/run_empty/episode_000/chunks?from=0") == (200, {"chunks": []})
    status, summary = client.get("/api/runs/run_empty/episode_000/summary")
    assert status == 200
    assert summary["alignment"]["chunks"] == []
    assert summary["result"] is None
    assert summary["stream_stats"] is None
    status, run = client.get("/api/runs/run_empty")
    assert status == 200
    assert run["config"] is None  # no config.yaml


def test_images(tmp_path, serve):
    run = make_run(tmp_path, check_obs=True)
    client = serve()
    status, headers, body = client.raw("/img/run_test/episode_000/head/0")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert body.endswith(b"\xff\xd9")
    assert client.raw("/img/run_test/episode_000/head/99999")[0] == 404
    status, headers, _ = client.raw("/img/run_test/check_obs/head")
    assert (status, headers["Content-Type"]) == (200, "image/jpeg")
    assert client.raw("/img/run_test/check_obs/nope")[0] == 404

    # a half-written JPEG (no EOI marker) is not served
    jpg = next((run / "episode_000" / "images" / "head").glob("step_*.jpg"))
    step = int(re.search(r"step_(\d+)", jpg.name).group(1))
    jpg.write_bytes(jpg.read_bytes()[:-20])
    assert client.raw(f"/img/run_test/episode_000/head/{step}")[0] == 404


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs/..",
        "/api/runs/%2e%2e/episode_000/steps",
        "/api/runs/run_test/episode_0/steps",
        "/api/runs/run_test/episode_000/other",
        "/img/run_test/episode_000/../x/0",
        "/img/run_test/episode_000/cam/0",
        "/img/run_test/episode_000/head/abc",
        "/img/run_test/episode_000/head/1234567890",
        "/static/../viewer.py",
        "/static/%2e%2e/viewer.py",
        "/static/viewer.py",
        "/static/Index.html",
        "/static/missing.js",
        "/api/runs/link/episode_000/steps",
        "/api/runs/link",
        "/api/runs/missing",
        "/nothing",
    ],
)
def test_path_traversal_rejected(tmp_path, serve, path):
    make_run(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    (outside / "episode_000").mkdir(parents=True)
    os.symlink(outside, tmp_path / "link")
    client = serve()
    status, headers, body = client.raw(path)
    assert status == 404
    assert json.loads(body) == {"error": "not found"}
    assert headers["Content-Type"].startswith("application/json")


def test_non_get_is_405(tmp_path, serve):
    make_run(tmp_path)
    client = serve()
    for method in ("POST", "PUT", "DELETE"):
        assert client.raw("/api/runs", method=method)[0] == 405


def test_runs_dir_appears_later(tmp_path, serve):
    root = tmp_path / "later"
    client = serve(root)
    assert client.get("/api/runs") == (200, {"runs": []})
    make_run(root)
    status, body = client.get("/api/runs")
    assert [r["name"] for r in body["runs"]] == ["run_test"]
    assert client.get("/api/runs/run_test")[0] == 200


def test_static_index(serve):
    client = serve()
    status, headers, body = client.raw("/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"EBIM task2 run viewer" in body
    assert client.raw("/static/index.html")[0] == 200
    files = [p for p in STATIC.rglob("*") if p.is_file()]
    assert files
    for p in files:
        assert not re.search(r"https?://", p.read_text()), p


def test_static_js_content_type(tmp_path, serve, monkeypatch):
    (tmp_path / "app.js").write_text("export const x = 1;\n")
    monkeypatch.setattr(viewer, "STATIC_DIR", tmp_path)
    status, headers, _ = serve().raw("/static/app.js")
    assert status == 200
    assert headers["Content-Type"].startswith("text/javascript")


def test_main_args(capsys):
    with pytest.raises(SystemExit) as exc:
        viewer.main(["--help"])
    assert exc.value.code == 0
    args = viewer.build_parser().parse_args(["--runs", "x"])
    assert (args.host, args.port) == ("127.0.0.1", 8765)
    assert "--verbose" in capsys.readouterr().out


def test_unexpected_error_is_500_and_server_survives(tmp_path, serve, monkeypatch, caplog):
    make_run(tmp_path)
    client = serve()

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(viewer.viewer_data, "list_runs", boom)
    status, body = client.get("/api/runs")
    assert (status, body) == (500, {"error": "internal"})
    monkeypatch.undo()
    assert client.get("/api/runs")[0] == 200


def test_no_forbidden_imports():
    code = (
        "import robot_client.viewer, sys; "
        "print(sorted(m for m in sys.modules if m.startswith(('rclpy', 'eval_client', "
        "'robot_client.recorder', 'robot_client.config'))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(viewer.__file__).parent.parent,
    )
    assert out.stdout.strip() == "[]"


# ---- Task 2 hardening, in viewer_data ----


def test_config_summary_odd_pin_returns_none(tmp_path):
    (tmp_path / "config.yaml").write_text("state:\n  pin:\n    notanint: 1.0\n")
    assert viewer_data.config_summary(tmp_path) is None
    (tmp_path / "config.yaml").write_text("state:\n  pin: [1, 2]\n")
    assert viewer_data.config_summary(tmp_path) is None


def test_chunks_cache_other_failures_keep_previous(tmp_path, monkeypatch):
    path = tmp_path / "chunks.npz"
    np.savez(
        path,
        obs_step=np.zeros(1, int),
        t_request=np.zeros(1),
        t_receive=np.zeros(1),
        latency_steps=np.zeros(1, int),
        chunks=np.zeros((1, 50, 16), np.float32),
    )
    cache = viewer_data.ChunksCache(path)
    first = cache.read()
    assert first is not None
    # replaced by an npz that lacks the keys -> KeyError -> previous value kept
    np.savez(path, other=np.zeros(3))
    assert cache.read() is first
    # EOFError / zlib.error are failed reads too
    import zlib

    for exc in (EOFError("x"), zlib.error("y")):
        os.utime(path, ns=(1, 1 + id(exc) % 1000))

        def bad(self, exc=exc):
            raise exc

        monkeypatch.setattr(viewer_data.ChunksCache, "_load", bad)
        assert cache.read() is first
