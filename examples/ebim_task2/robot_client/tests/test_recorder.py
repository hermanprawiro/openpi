import json

import numpy as np
import pytest
from robot_client import recorder
from robot_client.recorder import RunRecorder
import yaml

SECRET = "sk-super-secret-key"


def _config(*, masked: bool = True) -> dict:
    return {"stream_hz": 100, "server": {"host": "h", "port": 8000, "api_key": "***" if masked else SECRET}}


def _img(value: int = 7) -> np.ndarray:
    return np.full((224, 224, 3), value, dtype=np.uint8)


def test_layout_of_run_dir(tmp_path):
    rec = RunRecorder(tmp_path, "run1", _config(), {"layout": "munich16"})
    rec.event("hello", n=np.int64(3), q=np.zeros(2))
    ep = rec.episode(2)
    ep.step({"step": 0, "q": [0.0]})
    ep.finish("s", "ok", {"a": 1}, {"b": 2})

    run = tmp_path / "run1"
    assert yaml.safe_load((run / "config.yaml").read_text())["server"]["api_key"] == "***"
    assert json.loads((run / "server_metadata.json").read_text()) == {"layout": "munich16"}
    line = json.loads((run / "events.log").read_text().splitlines()[0])
    assert (line["kind"], line["n"], line["q"]) == ("hello", 3, [0.0, 0.0])
    assert {"wall", "mono"} <= line.keys()
    ep_dir = run / "episode_002"
    for name in ("steps.jsonl", "chunks.npz", "result.json", "stream_stats.json"):
        assert (ep_dir / name).is_file(), name


def test_raw_api_key_never_written(tmp_path):
    RunRecorder(tmp_path, "run1", _config(masked=False), None)
    text = (tmp_path / "run1" / "config.yaml").read_text()
    assert SECRET not in text
    assert (tmp_path / "run1" / "server_metadata.json").is_file()


def test_images_at_2hz(tmp_path):
    ep = RunRecorder(tmp_path, "run1", _config(), None).episode(0)
    for step in range(40):
        ep.images(step, {"head": _img(), "wrist_left": _img(), "wrist_right": _img()})
    for cam in ("head", "wrist_left", "wrist_right"):
        files = sorted((tmp_path / "run1" / "episode_000" / "images" / cam).glob("*.jpg"))
        assert len(files) == 4, cam
        assert files[1].name == "step_000010.jpg"


def test_finish_idempotent_and_chunks_npz_shapes(tmp_path):
    ep = RunRecorder(tmp_path, "run1", _config(), None).episode(1)
    for i in range(3):
        ep.chunk(i * 30, 1.0 + i, 1.5 + i, 10, np.full((50, 16), i, dtype=np.float32))
    ep.finish("f", "first", {"x": 1}, {"y": 2})
    ep.finish("s", "second", {"x": 9}, {"y": 9})
    ep.step({"step": 99})  # after finish: ignored, must not raise

    ep_dir = tmp_path / "run1" / "episode_001"
    npz = np.load(ep_dir / "chunks.npz")
    assert npz["chunks"].shape == (3, 50, 16)
    assert npz["obs_step"].tolist() == [0, 30, 60]
    assert npz["t_request"].shape == npz["t_receive"].shape == npz["latency_steps"].shape == (3,)
    result = json.loads((ep_dir / "result.json").read_text())
    assert (result["outcome"], result["note"], result["stats"]) == ("f", "first", {"x": 1})
    assert json.loads((ep_dir / "stream_stats.json").read_text()) == {"y": 2}
    assert (ep_dir / "steps.jsonl").read_text() == ""


def test_crash_mid_episode_leaves_readable_files(tmp_path):
    """No finish(): steps and chunks written so far are already on disk."""
    ep = RunRecorder(tmp_path, "run1", _config(), None).episode(0)
    ep.step({"step": 0})
    ep.step({"step": 1})
    ep.chunk(0, 1.0, 1.1, 2, np.zeros((50, 16), dtype=np.float32))
    ep_dir = tmp_path / "run1" / "episode_000"
    assert len((ep_dir / "steps.jsonl").read_text().splitlines()) == 2
    assert np.load(ep_dir / "chunks.npz")["chunks"].shape == (1, 50, 16)


def test_bad_outcome_rejected(tmp_path):
    ep = RunRecorder(tmp_path, "run1", _config(), None).episode(0)
    with pytest.raises(ValueError, match="outcome"):
        ep.finish("bogus", "", {}, {})


def test_wrong_shape_chunk_rejected_without_poisoning(tmp_path):
    ep = RunRecorder(tmp_path, "run1", _config(), None).episode(0)
    ep.chunk(0, 1.0, 1.1, 2, np.zeros((50, 16)))
    ep.chunk(30, 2.0, 2.1, 2, np.zeros((50, 15)))  # wrong dim
    ep.chunk(40, 2.0, 2.1, 2, np.zeros(16))  # wrong ndim
    ep.chunk(60, 3.0, 3.1, 2, np.ones((50, 16)))
    ep.finish("s", "", {}, {})
    ep_dir = tmp_path / "run1" / "episode_000"
    npz = np.load(ep_dir / "chunks.npz")
    assert npz["chunks"].shape == (2, 50, 16)
    assert npz["obs_step"].tolist() == [0, 60]
    result = json.loads((ep_dir / "result.json").read_text())
    assert result["n_chunks"] == 2
    assert len(result["recorder_errors"]) == 2


def test_io_errors_never_raise_and_finish_still_writes_result(tmp_path, monkeypatch):
    run = RunRecorder(tmp_path, "run1", _config(), None)
    ep = run.episode(0)
    ep.step({"step": 0})

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("robot_client.recorder.Image.Image.save", boom)
    monkeypatch.setattr("robot_client.recorder._write_atomic", boom)

    class BadFile:
        def write(self, _):
            raise OSError("disk full")

        def close(self):
            pass

    ep._steps = BadFile()  # noqa: SLF001
    run._events = BadFile()  # noqa: SLF001
    run.event("x")
    run.event("y")  # disabled after the first failure, still silent
    ep.step({"step": 1})
    ep.step({"step": 2})
    ep.chunk(0, 1.0, 1.1, 2, np.zeros((50, 16)))
    ep.images(0, {"head": _img()})
    ep.images(10, {"head": _img()})

    monkeypatch.undo()
    ep.finish("x", "disk trouble", {"k": 1}, {"s": 1})
    ep_dir = tmp_path / "run1" / "episode_000"
    result = json.loads((ep_dir / "result.json").read_text())
    assert result["outcome"] == "x"
    assert any(e.startswith("steps") for e in result["recorder_errors"])
    assert any(e.startswith("chunks") for e in result["recorder_errors"])
    assert any(e.startswith("images") for e in result["recorder_errors"])
    assert json.loads((ep_dir / "stream_stats.json").read_text()) == {"s": 1}
    assert np.load(ep_dir / "chunks.npz")["chunks"].shape == (1, 50, 16)


def test_finish_attempts_result_when_other_writes_fail(tmp_path, monkeypatch):
    ep = RunRecorder(tmp_path, "run1", _config(), None).episode(0)
    real = recorder._write_atomic  # noqa: SLF001

    def selective(path, write, **kw):
        if path.name != "result.json":
            raise OSError("nope")
        real(path, write, **kw)

    monkeypatch.setattr(recorder, "_write_atomic", selective)
    ep.finish("f", "n", {}, {})
    ep.finish("s", "again", {}, {})  # idempotent even after a partly failed finish
    result = json.loads((tmp_path / "run1" / "episode_000" / "result.json").read_text())
    assert result["outcome"] == "f"
    assert len(result["recorder_errors"]) == 2
