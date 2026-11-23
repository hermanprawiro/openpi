import json
import os
import sys
import time
import zipfile

import numpy as np
from robot_client import viewer_data
from viewer_fixture import make_run


def test_fixture_sanity(tmp_path):
    run = make_run(tmp_path)
    steps = [json.loads(line) for line in (run / "episode_000" / "steps.jsonl").read_text().splitlines()]
    modes = [s["mode"] for s in steps]
    assert modes[:3] == ["wait"] * 3
    assert modes[3] == "run"
    with np.load(run / "episode_000" / "chunks.npz") as z:
        assert z["chunks"].shape[0] >= 2


def test_list_runs_kinds_and_order(tmp_path):
    first = make_run(tmp_path, "run_a")
    second = make_run(tmp_path, "run_b", check_obs=True, episodes=False)
    (tmp_path / "fake_robot_events.jsonl").write_text("{}\n")
    os.utime(first, (1000, 1000))
    os.utime(second, (2000, 2000))

    runs = viewer_data.list_runs(tmp_path)

    assert [(r["name"], r["kind"]) for r in runs] == [("run_b", "check-obs"), ("run_a", "episodes")]
    assert runs[0]["mtime"] == 2000
    assert viewer_data.list_runs(tmp_path / "missing") == []


def test_list_runs_other_kind(tmp_path):
    (tmp_path / "run_x").mkdir()
    assert [r["kind"] for r in viewer_data.list_runs(tmp_path)] == ["other"]


def test_live_flag(tmp_path):
    live = make_run(tmp_path, "run_live", end=None)
    now = time.time()

    assert viewer_data.list_runs(tmp_path, now)[0]["live"] is True
    assert viewer_data.list_runs(tmp_path, now + 600)[0]["live"] is True  # silent at the start gate
    assert viewer_data.list_runs(tmp_path, now + 7201)[0]["live"] is False  # a crashed client leaves no run_end

    done = make_run(tmp_path, "run_done")
    assert not next(r for r in viewer_data.list_runs(tmp_path) if r["name"] == done.name)["live"]
    assert live.exists()


def test_steps_tail_partial_line(tmp_path):
    path = tmp_path / "steps.jsonl"
    path.write_text('{"step": 0}\n{"step": 1}\n{"step": 2, "mo')
    tail = viewer_data.StepsTail(path)

    assert tail.read() == [{"step": 0}, {"step": 1}]
    with open(path, "a") as f:
        f.write('de": "run"}\n')
    records = tail.read()
    assert len(records) == 3
    assert records[2] == {"step": 2, "mode": "run"}
    path.write_text("")
    assert tail.read() == []


def test_steps_tail_missing_file(tmp_path):
    assert viewer_data.StepsTail(tmp_path / "nope.jsonl").read() == []


def test_chunks_cache_reload_and_retry(tmp_path, monkeypatch):
    path = tmp_path / "episode_000" / "chunks.npz"
    cache = viewer_data.ChunksCache(path)
    assert cache.read() is None

    run = make_run(tmp_path, "run_c")
    cache = viewer_data.ChunksCache(run / "episode_000" / "chunks.npz")
    data = cache.read()
    assert data["chunks"].shape[1:] == (50, 16)
    n = data["chunks"].shape[0]
    assert n >= 2
    for key in ("obs_step", "t_request", "t_receive", "latency_steps"):
        assert len(data[key]) == n

    # Forces a reload on the next read, and the first np.load raises once: the retry still gets the data.
    real_load = np.load
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise zipfile.BadZipFile("half-written")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", flaky)
    bump = cache.path.stat().st_mtime_ns + 10**9
    os.utime(cache.path, ns=(bump, bump))
    again = cache.read()
    assert len(calls) == 2
    assert again["chunks"].shape == data["chunks"].shape

    # Always failing: the previous value is kept.
    def broken(*args, **kwargs):
        raise zipfile.BadZipFile("always")

    monkeypatch.setattr(np, "load", broken)
    bump += 10**9
    os.utime(cache.path, ns=(bump, bump))
    assert cache.read() is again


def test_chunks_cache_empty_is_none(tmp_path):
    path = tmp_path / "chunks.npz"
    empty = np.zeros(0)
    np.savez(path, obs_step=empty, t_request=empty, t_receive=empty, latency_steps=empty, chunks=np.zeros((0, 0, 0)))
    assert viewer_data.ChunksCache(path).read() is None


def test_episode_without_chunks(tmp_path):
    run = tmp_path / "run_e"
    ep = run / "episode_000"
    ep.mkdir(parents=True)
    (ep / "steps.jsonl").write_text("")
    assert viewer_data.StepsTail(ep / "steps.jsonl").read() == []
    assert viewer_data.ChunksCache(ep / "chunks.npz").read() is None

    episodes = viewer_data.episode_list(run, [], run_live=False)

    assert len(episodes) == 1
    assert episodes[0]["name"] == "episode_000"
    assert episodes[0]["index"] == 0
    assert episodes[0]["ended_by"] is None
    assert episodes[0]["has_result"] is False


def test_sanitize_nan():
    out = viewer_data.sanitize({"a": np.float32("nan"), "b": [np.inf, 1.5], "c": np.arange(2)})

    assert out == {"a": None, "b": [None, 1.5], "c": [0, 1]}
    json.dumps(out, allow_nan=False)


def test_sanitize_nan_array_and_scalars():
    out = viewer_data.sanitize({"x": np.array([1.0, np.nan]), "n": np.int64(3), "t": (1, float("-inf"))})

    assert out == {"x": [1.0, None], "n": 3, "t": [1, None]}
    json.dumps(out, allow_nan=False)


def test_config_summary(tmp_path, monkeypatch):
    run = make_run(tmp_path)

    summary = viewer_data.config_summary(run)

    assert summary["mode"] == "sync"
    assert summary["execute_horizon"] == 30
    assert summary["pin"] == {7: 0.0}
    assert summary["active_arms"] == ["left", "right"]
    assert isinstance(summary["safety"], dict)
    assert "episode_timeout_s" in summary["safety"]

    monkeypatch.setitem(sys.modules, "yaml", None)
    assert viewer_data.config_summary(run) is None


def test_config_summary_missing(tmp_path):
    (tmp_path / "run_z").mkdir()
    assert viewer_data.config_summary(tmp_path / "run_z") is None


def test_episode_list_endings(tmp_path):
    done = make_run(tmp_path, "run_op", end="operator")
    ep = viewer_data.episode_list(done, viewer_data.read_events(done), run_live=False)[0]
    assert ep["outcome"] == "s"
    assert ep["ended_by"] == "operator"
    assert ep["has_result"] is True
    assert ep["live"] is False
    assert ep["dry_run"] is False

    cut = make_run(tmp_path, "run_int", end="interrupted")
    ep = viewer_data.episode_list(cut, viewer_data.read_events(cut), run_live=False)[0]
    assert ep["ended_by"] == "interrupted"
    assert ep["outcome"] == "freeze"
    assert "interrupted" in ep["reason"]

    frozen = make_run(tmp_path, "run_fr", end="freeze")
    ep = viewer_data.episode_list(frozen, viewer_data.read_events(frozen), run_live=False)[0]
    assert ep["ended_by"] == "freeze"
    assert ep["outcome"] == "freeze"

    timed = make_run(tmp_path, "run_to", end="timeout")
    ep = viewer_data.episode_list(timed, viewer_data.read_events(timed), run_live=False)[0]
    assert ep["ended_by"] == "timeout"

    live = make_run(tmp_path, "run_lv", end=None)
    ep = viewer_data.episode_list(live, viewer_data.read_events(live), run_live=True)[0]
    assert ep["live"] is True
    assert ep["ended_by"] is None
    assert ep["outcome"] is None


def test_episode_list_dry_run(tmp_path):
    run = make_run(tmp_path, dry_run=True)
    ep = viewer_data.episode_list(run, viewer_data.read_events(run), run_live=False)[0]
    assert ep["dry_run"] is True


def test_run_events(tmp_path):
    run = tmp_path / "run_f"
    run.mkdir()
    lines = [
        {"wall": 1.0, "kind": "run_start"},
        {"wall": 2.0, "kind": "freeze_before_start", "episode": 0, "reason": "camera stale"},
        {"wall": 3.0, "kind": "episode_start", "episode": 0},
        {"wall": 4.0, "kind": "run_end"},
    ]
    (run / "events.log").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")

    events = viewer_data.read_events(run)
    assert len(events) == 4  # the unparsable line is skipped
    listed = viewer_data.run_events(events)
    assert [e["kind"] for e in listed] == ["freeze_before_start"]
    assert listed[0]["reason"] == "camera stale"
    assert viewer_data.episode_list(run, events, run_live=False) == []


def test_read_events_missing(tmp_path):
    assert viewer_data.read_events(tmp_path) == []


def test_image_index_and_check_obs(tmp_path):
    run = make_run(tmp_path, check_obs=True)
    index = viewer_data.image_index(run / "episode_000")
    assert set(index) == {"head", "wrist_left", "wrist_right"}
    steps = index["head"]
    assert steps == sorted(steps)
    assert steps[:3] == [0, 10, 20]
    assert viewer_data.check_obs_cameras(run) == ["head", "wrist_left", "wrist_right"]
    assert viewer_data.image_index(tmp_path / "nowhere") == {}


def test_fixture_nan_t_request_and_hold(tmp_path):
    run = make_run(tmp_path, nan_t_request=True, hold_at=(25, 35))
    data = viewer_data.ChunksCache(run / "episode_000" / "chunks.npz").read()
    assert np.isnan(data["t_request"][0])
    steps = viewer_data.StepsTail(run / "episode_000" / "steps.jsonl").read()
    assert {s["mode"] for s in steps[25:35]} == {"hold"}


# ---- a run folder reused by a second `run --run-name NAME` (events.log is append-only) ----


def _live_names(run):
    events = viewer_data.read_events(run)
    return {e["name"]: e["live"] for e in viewer_data.episode_list(run, events, viewer_data.run_is_live(run))}


def test_reused_run_folder_is_live_again(tmp_path):
    run = make_run(tmp_path, "run_r")  # run_start, episode_000, run_end
    assert viewer_data.run_is_live(run) is False

    make_run(tmp_path, "run_r", end=None, index=1)  # a second run_start, episode_001 still running

    kinds = [e["kind"] for e in viewer_data.read_events(run)]
    assert kinds.count("run_start") == 2
    assert kinds.count("run_end") == 1
    assert viewer_data.run_is_live(run) is True
    assert viewer_data.list_runs(tmp_path)[0]["live"] is True
    assert _live_names(run) == {"episode_000": False, "episode_001": True}

    with open(run / "events.log", "a") as f:
        f.write(json.dumps({"wall": time.time(), "kind": "run_end"}) + "\n")
    assert viewer_data.run_is_live(run) is False
    assert _live_names(run) == {"episode_000": False, "episode_001": False}


def test_reused_after_crash_old_episode_not_live(tmp_path):
    run = make_run(tmp_path, "run_c", end=None)  # killed client: no episode_end, no run_end
    make_run(tmp_path, "run_c", end=None, index=1)

    assert viewer_data.run_is_live(run) is True
    assert _live_names(run) == {"episode_000": False, "episode_001": True}


def test_reused_folder_at_start_gate(tmp_path):
    run = make_run(tmp_path, "run_g")
    with open(run / "events.log", "a") as f:
        f.write(json.dumps({"wall": time.time(), "kind": "run_start"}) + "\n")

    assert viewer_data.run_is_live(run) is True
    assert _live_names(run) == {"episode_000": False}


def test_interrupted_in_a_later_run_is_not_the_old_episode(tmp_path):
    run = make_run(tmp_path, "run_k", end=None)  # killed mid-episode
    with open(run / "events.log", "a") as f:
        for kind in ("run_start", "interrupted", "run_end"):
            f.write(json.dumps({"wall": time.time(), "kind": kind, "note": "x"}) + "\n")

    ep = viewer_data.episode_list(run, viewer_data.read_events(run), run_live=False)[0]
    assert ep["ended_by"] is None


def test_list_runs_started_and_age(tmp_path):
    run = make_run(tmp_path, "run_a", end=None)
    now = time.time()
    entry = viewer_data.list_runs(tmp_path, now + 30)[0]
    starts = [e["wall"] for e in viewer_data.read_events(run) if e["kind"] == "run_start"]
    assert entry["started"] == starts[-1]
    assert 29 <= entry["age_s"] <= 40
    (tmp_path / "run_x").mkdir()
    other = next(r for r in viewer_data.list_runs(tmp_path) if r["name"] == "run_x")
    assert other["started"] is None
    assert other["age_s"] is None
