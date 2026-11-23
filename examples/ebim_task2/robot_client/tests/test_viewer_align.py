"""Chunk alignment against episodes recorded by the real `Runner` (through `test_runner.Rig`) into a real
`EpisodeRecorder`, read back with `StepsTail` / `ChunksCache` exactly as the viewer reads them."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from robot_client.recorder import RunRecorder
from robot_client.viewer_align import align
from robot_client.viewer_data import ChunksCache
from robot_client.viewer_data import StepsTail
from test_runner import Rig
from viewer_fixture import make_run

H = 30  # inference.execute_horizon in robot_site.munich.yaml
ARMS = ["left", "right"]


def _read(ep_dir: Path):
    return StepsTail(ep_dir / "steps.jsonl").read(), ChunksCache(ep_dir / "chunks.npz").read()


def _episode(run_dir: Path):
    return _read(run_dir / "episode_000")


def _align(steps, chunks, mode="sync"):
    return align(steps, chunks, mode=mode, execute_horizon=H, active_arms=ARMS)


def _run_ticks(steps, al, k):
    return [s["step"] for s, (ck, _, _) in zip(steps, al.per_step, strict=True) if ck == k]


def _recorded_rig_episode(out_dir: Path, rig: Rig) -> Path:
    """Drive `rig`'s real Runner into a real EpisodeRecorder (for situations `make_run` cannot set up)."""
    rec = RunRecorder(out_dir, "run_rig", rig.cfg.redacted_dict(), {"layout": "munich16"})
    ep = rec.episode(0)
    rig.runner.recorder = ep
    outcome = rig.run()
    ep.finish("s", "", {"ended_by": outcome.ended_by}, {})
    rec.close()
    return ep.dir


def _assert_rows_match_actions(steps, chunks, al):
    """Every run tick is assigned and its recorded action equals the assigned chunk row (float32 tolerance)."""
    for s, (k, r, _) in zip(steps, al.per_step, strict=True):
        if s["mode"] == "run":
            assert k is not None, s["step"]
            np.testing.assert_allclose(s["action"], chunks["chunks"][k, r], atol=1e-5, err_msg=f"step {s['step']}")
        else:
            assert (k, r) == (None, None)


# ----------------------------------------------------------------------------------------------------
# Sync
# ----------------------------------------------------------------------------------------------------


def test_sync_basic(tmp_path):
    steps, chunks = _episode(make_run(tmp_path))
    al = _align(steps, chunks)

    # As test_runner.test_sync_executes_horizon_then_requests: latency 3, horizon 30.
    assert _run_ticks(steps, al, 0) == list(range(3, 33))
    assert _run_ticks(steps, al, 1) == list(range(36, 66))
    assert _run_ticks(steps, al, 2) == list(range(69, 99))
    assert [al.per_step[i][1] for i in range(3, 33)] == list(range(30))
    assert al.mismatches == 0

    c0, c1 = al.chunks[0], al.chunks[1]
    assert c0["k"] == 0
    assert c0["receipt_step"] == 3
    assert c0["receipt_recorded"] is True
    assert c0["receipt_approx"] is False
    assert c0["first"] is True
    assert c1["first"] is False
    assert c0["stale_rows"] == 0
    assert c0["executed"] == [3, 32]
    assert c0["n_executed"] == 30
    assert c0["n_clamped"] == 0
    assert c1["receipt_step"] == 36
    assert c0["start_offset"]["right"] == pytest.approx(0.0, abs=1e-6)
    assert set(c0["start_offset"]) == {"left", "right"}
    for s, (_, _, clamped) in zip(steps, al.per_step, strict=True):
        assert clamped is False, s["step"]


def test_sync_leash(tmp_path):
    """`make_run` cannot provoke a leash (FakeStreamer's leash ignores max_track_lag_rad), so this drives the real
    Runner through a Rig directly: the streamer reports the leash engaged on ticks 10-13 (mid chunk 0) and 40-41."""
    rig = Rig(respond=lambda n: (3, rig.chunk(tag=0.05 * n), None), keys={100: "q"}, timeout=6.0)
    leash = {*range(10, 14), 40, 41}
    rig.streamer.leash_fn = lambda tick, side: tick in leash
    steps, chunks = _read(_recorded_rig_episode(tmp_path, rig))
    al = _align(steps, chunks)

    assert [s["step"] for s in steps if s["mode"] == "leash"] == sorted(leash)
    for s, entry in zip(steps, al.per_step, strict=True):
        if s["mode"] == "leash":
            assert entry == (None, None, False)
    # Leash ticks pop nothing: chunk 0's 30 rows run on 3-9 and 14-36.
    assert _run_ticks(steps, al, 0) == [*range(3, 10), *range(14, 37)]
    assert al.chunks[0]["n_executed"] == 30
    assert al.mismatches == 0
    _assert_rows_match_actions(steps, chunks, al)


def test_hold_drops_queue(tmp_path):
    steps, chunks = _episode(make_run(tmp_path, hold_at=(25, 35)))
    al = _align(steps, chunks)

    assert al.chunks[0]["n_executed"] == 22
    assert al.chunks[0]["executed"] == [3, 24]
    nxt = al.chunks[1]
    assert nxt["receipt_step"] == 38
    assert nxt["first"] is True
    assert nxt["executed"] == [38, 67]
    assert al.chunks[2]["first"] is False
    assert al.mismatches == 0
    _assert_rows_match_actions(steps, chunks, al)


def test_start_check_freeze(tmp_path):
    helper = Rig()  # same config -> same start pose as make_run's rig
    run = make_run(tmp_path, respond=lambda n: (3, helper.chunk(tag=0.5), None))
    steps, chunks = _episode(run)
    al = _align(steps, chunks)

    assert chunks["chunks"].shape[0] == 1
    c0 = al.chunks[0]
    assert c0["receipt_recorded"] is False
    assert c0["receipt_step"] == steps[-1]["step"] + 1 == 3
    assert c0["n_executed"] == 0
    assert c0["executed"] is None
    assert c0["start_offset"]["right"] == pytest.approx(0.5, abs=1e-6)
    assert c0["start_offset"]["left"] == pytest.approx(0.5, abs=1e-6)
    assert c0["first"] is True
    assert al.mismatches == 0


def test_sync_detects_mismatch(tmp_path):
    steps, chunks = _episode(make_run(tmp_path))
    steps = copy.deepcopy(steps)
    steps[40]["action"][9] += 0.01
    assert steps[40]["mode"] == "run"

    al = _align(steps, chunks)

    assert al.mismatches == 1
    assert al.per_step[40][:2] == (1, 4)  # still assigned: the reconstruction does not depend on values


def test_sync_without_chunks_counts_unassigned_run_ticks(tmp_path):
    steps, _ = _episode(make_run(tmp_path))
    al = _align(steps, None)
    assert al.chunks == []
    assert al.mismatches == sum(s["mode"] == "run" for s in steps)
    assert all(entry[:2] == (None, None) for entry in al.per_step)


def test_clamped_flag(tmp_path):
    """Hand-built: one run record's commanded right waypoint differs from its action (as clamp_step would cut)."""
    steps, chunks = _episode(make_run(tmp_path))
    steps = copy.deepcopy(steps)
    i = 10
    assert steps[i]["mode"] == "run"
    steps[i]["waypoints"]["right"][2] += 0.01

    al = _align(steps, chunks)

    assert al.per_step[i] == (0, 7, True)
    assert sum(entry[2] for entry in al.per_step) == 1
    assert al.chunks[0]["n_clamped"] == 1
    assert al.chunks[1]["n_clamped"] == 0
    # An inactive arm's difference does not count.
    al_right_only = align(steps, chunks, mode="sync", execute_horizon=H, active_arms=["left"])
    assert al_right_only.per_step[i][2] is False


def test_clamped_flag_from_real_step_clamp(tmp_path):
    """Rows 0.15 rad apart exceed max_step_rad 0.10: the Runner cuts every waypoint after row 0."""
    helper = Rig()
    steps, chunks = _episode(make_run(tmp_path, respond=lambda n: (3, helper.chunk(d=0.15 if n == 0 else 0.0), None)))
    al = _align(steps, chunks)

    flags = {s["step"]: c for s, (_, _, c) in zip(steps, al.per_step, strict=True) if s["mode"] == "run"}
    assert flags[3] is False
    assert all(flags[t] for t in range(4, 33))
    assert al.chunks[0]["n_clamped"] == 29


def test_receipt_not_yet_recorded(tmp_path):
    """Hand-built: a live episode cut after step 34 while chunk 1 (receipt plan step 36) is already in chunks.npz."""
    steps, chunks = _episode(make_run(tmp_path))
    steps = steps[:35]
    assert chunks["obs_step"][1] + chunks["latency_steps"][1] > steps[-1]["plan_step"] + 1

    al = _align(steps, chunks)

    c1 = al.chunks[1]
    assert c1["receipt_step"] is None
    assert c1["receipt_recorded"] is False
    assert c1["first"] is None
    assert c1["executed"] is None
    assert c1["n_executed"] == 0
    assert c1["start_offset"] == {"left": None, "right": None}
    assert al.chunks[0]["executed"] == [3, 32]
    assert al.mismatches == 0


# ----------------------------------------------------------------------------------------------------
# Async
# ----------------------------------------------------------------------------------------------------


def test_async_first_and_later_chunks(tmp_path):
    steps, chunks = _episode(make_run(tmp_path, overrides=("inference.mode=async",)))
    al = _align(steps, chunks, mode="async")

    c0, c1 = al.chunks[0], al.chunks[1]
    assert c0["first"] is True
    assert c0["stale_rows"] == 0
    assert c0["receipt_approx"] is True
    i0 = next(i for i, s in enumerate(steps) if s["step"] == c0["receipt_step"])
    assert al.per_step[i0][:2] == (0, 0)  # row 0 lands on the receipt step
    assert c1["first"] is False
    assert c1["stale_rows"] == chunks["latency_steps"][1] == 3
    for s, (k, _, _) in zip(steps, al.per_step, strict=True):
        assert (k is not None) == (s["mode"] == "run"), s["step"]
    assert al.mismatches is None
    # Chunk 1 (obs 18, received at plan step 21) takes over from its receipt with row 3.
    i1 = next(i for i, s in enumerate(steps) if s["step"] == c1["receipt_step"])
    assert steps[i1]["plan_step"] == 21
    assert al.per_step[i1][:2] == (1, 3)
    assert c0["executed"] == [3, 20]


def test_async_replace_rows_match_actions(tmp_path):
    """With merge=replace nothing is blended, so the async mapping can be checked against the recorded values."""
    helper = Rig()
    run = make_run(
        tmp_path,
        overrides=("inference.mode=async", "inference.async.merge=replace"),
        respond=lambda n: (3 + n % 3, helper.chunk(tag=0.05 * n), None),
    )
    steps, chunks = _episode(run)
    al = _align(steps, chunks, mode="async")

    assert chunks["chunks"].shape[0] >= 4
    _assert_rows_match_actions(steps, chunks, al)


def test_async_leash_freezes_plan_time(tmp_path):
    """Real Runner through a Rig with the leash engaged on ticks 19-23 (chunk 1 is received during the leash):
    leash ticks repeat a plan step, so the receipt is approximate but every run tick still maps to the row it ran."""
    rig = Rig(
        ["inference.mode=async", "inference.async.merge=replace"],
        respond=lambda n: (3, rig.chunk(tag=0.05 * n), None),
        keys={100: "q"},
        timeout=6.0,
    )
    leash = set(range(19, 24))
    rig.streamer.leash_fn = lambda tick, side: tick in leash
    steps, chunks = _read(_recorded_rig_episode(tmp_path, rig))
    al = _align(steps, chunks, mode="async")

    leash_steps = [s for s in steps if s["mode"] == "leash"]
    assert [s["step"] for s in leash_steps] == sorted(leash)
    assert {s["plan_step"] for s in leash_steps} == {19}  # plan time frozen
    # Chunk 1 (requested at tick 18) really arrives at tick 21, mid leash, with plan step 19 -> latency 1. The
    # first record with plan step 19 is tick 19: the receipt is early by the two leash ticks before it.
    assert chunks["latency_steps"][1] == 1
    assert al.chunks[1]["receipt_step"] == 19
    assert all(
        entry == (None, None, False) for s, entry in zip(steps, al.per_step, strict=True) if s["mode"] == "leash"
    )
    _assert_rows_match_actions(steps, chunks, al)
    assert all(c["receipt_approx"] for c in al.chunks)


def test_to_json_is_plain(tmp_path):
    steps, chunks = _episode(make_run(tmp_path))
    steps = steps[:35]
    out = _align(steps, chunks).to_json()
    assert set(out) == {"per_step", "chunks", "mismatches"}
    assert out["per_step"][3] == [0, 0, False]
    assert out["chunks"][0]["executed"] == [3, 32]
    assert out["chunks"][1]["start_offset"] == {"left": None, "right": None}
    json.dumps(out, allow_nan=False)
    assert isinstance(out["chunks"][0]["start_offset"]["right"], float)
