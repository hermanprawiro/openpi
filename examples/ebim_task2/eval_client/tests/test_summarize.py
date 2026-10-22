"""Tests for summarize_eval.py (examples/ebim_task2/, on sys.path via conftest)."""

import json

import pytest
import summarize_eval


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _evaluator_result(iou, *, orientation_ok, call_index=1):
    return {
        "success": True,
        "trigger": "reset_request",
        "sim_time": 88.5,
        "call_index": call_index,
        "iou": {
            "iou_thermalpad_vs_target_current": iou,
            "is_orientation_correct": orientation_ok,
            "orientation_case": "aligned" if orientation_ok else "flipped",
            "orientation_confidence": 0.9,
            "evaluator_version": "v5",
            "physical_audit": {"coverage": 0.8, "spill": 0.01, "off_board": False},
        },
    }


@pytest.fixture
def session_dir(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _write_json(session / ".status.json", {"ts": 1, "state": "watching", "resets": 3})
    _write_json(session / ".session_provenance.json", {"wall_time_utc": "2026-08-26T10:00:00Z"})

    ep1 = session / "ep_001"
    _write_json(ep1 / "manifest.json", {"discarded": False, "start_reset_event": {"target_slot": "slot_a"}})
    _write_json(ep1 / "evaluator" / "call_001" / "evaluator_result.json", _evaluator_result(0.72, orientation_ok=True))
    art = ep1 / "evaluator" / "call_001" / "artifacts"
    art.mkdir(parents=True)
    (art / "eval_camera_rgb_000123.jpg").write_bytes(b"fakejpegdata")

    ep2 = session / "ep_002"
    _write_json(ep2 / "manifest.json", {"discarded": False})
    _write_json(ep2 / "evaluator" / "call_001" / "evaluator_result.json", _evaluator_result(0.10, orientation_ok=False))

    # Trailing episode: opened by the client's final reset, discarded by the wrapper.
    ep3 = session / "ep_003"
    _write_json(ep3 / "manifest.json", {"discarded": True})
    return session


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    records = [
        {
            "episode": 0,
            "aborted": False,
            "steps": 120,
            "duration_s": 45.6,
            "infer_ms_mean": 480.0,
            "switch_jump_max": 0.0312,
            "switch_jump_mean": 0.0104,
        },
        {"episode": 1, "aborted": True, "abort_reason": "stale streams", "steps": 10, "duration_s": 3.2},
    ]
    with open(run / "result.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return run


class TestWriteSummary:
    def test_writes_markdown_with_merged_rows(self, session_dir, run_dir):
        out = summarize_eval.write_summary(session_dir, run_dir)
        assert out == run_dir / "summary.md"
        text = out.read_text(encoding="utf-8")
        assert "ep_001" in text
        assert "0.7200" in text  # evaluator IoU
        assert "120" in text  # client steps
        assert "stale streams" in text  # client abort reason

    def test_handoff_jump_metric_reported(self, session_dir, run_dir):
        text = summarize_eval.write_summary(session_dir, run_dir).read_text(encoding="utf-8")
        assert "| Mean hand-off jump, max per episode (rad) | 0.0312 |" in text
        assert "| Jump (rad) |" in text  # per-episode client column
        ep1_row = next(line for line in text.splitlines() if line.startswith("| `ep_001`"))
        assert "| 0.0312 |" in ep1_row
        ep2_row = next(line for line in text.splitlines() if line.startswith("| `ep_002`"))
        assert "| - | yes: stale streams" in ep2_row  # aborted episode has no jump stats

    def test_discarded_episode_excluded_from_stats(self, session_dir, run_dir):
        text = summarize_eval.write_summary(session_dir, run_dir).read_text(encoding="utf-8")
        assert "| Episodes recorded | 2 |" in text
        assert "discarded" in text.lower()  # footnote mentions the skipped episode

    def test_images_copied_into_run_dir(self, session_dir, run_dir):
        text = summarize_eval.write_summary(session_dir, run_dir).read_text(encoding="utf-8")
        copied = run_dir / "images" / "ep_001.jpg"
        assert copied.is_file()
        assert copied.read_bytes() == b"fakejpegdata"
        assert "](images/ep_001.jpg)" in text

    def test_count_mismatch_warns(self, session_dir, run_dir):
        lines = (run_dir / "result.jsonl").read_text(encoding="utf-8").splitlines()
        (run_dir / "result.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
        text = summarize_eval.write_summary(session_dir, run_dir).read_text(encoding="utf-8")
        assert "WARNING" in text

    def test_missing_result_jsonl_degrades_gracefully(self, session_dir, tmp_path):
        empty_run = tmp_path / "empty_run"
        empty_run.mkdir()
        text = summarize_eval.write_summary(session_dir, empty_run).read_text(encoding="utf-8")
        assert "ep_001" in text
        assert "WARNING" in text

    def test_slot_and_success_columns(self, session_dir, run_dir):
        text = summarize_eval.write_summary(session_dir, run_dir).read_text(encoding="utf-8")
        assert "| Episode | Slot | Calls | Call used | Trigger | Success | IoU |" in text
        rows = text.splitlines()
        ep1_row = next(line for line in rows if line.startswith("| `ep_001`"))
        assert ep1_row.startswith("| `ep_001` | slot_a | 1 | 1 | reset_request | yes | 0.7200 |")
        # IoU > 0 but wrong orientation is not a success; no recorded slot renders as "-".
        ep2_row = next(line for line in rows if line.startswith("| `ep_002`"))
        assert ep2_row.startswith("| `ep_002` | - | 1 | 1 | reset_request | no | 0.1000 |")

    def test_success_by_slot_table(self, session_dir, run_dir):
        text = summarize_eval.write_summary(session_dir, run_dir).read_text(encoding="utf-8")
        assert "| Successful episodes (IoU > 0 and orientation correct) | 1 / 2 |" in text
        assert "### Success by target slot" in text
        assert "| Slot | Successes | Episodes | Success rate |" in text
        assert "| slot_a | 1 | 1 | 100% |" in text
        assert "| - | 0 | 1 | 0% |" in text
        assert "| **All** | 1 | 2 | 50% |" in text

    def test_success_by_slot_reports_non_home_slots(self, session_dir, run_dir):
        manifest = {"discarded": False, "start_reset_event": {"target_slot": "B"}}
        _write_json(session_dir / "ep_002" / "manifest.json", manifest)
        text = summarize_eval.write_summary(session_dir, run_dir).read_text(encoding="utf-8")
        assert "| B (home) | 0 | 1 | 0% |" in text
        assert "| Other than B | 1 | 1 | 100% |" in text
