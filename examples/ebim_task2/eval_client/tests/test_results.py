import json
from pathlib import Path

from eval_client import results
import numpy as np
import yaml


class TestRunLayout:
    def test_paths(self, tmp_path):
        layout = results.run_layout(tmp_path / "out", 1234)
        assert layout.run_dir == tmp_path / "out" / "eval_1234"
        assert layout.results_path == layout.run_dir / "result.jsonl"
        assert layout.config_path == layout.run_dir / "config.yaml"
        assert layout.run_dir.is_dir()


class TestAppendResult:
    def test_appends_json_lines(self, tmp_path):
        path = tmp_path / "result.jsonl"
        results.append_result(path, {"episode": 0, "aborted": False})
        results.append_result(path, {"episode": 1, "aborted": True})
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert lines == [{"episode": 0, "aborted": False}, {"episode": 1, "aborted": True}]


class TestDumpRunConfig:
    def test_yaml_safe_coercion(self, tmp_path):
        path = tmp_path / "config.yaml"
        cfg = {"out_dir": Path("eval_results"), "threshold": np.float32(0.5), "dims": (1, 2)}
        metadata = {"actions": np.array([1, 2])}
        results.dump_run_config(path, cfg, metadata)
        loaded = yaml.safe_load(path.read_text())
        assert loaded["eval_config"]["out_dir"] == "eval_results"
        assert loaded["eval_config"]["threshold"] == 0.5
        assert loaded["eval_config"]["dims"] == [1, 2]
        assert loaded["server_metadata"]["actions"] == [1, 2]


class TestSummarize:
    def test_summary(self, tmp_path):
        path = tmp_path / "result.jsonl"
        results.append_result(
            path, {"episode": 0, "aborted": False, "steps": 100, "duration_s": 3.3, "infer_ms_mean": 500.0}
        )
        results.append_result(
            path,
            {
                "episode": 1,
                "aborted": True,
                "abort_reason": "stale",
                "steps": 10,
                "duration_s": 0.3,
                "infer_ms_mean": 400.0,
            },
        )
        summary = results.summarize(path)
        assert summary["episodes"] == 2
        assert summary["aborted"] == 1
        assert summary["steps_mean"] == 55.0
        assert summary["infer_ms_mean"] == 450.0
