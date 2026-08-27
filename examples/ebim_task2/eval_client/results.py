"""Per-run output layout and result.jsonl bookkeeping."""

import dataclasses
import json
from pathlib import Path
import time

import numpy as np


@dataclasses.dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    results_path: Path
    config_path: Path


def run_layout(out_dir: Path, run_name: str | None) -> RunLayout:
    run_dir = Path(out_dir) / (run_name or f"eval_{int(time.time())}")
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Appending to an existing result.jsonl would restart episode
        # numbering and config.yaml would be overwritten.
        raise FileExistsError(
            f"run dir already exists: {run_dir} -- pick a new --run-name or remove the old run"
        ) from None
    return RunLayout(
        run_dir=run_dir,
        results_path=run_dir / "result.jsonl",
        config_path=run_dir / "config.yaml",
    )


def append_result(path: Path, record: dict) -> None:
    """One JSON line per episode, flushed immediately (crash-safe partials)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _yaml_safe(value):
    """Coerce to plain YAML-representable types (server metadata may carry
    tuples/numpy scalars; the config carries Path objects)."""
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _yaml_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def dump_run_config(path: Path, cfg, server_metadata: dict) -> None:
    import yaml

    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)
    payload = {
        "server_metadata": _yaml_safe(server_metadata or {}),
        "eval_config": _yaml_safe(cfg),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=True)


def summarize(results_path: Path) -> dict:
    records = [json.loads(line) for line in Path(results_path).read_text().splitlines()]

    def mean_of(key):
        values = [r[key] for r in records if r.get(key) is not None]
        return round(float(np.mean(values)), 1) if values else None

    return {
        "episodes": len(records),
        "aborted": sum(1 for r in records if r.get("aborted")),
        "steps_mean": mean_of("steps"),
        "duration_s_mean": mean_of("duration_s"),
        "infer_ms_mean": mean_of("infer_ms_mean"),
    }
