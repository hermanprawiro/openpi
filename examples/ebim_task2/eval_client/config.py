"""Eval client configuration.

One flat dataclass; precedence is dataclass defaults < YAML file
(--config) < explicit CLI flags. host/port defaults come from the
POLICY_SERVER_HOST / POLICY_SERVER_PORT environment variables so the
same compose file works cross-machine.
"""

import argparse
import dataclasses
import os
from pathlib import Path
import re

from eval_client.chunking import CHUNK_MERGE_STRATEGIES
from eval_client.constants import DEFAULT_PROMPT

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval_config.yaml"
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_CLI_ONLY_KEYS = {"config"}
# Connection settings live in the environment / .env, not in the shipped
# YAML, and are excluded from the config.yaml written into eval results
# (that artifact may be shared; see redacted_dict).
CONNECTION_KEYS = ("host", "port", "api_key")
# One path component, no whitespace: the name is used verbatim as a directory
# name here and as the recorder's --eval-name, and must survive EVAL_ARGS
# word-splitting in docker compose.
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE .env loader (comments, blank lines, optional
    'export ' prefix and quotes). Real environment variables win; a
    missing file is fine."""
    path = Path(path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclasses.dataclass
class EvalConfig:
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None
    num_episodes: int = 10
    episode_timeout: float = 90.0  # simulated seconds
    prompt: str = DEFAULT_PROMPT
    replan_steps: int = 25
    pedal_threshold: float = 0.25
    async_inference: bool = True
    chunk_size_threshold: float = 0.5
    async_execute_horizon: int | None = None
    chunk_merge: str = "replace"
    queue_dry_abort_steps: int = 90
    max_reconnects: int = 3
    final_reset: bool = True
    out_dir: Path = Path("eval_results")
    run_name: str | None = None

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        if self.run_name is not None and not RUN_NAME_RE.match(self.run_name):
            raise ValueError(
                f"run_name must match {RUN_NAME_RE.pattern} (single path component, no spaces), got {self.run_name!r}"
            )
        if self.chunk_merge not in CHUNK_MERGE_STRATEGIES:
            raise ValueError(f"chunk_merge must be one of {sorted(CHUNK_MERGE_STRATEGIES)}, got {self.chunk_merge!r}")
        if not 0.0 <= self.chunk_size_threshold < 1.0:
            raise ValueError(f"chunk_size_threshold must be in [0.0, 1.0), got {self.chunk_size_threshold}")
        if self.replan_steps < 1:
            raise ValueError(f"replan_steps must be >= 1, got {self.replan_steps}")
        if self.async_execute_horizon is not None and self.async_execute_horizon < 1:
            raise ValueError(f"async_execute_horizon must be >= 1, got {self.async_execute_horizon}")

    @property
    def execute_horizon(self) -> int:
        return self.async_execute_horizon or self.replan_steps

    def redacted_dict(self) -> dict:
        """Config as a dict without the connection settings -- the form
        that may be written into shareable eval-result artifacts."""
        data = dataclasses.asdict(self)
        for key in CONNECTION_KEYS:
            data.pop(key, None)
        return data

    @classmethod
    def from_cli(cls, argv=None) -> "EvalConfig":
        load_dotenv(DEFAULT_ENV_PATH)
        parser = _build_parser()
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
        pre_args, _ = pre.parse_known_args(argv)
        parser.set_defaults(**_load_yaml_defaults(pre_args.config, parser))
        args = vars(parser.parse_args(argv))
        args.pop("config", None)
        return cls(**args)


def _env_default(name: str, fallback, cast):
    value = os.environ.get(name)
    return cast(value) if value else fallback


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    fields = {f.name: f for f in dataclasses.fields(EvalConfig)}
    defaults = {name: field.default for name, field in fields.items() if field.default is not dataclasses.MISSING}
    defaults["host"] = _env_default("POLICY_SERVER_HOST", defaults["host"], str)
    defaults["port"] = _env_default("POLICY_SERVER_PORT", defaults["port"], int)
    defaults["api_key"] = _env_default("POLICY_SERVER_API_KEY", defaults["api_key"], str)

    for name, default in defaults.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=default)
        elif name in ("api_key", "run_name"):
            parser.add_argument(flag, dest=name, default=default)
        elif name == "async_execute_horizon":
            parser.add_argument(flag, dest=name, type=int, default=default)
        elif name == "out_dir":
            parser.add_argument(flag, dest=name, type=Path, default=default)
        else:
            parser.add_argument(flag, dest=name, type=type(default), default=default)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML defaults file; CLI flags override it.")
    return parser


def _load_yaml_defaults(config_path, parser) -> dict:
    """YAML layer between dataclass defaults and CLI flags. Fails hard on a
    missing file or an unknown/CLI-only key; null keeps the default."""
    import yaml

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Eval config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Eval config must be a YAML mapping: {config_path}")

    valid = {f.name for f in dataclasses.fields(EvalConfig)} - _CLI_ONLY_KEYS
    defaults = {}
    for key, value in loaded.items():
        if key not in valid:
            raise RuntimeError(f"Unknown or CLI-only key {key!r} in {config_path}; valid keys: {sorted(valid)}")
        if value is None:
            continue  # null keeps the dataclass default
        defaults[key] = value
    return defaults
