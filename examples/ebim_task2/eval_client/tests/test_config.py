from pathlib import Path
import textwrap

from eval_client import config as config_mod
from eval_client.config import EvalConfig
import pytest


@pytest.fixture(autouse=True)
def _hermetic_env(tmp_path, monkeypatch):
    """Isolate every test from the developer's real environment and
    examples/ebim_task2/.env (from_cli loads it into os.environ)."""
    for key in ("POLICY_SERVER_HOST", "POLICY_SERVER_PORT", "POLICY_SERVER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_mod, "DEFAULT_ENV_PATH", tmp_path / "absent.env")


@pytest.fixture
def config_file(tmp_path):
    def make(content: str) -> str:
        path = tmp_path / "eval.yaml"
        path.write_text(textwrap.dedent(content))
        return str(path)

    return make


class TestDefaults:
    def test_dataclass_defaults(self, config_file):
        cfg = EvalConfig.from_cli(["--config", config_file("{}")])
        assert cfg.host == "localhost"
        assert cfg.port == 8000
        assert cfg.num_episodes == 10
        assert cfg.episode_timeout == 90.0
        assert cfg.replan_steps == 25
        assert cfg.async_inference is True
        assert cfg.chunk_size_threshold == 0.5
        assert cfg.async_execute_horizon is None
        assert cfg.chunk_merge == "replace"
        assert cfg.chunk_blend_steps == 10
        assert cfg.queue_dry_abort_steps == 90
        assert cfg.max_reconnects == 3
        assert cfg.final_reset is True
        assert cfg.pedal_threshold == 0.25
        assert cfg.out_dir == Path("eval_results")
        assert cfg.run_name is None

    def test_env_host_port(self, config_file, monkeypatch):
        monkeypatch.setenv("POLICY_SERVER_HOST", "10.0.0.7")
        monkeypatch.setenv("POLICY_SERVER_PORT", "9000")
        cfg = EvalConfig.from_cli(["--config", config_file("{}")])
        assert cfg.host == "10.0.0.7"
        assert cfg.port == 9000

    def test_shipped_config_loads(self):
        shipped = Path(config_mod.__file__).resolve().parent.parent / "eval_config.yaml"
        cfg = EvalConfig.from_cli(["--config", str(shipped)])
        assert cfg.num_episodes == 10


class TestPrecedence:
    def test_yaml_overrides_defaults(self, config_file):
        cfg = EvalConfig.from_cli(
            ["--config", config_file("num_episodes: 3\nchunk_merge: offset_decay\nchunk_blend_steps: 6\n")]
        )
        assert cfg.num_episodes == 3
        assert cfg.chunk_merge == "offset_decay"
        assert cfg.chunk_blend_steps == 6

    def test_cli_overrides_yaml(self, config_file):
        cfg = EvalConfig.from_cli(["--config", config_file("num_episodes: 3\n"), "--num-episodes", "7"])
        assert cfg.num_episodes == 7

    def test_yaml_null_keeps_default(self, config_file):
        cfg = EvalConfig.from_cli(["--config", config_file("num_episodes: null\n")])
        assert cfg.num_episodes == 10

    def test_boolean_flag_forms(self, config_file):
        path = config_file("{}")
        assert EvalConfig.from_cli(["--config", path, "--no-async-inference"]).async_inference is False
        assert EvalConfig.from_cli(["--config", config_file("async_inference: false\n")]).async_inference is False


class TestRunName:
    def test_cli_flag(self, config_file):
        cfg = EvalConfig.from_cli(["--config", config_file("{}"), "--run-name", "foo_01"])
        assert cfg.run_name == "foo_01"

    def test_yaml_value(self, config_file):
        cfg = EvalConfig.from_cli(["--config", config_file("run_name: foo\n")])
        assert cfg.run_name == "foo"

    def test_cli_overrides_yaml(self, config_file):
        cfg = EvalConfig.from_cli(["--config", config_file("run_name: foo\n"), "--run-name", "bar"])
        assert cfg.run_name == "bar"

    def test_yaml_null_keeps_default(self, config_file):
        cfg = EvalConfig.from_cli(["--config", config_file("run_name: null\n")])
        assert cfg.run_name is None

    def test_invalid_name_fails_via_cli(self, config_file):
        with pytest.raises(ValueError, match="run_name"):
            EvalConfig.from_cli(["--config", config_file("{}"), "--run-name", "a/b"])

    @pytest.mark.parametrize("bad", ["a/b", "", "-x", "a b", ".hidden"])
    def test_invalid_names_fail(self, bad):
        with pytest.raises(ValueError, match="run_name"):
            EvalConfig(run_name=bad)


class TestValidation:
    def test_unknown_yaml_key_fails(self, config_file):
        with pytest.raises(RuntimeError, match="wat"):
            EvalConfig.from_cli(["--config", config_file("wat: 1\n")])

    def test_missing_config_file_fails(self):
        with pytest.raises(FileNotFoundError):
            EvalConfig.from_cli(["--config", "/nonexistent/eval.yaml"])

    def test_bad_merge_name_fails(self, config_file):
        with pytest.raises(ValueError, match="chunk_merge"):
            EvalConfig.from_cli(["--config", config_file("chunk_merge: nope\n")])

    def test_bad_blend_steps_fails(self, config_file):
        with pytest.raises(ValueError, match="chunk_blend_steps"):
            EvalConfig.from_cli(["--config", config_file("chunk_blend_steps: 0\n")])

    def test_bad_threshold_fails(self, config_file):
        with pytest.raises(ValueError, match="chunk_size_threshold"):
            EvalConfig.from_cli(["--config", config_file("chunk_size_threshold: 1.5\n")])

    def test_bad_horizon_fails(self, config_file):
        with pytest.raises(ValueError, match="replan_steps"):
            EvalConfig.from_cli(["--config", config_file("replan_steps: 0\n")])


class TestDotenv:
    def test_parses_and_sets_missing_keys(self, tmp_path, monkeypatch):
        monkeypatch.delenv("POLICY_SERVER_HOST", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\n"
            "\n"
            "POLICY_SERVER_HOST=10.1.2.3\n"
            'POLICY_SERVER_PORT="8123"\n'
            "export POLICY_SERVER_API_KEY='secret-key'\n"
        )
        config_mod.load_dotenv(env_file)
        import os

        assert os.environ["POLICY_SERVER_HOST"] == "10.1.2.3"
        assert os.environ["POLICY_SERVER_PORT"] == "8123"
        assert os.environ["POLICY_SERVER_API_KEY"] == "secret-key"

    def test_does_not_override_real_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLICY_SERVER_HOST", "from-real-env")
        env_file = tmp_path / ".env"
        env_file.write_text("POLICY_SERVER_HOST=from-dotenv\n")
        config_mod.load_dotenv(env_file)
        import os

        assert os.environ["POLICY_SERVER_HOST"] == "from-real-env"

    def test_missing_file_is_fine(self, tmp_path):
        config_mod.load_dotenv(tmp_path / "nope.env")

    def test_from_cli_reads_dotenv(self, tmp_path, monkeypatch, config_file):
        for key in ("POLICY_SERVER_HOST", "POLICY_SERVER_PORT", "POLICY_SERVER_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("POLICY_SERVER_HOST=envfile-host\nPOLICY_SERVER_API_KEY=k1\n")
        monkeypatch.setattr(config_mod, "DEFAULT_ENV_PATH", env_file)
        cfg = EvalConfig.from_cli(["--config", config_file("{}")])
        assert cfg.host == "envfile-host"
        assert cfg.api_key == "k1"


class TestApiKeyEnv:
    def test_api_key_from_env(self, config_file, monkeypatch):
        monkeypatch.setenv("POLICY_SERVER_API_KEY", "abc123")
        cfg = EvalConfig.from_cli(["--config", config_file("{}")])
        assert cfg.api_key == "abc123"


class TestRedactedDict:
    def test_excludes_connection_fields(self):
        cfg = EvalConfig(host="10.0.0.1", port=9999, api_key="secret")
        data = cfg.redacted_dict()
        assert "host" not in data
        assert "port" not in data
        assert "api_key" not in data
        assert data["num_episodes"] == 10
        assert data["chunk_merge"] == "replace"
