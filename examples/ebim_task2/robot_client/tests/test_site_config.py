import dataclasses
from pathlib import Path

import pytest
from robot_client import config as config_mod
from robot_client import layout
from robot_client.config import ConfigError
from robot_client.config import check_ros_env
from robot_client.config import load_site_config
import yaml

MUNICH_YAML = Path(config_mod.__file__).with_name("robot_site.munich.yaml")


@pytest.fixture
def load():
    def _load(overrides=(), env=None):
        return load_site_config(MUNICH_YAML, overrides, env or {})

    return _load


def errors_of(load, overrides, env=None) -> str:
    with pytest.raises(ConfigError) as exc:
        load(overrides, env)
    return str(exc.value)


def test_munich_yaml_loads_with_defaults(load):
    cfg = load()
    assert cfg.inference.mode == "sync"
    assert cfg.inference.execute_horizon == 30
    assert cfg.inference.async_cfg.merge == "blend_favor_old"
    assert cfg.inference.async_cfg.trigger_threshold == 0.5
    assert cfg.policy.prompt == layout.PROMPT
    assert cfg.state.pin == {7: 0.0}
    assert cfg.active_arms == ("left", "right")
    assert cfg.command_mode == "absolute"
    assert cfg.arms["left"].gripper.required is False
    assert cfg.arms["right"].gripper.required is True
    assert cfg.spine.height == 0.434
    assert cfg.safety.max_track_lag_rad == 0.5
    assert cfg.safety.episode_timeout_s == 120.0
    assert cfg.safety.stream_gap_freeze_s == 0.1
    assert cfg.safety.joint_limits["left"][0][0] == -2.7437
    assert cfg.takeover.mode == "switch"
    assert cfg.stream_hz == 100
    assert set(cfg.cameras) == {"head", "wrist_left", "wrist_right"}
    assert cfg.arms["right"].command_joint_names is None


def test_munich_head_train_res_matches_layout(load):
    cfg = load()
    assert cfg.cameras["head"].train_res == layout.HEAD_TRAIN_RES
    assert cfg.cameras["wrist_left"].train_res is None
    assert cfg.cameras["wrist_right"].train_res is None


def test_munich_start_poses_are_the_medoids(load):
    import json

    stats_path = Path.home() / ".cache/huggingface/lerobot/ebim/task2_munich_real/conversion/start_pose_stats.json"
    if not stats_path.exists():
        pytest.skip("start_pose_stats.json not available")
    stats = json.loads(stats_path.read_text())
    cfg = load()
    for arm in ("left", "right"):
        assert cfg.arms[arm].start_pose == tuple(round(v, 3) for v in stats[arm]["medoid"])


def test_override_dotted_key(load):
    cfg = load(["safety.max_step_rad=0.05", "inference.mode=async", "inference.async.blend_steps=5"])
    assert cfg.safety.max_step_rad == 0.05
    assert cfg.inference.mode == "async"
    assert cfg.inference.async_cfg.blend_steps == 5
    # Unrelated fields keep the YAML values.
    assert cfg.safety.max_track_lag_rad == 0.5


def test_override_complex_values(load):
    cfg = load(["active_arms=[right]", "state.pin.7=0.5", "cameras.head.crop=[0,10,640,360]"])
    assert cfg.active_arms == ("right",)
    assert cfg.state.pin == {7: 0.5}
    assert cfg.cameras["head"].crop == (0, 10, 640, 360)


def test_stream_gap_needs_two_stream_periods(load):
    msg = errors_of(load, ["stream_hz=10", "safety.stream_gap_freeze_s=0.15"])
    assert "safety.stream_gap_freeze_s" in msg
    assert "stream_hz" in msg
    cfg = load(["stream_hz=20", "safety.stream_gap_freeze_s=0.1"])  # exactly two periods is fine
    assert cfg.safety.stream_gap_freeze_s == 0.1


def test_unknown_key_named(load):
    msg = errors_of(load, ["safety.max_stp_rad=0.05"])
    assert "safety.max_stp_rad" in msg
    assert "max_step_rad" in msg  # suggestion


def test_unknown_key_in_yaml_file(tmp_path):
    data = yaml.safe_load(MUNICH_YAML.read_text())
    data["arms"]["left"]["ptp_actoin"] = "/x"
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=r"arms\.left\.ptp_actoin"):
        load_site_config(path, env={})


def test_all_errors_reported_together(load):
    msg = errors_of(load, ["inference.execute_horizon=99", "spine.height=434", "safety.max_step_rad=fast"])
    assert "inference.execute_horizon" in msg
    assert "spine.height" in msg
    assert "safety.max_step_rad" in msg
    assert "3 problem(s)" in msg


def test_wrong_prompt_is_an_error_naming_the_key(load):
    msg = errors_of(load, ["policy.prompt=Pick up the thermal pad."])
    assert "policy.prompt" in msg


def test_camera_and_arm_keys_must_be_exact(load):
    msg = errors_of(load, ["cameras.top={topic: /t}", "arms.middle={}"])
    assert "cameras: keys must be exactly" in msg
    assert "arms: keys must be exactly" in msg


def test_missing_camera_reported(tmp_path):
    data = yaml.safe_load(MUNICH_YAML.read_text())
    del data["cameras"]["wrist_left"]
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=r"cameras: keys must be exactly .*missing \['wrist_left'\]"):
        load_site_config(path, env={})


def test_missing_required_key_named(tmp_path):
    data = yaml.safe_load(MUNICH_YAML.read_text())
    del data["arms"]["right"]["gripper"]["command_topic"]
    del data["spine"]
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(data))
    msg = errors_of(lambda o, e: load_site_config(path, o, e or {}), [])
    assert "arms.right.gripper.command_topic: required key is missing" in msg
    assert "spine: required key is missing" in msg


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ("active_arms=[]", "active_arms"),
        ("active_arms=[middle]", "active_arms"),
        ("active_arms=[left,left]", "active_arms"),
        ("arms.left.joint_names=[a,b]", "arms.left.joint_names"),
        ("arms.right.start_pose=[0,0,0]", "arms.right.start_pose"),
        ("joint_signs.left=[1,1,1,1,1,1,2]", "joint_signs.left"),
        ("joint_signs.right=[1,1]", "joint_signs.right"),
        ("inference.execute_horizon=0", "inference.execute_horizon"),
        ("inference.execute_horizon=51", "inference.execute_horizon"),
        ("inference.mode=maybe", "inference.mode"),
        ("inference.async.merge=nope", "inference.async.merge"),
        ("spine.height=434", "spine.height"),
        ("spine.height=-0.1", "spine.height"),
        ("command_mode=delta", "command_mode"),
        ("safety.obs_stale_freeze_s=0.5", "safety.obs_stale_freeze_s"),
        ("safety.stream_gap_freeze_s=0", "safety.stream_gap_freeze_s"),
        ("safety.joint_limits={left: [[0],[1]]}", "safety.joint_limits"),
        ("smoother.velocity_scale=1.5", "smoother.velocity_scale"),
        ("smoother.v_max=[1,2]", "smoother.v_max"),
        ("prepare.timeout_s=0", "prepare.timeout_s"),
        ("cameras.head.train_res=[0,188]", "cameras.head.train_res"),
        ("cameras.head.crop=[0,0,0,10]", "cameras.head.crop"),
        ("cameras.head.type=jpeg", "cameras.head.type"),
        ("state.pin.16=0.0", "state.pin.16"),
        ("ros.domain_id=1000", "ros.domain_id"),
        ("stream_hz=0", "stream_hz"),
        ("takeover.mode=magic", "takeover.mode"),
        ("arms.right.start_pose=[9,0,0,-1,0,1,0]", "arms.right.start_pose[0]"),
    ],
)
def test_invalid_value_names_the_dotted_key(load, override, key):
    assert key in errors_of(load, [override])


def test_bad_override_syntax(load):
    msg = errors_of(load, ["safety.max_step_rad", "ros.rmw.x=1"])
    assert "expected dotted.key=yaml_value" in msg
    assert "ros.rmw" in msg


def test_env_server_and_redaction(load):
    env = {"POLICY_SERVER_HOST": "10.1.2.3", "POLICY_SERVER_PORT": "9000", "POLICY_SERVER_API_KEY": "s3cret"}
    cfg = load(env=env)
    assert (cfg.server.host, cfg.server.port, cfg.server.api_key) == ("10.1.2.3", 9000, "s3cret")
    assert cfg.server.connect_timeout_s == 10.0
    dumped = cfg.redacted_dict()
    assert dumped["server"]["api_key"] == "***"
    assert "s3cret" not in repr(dumped)
    assert dumped["server"]["host"] == "10.1.2.3"
    assert "async" in dumped["inference"]  # YAML key names
    yaml.safe_dump(dumped)  # plain data only


def test_server_defaults_without_env(load):
    cfg = load()
    assert (cfg.server.host, cfg.server.port, cfg.server.api_key) == ("localhost", 8000, None)
    assert cfg.redacted_dict()["server"]["api_key"] is None


def test_server_connection_not_allowed_in_yaml_or_set(load):
    msg = errors_of(load, ["server.host=evil", "server.api_key=x"])
    assert "server.host" in msg and "POLICY_SERVER_HOST" in msg
    assert "server.api_key" in msg and "POLICY_SERVER_API_KEY" in msg


def test_bad_env_port(load):
    assert "POLICY_SERVER_PORT" in errors_of(load, [], {"POLICY_SERVER_PORT": "eight"})


def test_empty_env_values_are_unset(load):
    cfg = load(env={"POLICY_SERVER_HOST": "", "POLICY_SERVER_PORT": "", "POLICY_SERVER_API_KEY": ""})
    assert (cfg.server.host, cfg.server.port, cfg.server.api_key) == ("localhost", 8000, None)


def test_ros_env_mismatch_reported(load):
    cfg = load()
    ok = {"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp", "ROS_DOMAIN_ID": "0"}
    assert check_ros_env(cfg, ok) == []
    assert check_ros_env(cfg, {}) == []  # unset = ROS defaults
    problems = check_ros_env(cfg, {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp", "ROS_DOMAIN_ID": "5"})
    assert len(problems) == 2
    assert "RMW_IMPLEMENTATION" in problems[0] and "rmw_cyclonedds_cpp" in problems[0]
    assert "ROS_DOMAIN_ID" in problems[1]
    assert "ROS_DOMAIN_ID" in check_ros_env(cfg, {"ROS_DOMAIN_ID": "x"})[0]


def test_file_problems(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_site_config(tmp_path / "absent.yaml", env={})
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_site_config(bad, env={})
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="top level"):
        load_site_config(bad, env={})


def test_config_is_frozen(load):
    cfg = load()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.safety.max_step_rad = 1.0


def test_load_env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# c\nPOLICY_SERVER_HOST=10.0.0.1\nexport POLICY_SERVER_PORT='9000'\n\nbogus\n")
    assert config_mod.load_env_file(path) == {"POLICY_SERVER_HOST": "10.0.0.1", "POLICY_SERVER_PORT": "9000"}
    assert config_mod.load_env_file(tmp_path / "absent.env") == {}
