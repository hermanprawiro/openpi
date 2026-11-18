import numpy as np
from openpi_client import image_tools
from PIL import Image
import pytest
from robot_client import layout
from robot_client import preprocess
from robot_client.config import CameraConfig

CAMERAS = {
    "head": CameraConfig(topic="/head", train_res=layout.HEAD_TRAIN_RES),
    "wrist_left": CameraConfig(topic="/wl"),
    "wrist_right": CameraConfig(topic="/wr"),
}
LEFT_Q = np.linspace(0.1, 0.7, 7)
RIGHT_Q = np.linspace(-0.7, -0.1, 7)


def _snapshot(*, knuckle=None, images=None):
    rng = np.random.default_rng(0)
    if images is None:
        images = {
            "head": rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
            "wrist_left": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist_right": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        }
    return preprocess.Snapshot(
        images=images,
        arm_q={"left": LEFT_Q, "right": RIGHT_Q},
        knuckle={"left": 0.003, "right": 0.4} if knuckle is None else knuckle,
        ages={},
        t_mono=0.0,
    )


# --- image decoding -------------------------------------------------------------------------------


def test_decode_bgra8_and_reject_mono():
    # 2x2 bgra8 with 4 bytes of row padding (step = 12 instead of 8).
    rows = [
        [(1, 2, 3, 255), (4, 5, 6, 255), (9, 9, 9, 9)],  # last pixel is padding
        [(7, 8, 9, 255), (10, 11, 12, 255), (9, 9, 9, 9)],
    ]
    data = np.array(rows, np.uint8).tobytes()
    out = preprocess.decode_raw_image(data, height=2, width=2, step=12, encoding="bgra8")
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8
    assert out.tolist() == [[[3, 2, 1], [6, 5, 4]], [[9, 8, 7], [12, 11, 10]]]
    with pytest.raises(preprocess.UnsupportedEncodingError, match="mono8"):
        preprocess.decode_raw_image(bytes(4), 2, 2, 2, "mono8")


@pytest.mark.parametrize(
    ("encoding", "channels", "order"),
    [("rgb8", 3, (0, 1, 2)), ("bgr8", 3, (2, 1, 0)), ("rgba8", 4, (0, 1, 2)), ("bgra8", 4, (2, 1, 0))],
)
def test_decode_raw_encodings(encoding, channels, order):
    raw = np.random.default_rng(1).integers(0, 255, (3, 5, channels), dtype=np.uint8)
    out = preprocess.decode_raw_image(raw.tobytes(), 3, 5, 5 * channels, encoding.upper())
    assert np.array_equal(out, raw[:, :, list(order)])
    assert out.flags.c_contiguous and out.flags.writeable


def test_decode_raw_rejects_inconsistent_sizes():
    with pytest.raises(ValueError, match="step"):
        preprocess.decode_raw_image(bytes(30), 2, 5, 6, "rgb8")  # step < width * 3
    with pytest.raises(ValueError, match="bytes"):
        preprocess.decode_raw_image(bytes(10), 2, 2, 6, "rgb8")


def test_decode_compressed_roundtrip():
    cv2 = pytest.importorskip("cv2")
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:, :, 0] = 200  # red
    ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])  # cv2 expects BGR
    assert ok
    out = preprocess.decode_compressed_image(buf.tobytes())
    assert np.array_equal(out, rgb)
    with pytest.raises(ValueError, match="decode"):
        preprocess.decode_compressed_image(b"not an image")


# --- joint mapping --------------------------------------------------------------------------------


def test_state_maps_by_name_and_reports_missing():
    wanted = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
    names = ["gripper_x", *reversed(wanted), "extra"]
    positions = [99.0, *[float(i) for i in range(7, 0, -1)], 98.0]
    out = preprocess.map_joints(names, positions, wanted)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    seen = [n for n in names if n != "right_fr3v2_joint3"]
    with pytest.raises(preprocess.MissingJointError) as exc:
        preprocess.map_joints(seen, [0.0] * len(seen), wanted)
    assert exc.value.missing == ["right_fr3v2_joint3"]
    assert "right_fr3v2_joint3" in str(exc.value)
    assert "gripper_x" in str(exc.value)  # the names that were seen


def test_map_joints_length_mismatch():
    with pytest.raises(ValueError, match="positions"):
        preprocess.map_joints(["a", "b"], [0.0], ["a"])


# --- state ----------------------------------------------------------------------------------------


def test_build_state_layout():
    state = preprocess.build_state(_snapshot(), {})
    assert state.shape == (layout.STATE_DIM,) and state.dtype == np.float32
    assert np.allclose(state[layout.ARM_SLICES["left"]], LEFT_Q)
    assert np.allclose(state[layout.ARM_SLICES["right"]], RIGHT_Q)
    assert state[layout.GRIPPER_INDEX["left"]] == pytest.approx(0.003)
    assert state[layout.GRIPPER_INDEX["right"]] == pytest.approx(0.4)


def test_pin_overrides_left_knuckle():
    state = preprocess.build_state(_snapshot(), {7: 0.0})
    assert state[7] == 0.0
    assert state[15] == pytest.approx(0.4)  # unpinned dim keeps the live value


def test_missing_optional_gripper_uses_pin():
    snap = _snapshot(knuckle={"right": 0.4})
    assert preprocess.build_state(snap, {7: 0.0})[7] == 0.0
    with pytest.raises(preprocess.StateError, match="left"):
        preprocess.build_state(snap, {})  # no sample and no pin: refuse to invent a value


def test_build_state_rejects_missing_arm_and_non_finite():
    snap = _snapshot()
    del snap.arm_q["right"]
    with pytest.raises(preprocess.StateError, match="right"):
        preprocess.build_state(snap, {7: 0.0})
    snap = _snapshot()
    snap.arm_q["left"] = np.array([np.nan, *LEFT_Q[1:]])
    with pytest.raises(preprocess.StateError, match="finite"):
        preprocess.build_state(snap, {7: 0.0})


def _norm_stats():
    q01 = np.full(16, -1.0)
    q99 = np.full(16, 1.0)
    q01[7] = q99[7] = 0.0  # left knuckle is constant in the data
    return {"norm_stats": {"state": {"q01": q01.tolist(), "q99": q99.tolist()}}}


def test_state_range_report_flags_out_of_range():
    state = np.zeros(16, np.float32)
    assert preprocess.state_range_report(state, _norm_stats()) == []
    assert preprocess.state_range_report(state, None) == []
    state[3] = 1.15  # within 10 % of the [-1, 1] range (1.2)
    assert preprocess.state_range_report(state, _norm_stats()) == []
    state[3] = 1.25
    state[7] = 0.003  # constant dim: any deviation is out of range
    report = preprocess.state_range_report(state, _norm_stats())
    assert len(report) == 2
    assert "dim 3" in report[0] and "1.25" in report[0]
    assert "dim 7" in report[1]


def test_state_range_report_accepts_bare_state_stats_and_flags_nan():
    stats = _norm_stats()["norm_stats"]["state"]
    state = np.zeros(16, np.float32)
    state[0] = np.nan
    report = preprocess.state_range_report(state, stats)
    assert len(report) == 1 and "dim 0" in report[0]
    with pytest.raises(ValueError, match="16"):
        preprocess.state_range_report(np.zeros(8, np.float32), stats)


# --- observation ----------------------------------------------------------------------------------


def test_build_observation_keys_and_shapes():
    obs = preprocess.build_observation(_snapshot(), CAMERAS, layout.PROMPT, pin={7: 0.0})
    assert set(obs) == {
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
        "observation/state",
        "prompt",
    }
    assert obs["prompt"] == layout.PROMPT
    assert obs["observation/state"].shape == (16,) and obs["observation/state"].dtype == np.float32
    assert obs["observation/state"][7] == 0.0
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        assert obs[key].shape == (224, 224, 3) and obs[key].dtype == np.uint8


def test_build_observation_uses_each_cameras_pipeline():
    snap = _snapshot()
    obs = preprocess.build_observation(snap, CAMERAS, layout.PROMPT, pin={7: 0.0})
    assert np.array_equal(obs["observation/image"], preprocess.to_policy_image(snap.images["head"], (336, 188)))
    assert np.array_equal(
        obs["observation/left_wrist_image"], preprocess.to_policy_image(snap.images["wrist_left"], None)
    )


def test_build_observation_reports_missing_image_and_state():
    snap = _snapshot()
    del snap.images["wrist_right"]
    with pytest.raises(ValueError, match="wrist_right"):
        preprocess.build_observation(snap, CAMERAS, layout.PROMPT, pin={7: 0.0})
    with pytest.raises(preprocess.StateError):
        preprocess.build_observation(_snapshot(knuckle={"right": 0.4}), CAMERAS, layout.PROMPT)


def test_crop_applied_before_resize():
    # Left half black, right half white; the crop keeps only the white half, whose head-resize is then white.
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 640:] = 255
    cam = CameraConfig(topic="/h", train_res=(336, 188), crop=(640, 0, 640, 720))
    snap = _snapshot(images={"head": frame, "wrist_left": frame[:480, :640], "wrist_right": frame[:480, :640]})
    obs = preprocess.build_observation(snap, {**CAMERAS, "head": cam}, layout.PROMPT, pin={7: 0.0})
    head = obs["observation/image"]
    assert head[112, 112].tolist() == [255, 255, 255]
    assert head.min() == 0  # letterbox bars (the crop is 640x720 -> resized to 336x188, then padded)


# --- to_policy_image ------------------------------------------------------------------------------


def _gradient(h, w):
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack([xs * 255 // (w - 1), ys * 255 // (h - 1), (xs + ys) % 256], axis=-1).astype(np.uint8)


def test_head_non_uniform_matches_pil_bilinear_then_letterbox():
    frame = _gradient(720, 1280)
    resized = np.asarray(Image.fromarray(frame).resize((336, 188), Image.BILINEAR))
    expected = image_tools.convert_to_uint8(image_tools.resize_with_pad(resized, 224, 224))
    assert np.array_equal(preprocess.to_policy_image(frame, (336, 188)), expected)


def test_head_at_train_res_is_not_resized():
    frame = _gradient(188, 336)
    expected = image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, 224, 224))
    assert np.array_equal(preprocess.to_policy_image(frame, (336, 188)), expected)


@pytest.mark.parametrize(
    "crop",
    [
        (0, 0, 0, 10),
        (0, 0, 10, 0),
        (-1, 0, 10, 10),
        (0, -5, 10, 10),
        (150, 0, 100, 10),
        (0, 60, 10, 50),
        (300, 300, 5, 5),
    ],
)
def test_bad_crop_raises_naming_crop_and_frame(crop):
    frame = np.zeros((100, 200, 3), np.uint8)
    with pytest.raises(ValueError, match="crop") as exc:
        preprocess.to_policy_image(frame, None, crop=crop)
    assert str(crop) in str(exc.value)
    assert "200x100" in str(exc.value)  # frame W x H


def test_crop_touching_frame_edge_is_valid():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert preprocess.to_policy_image(frame, None, crop=(100, 50, 100, 50)).shape == (224, 224, 3)
