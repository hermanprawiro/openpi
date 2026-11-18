import numpy as np
from robot_client import preprocess


def test_head_resized_to_train_res_then_letterboxed():
    frame = np.full((720, 1280, 3), 200, np.uint8)
    out = preprocess.to_policy_image(frame, (336, 188))
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    native = preprocess.to_policy_image(np.full((188, 336, 3), 200, np.uint8), (336, 188))
    assert np.array_equal(out, native)  # uniform image: identical after both paths


def test_wrist_letterbox_only():
    assert preprocess.to_policy_image(np.zeros((480, 640, 3), np.uint8), None).shape == (224, 224, 3)


def test_crop_applied_before_resize():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, 100:] = 255
    out = preprocess.to_policy_image(frame, None, crop=(100, 0, 100, 100))
    assert out.shape == (224, 224, 3)
    assert out.min() == 255  # only the white half survives the crop; square crop has no letterbox bars
