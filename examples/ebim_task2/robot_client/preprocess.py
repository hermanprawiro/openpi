"""Camera-frame preprocessing for the policy. Must match what the converter bakes into the dataset."""

import numpy as np
from openpi_client import image_tools
from PIL import Image

POLICY_IMAGE_SIZE = 224


def to_policy_image(
    frame: np.ndarray,
    train_res: tuple[int, int] | None,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """HxWx3 uint8 RGB -> 224x224x3 uint8.

    `crop=(x, y, w, h)` is applied first (a site knob; the converter never crops). If `train_res=(W, H)` is
    given and differs from the frame size, the frame is PIL-BILINEAR resized to it (the resolution the
    training images were stored at). Finally the frame is letterboxed to 224x224.
    """
    if crop is not None:
        x, y, w, h = crop
        frame = frame[y : y + h, x : x + w]
    if train_res is not None and (frame.shape[1], frame.shape[0]) != tuple(train_res):
        frame = np.asarray(Image.fromarray(frame).resize(tuple(train_res), Image.BILINEAR))
    frame = image_tools.resize_with_pad(frame, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)
    return image_tools.convert_to_uint8(frame)
