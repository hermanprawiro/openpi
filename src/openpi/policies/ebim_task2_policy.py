import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Action layout (20 dims, shared by state): base twist vx,vy,wz | left arm joints 1-7 |
# right arm joints 1-7 | left gripper open-fraction | right gripper open-fraction | spine height.
ACTION_DIM = 20


def make_ebim_task2_example() -> dict:
    """Creates a random input example for the EBIM task2 policy."""
    return {
        "observation/state": np.random.rand(ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the thermal pad, place it on the target RAM board, and go back to starting position.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class EbimTask2Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class EbimTask2Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Model actions are padded to the model action dim (32); the first 20 are real.
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
