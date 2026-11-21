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
    # Number of real action dims. 20 for the sim layout, 16 for the Munich real-robot layout.
    action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        # Model actions are padded to the model action dim (32); the first `action_dim` are real.
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


# Munich real-robot layout (munich16, see examples/ebim_task2/robot_client/layout.py):
#   [0:7] L arm | [7] L gripper | [8:15] R arm | [15] R gripper
# State grippers are knuckle angles in rad (0 = open), action grippers are width % (high = open).
MUNICH_DIM = 16


def _munich_to_sim(x: np.ndarray, *, negate_grippers: bool) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[-1] != MUNICH_DIM:
        raise ValueError(f"expected a {MUNICH_DIM}-dim munich16 vector, got shape {x.shape}")
    out = np.zeros((*x.shape[:-1], ACTION_DIM), dtype=x.dtype)
    out[..., 3:10] = x[..., 0:7]
    out[..., 10:17] = x[..., 8:15]
    grippers = x[..., [7, 15]]
    out[..., 17:19] = -grippers if negate_grippers else grippers
    return out


@dataclasses.dataclass(frozen=True)
class MunichToSimLayout(transforms.DataTransformFn):
    """Moves munich16 state/actions into the 20-dim sim slots (input side).

    Lets a sim-fine-tuned checkpoint be fine-tuned on (and served for) the real robot with every joint
    in the slot, and in the state-token position, that the sim model learned it in. Base twist and spine
    (unused on the real robot) are zero. State grippers are negated so that, like the sim open-fraction,
    they rise as the gripper opens; action grippers (width %) already point that way. The client still
    speaks munich16: SimToMunichLayout undoes this on the output side.
    """

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        data["state"] = _munich_to_sim(data["state"], negate_grippers=True)
        if "actions" in data:
            data["actions"] = _munich_to_sim(data["actions"], negate_grippers=False)
        return data


@dataclasses.dataclass(frozen=True)
class SimToMunichLayout(transforms.DataTransformFn):
    """Output side of MunichToSimLayout: padded model actions in sim slots -> munich16 actions."""

    def __call__(self, data: dict) -> dict:
        a = np.asarray(data["actions"])
        return {"actions": np.concatenate([a[..., 3:10], a[..., 17:18], a[..., 10:17], a[..., 18:19]], axis=-1)}
