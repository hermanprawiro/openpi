"""Command adapter: smoothed joint targets -> what goes on the command topic (spec §6.5, `command_mode`).

`absolute` publishes the targets as-is. `relative_to_activation` publishes
`ref_cmd + signs * (target - q_activation)`, the reference captured by `activate` (called by takeover
immediately before the controller is switched on, with `ref_cmd` = the last command streamed); before
`activate` and after `deactivate` it is the identity, so the hold stream is continuous at activation.
"""

import numpy as np

MODES = ("absolute", "relative_to_activation")


def _vector(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.array(value, dtype=np.float64)  # a copy: later changes by the caller must not leak in
    if arr.shape != shape:
        raise ValueError(f"{name}: shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: not finite: {arr.tolist()}")
    return arr


class CommandAdapter:
    def __init__(self, mode: str, signs: np.ndarray):
        if mode not in MODES:
            raise ValueError(f"command mode must be one of {MODES}, got {mode!r}")
        signs = np.asarray(signs, dtype=np.float64)
        if signs.ndim != 1 or not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError(f"signs must be a 1-D array of 1 / -1, got {signs.tolist()}")
        self.mode = mode
        self.signs = signs
        # (q_activation, ref_cmd), replaced as one object so the streamer thread never sees half of it.
        self._reference: tuple[np.ndarray, np.ndarray] | None = None

    def activate(self, q_activation: np.ndarray, ref_cmd: np.ndarray) -> None:
        shape = self.signs.shape
        self._reference = (_vector("q_activation", q_activation, shape), _vector("ref_cmd", ref_cmd, shape))

    def deactivate(self) -> None:
        self._reference = None

    def __call__(self, target: np.ndarray) -> np.ndarray:
        target = np.array(target, dtype=np.float64)
        reference = self._reference
        if self.mode == "absolute" or reference is None:
            return target
        q_activation, ref_cmd = reference
        return ref_cmd + self.signs * (target - q_activation)
