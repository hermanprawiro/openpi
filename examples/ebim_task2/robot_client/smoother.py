"""Joint-space smoothing between the 20 Hz policy waypoints (linearly interpolated by the streamer) and
the 100 Hz command stream.

`RuckigSmoother` (jerk-limited, online ruckig) is the default; `RateLimiter` (velocity- and
acceleration-limited, pure numpy) is used for `impl: clamp` and as the fallback when ruckig is not installed.
Targets arrive already limit-clamped, so a smoother's internal state is always what gets published.
Any invalid input or solver failure raises `SmootherError`; the streamer turns that into a fault.

Why `target_vel` (the interpolation segment's velocity) is accepted but not fed forward: the target
is piecewise linear and in sync mode stops dead at every chunk end. Feeding its velocity forward
made both smoothers overshoot and come back by up to v^2 / (2 a) at those stops (~0.1 rad at
1 rad/s, a = 5 rad/s^2). Ruckig has a second problem: it plans to a fixed (position, velocity)
state, and a state just ahead at a slightly higher speed can only be reached by looping back, so it
braked hard mid-motion (from 1.0 to 0.08 rad/s while the target kept moving). Re-planning every
period to rest at the target gives no extra direction changes on ramps, sines and stop-and-go
chunks. The cost is a lag of about the braking distance while moving.
"""

import logging
from typing import Protocol

import numpy as np

from robot_client.config import SmootherConfig

logger = logging.getLogger(__name__)


class SmootherError(RuntimeError):
    """Invalid smoother input or a trajectory-generation failure."""


class Smoother(Protocol):
    def reset(self, q: np.ndarray) -> None:
        """Start at rest at `q` and leave the stopped state."""

    def step(self, target: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """Advance one period towards `target` (moving at `target_vel`); return the new position."""

    def stop(self) -> None:
        """From now until `reset`, ignore targets and decelerate to rest wherever that happens to be."""


def _limits(name: str, value, dofs: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or (dofs is not None and arr.shape != (dofs,)):
        raise ValueError(f"{name}: expected {dofs or 'a 1-D array of'} values, got shape {arr.shape}")
    if not np.all(np.isfinite(arr) & (arr > 0)):
        raise ValueError(f"{name}: every limit must be finite and > 0, got {arr.tolist()}")
    return arr


class _Base:
    def __init__(self, v_max, a_max, dt: float):
        if not dt > 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.v_max = _limits("v_max", v_max)
        self.dofs = len(self.v_max)
        self.a_max = _limits("a_max", a_max, self.dofs)
        self.dt = float(dt)
        self._stopping = False
        self._ready = False

    def _check(self, name: str, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (self.dofs,):
            raise SmootherError(f"{name}: shape {arr.shape} does not match ({self.dofs},)")
        if not np.all(np.isfinite(arr)):
            raise SmootherError(f"{name}: not finite: {arr.tolist()}")
        return arr

    def _check_step(self, target: np.ndarray, target_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._ready:
            raise SmootherError("step() before reset(): the smoother has no start state")
        return self._check("target", target), self._check("target_vel", target_vel)

    def stop(self) -> None:
        self._stopping = True


class RateLimiter(_Base):
    """Per-joint velocity and acceleration limits (no jerk limit).

    Each period it picks the fastest velocity from which the joint can still brake onto the target in
    whole periods at `a_max` (exact in discrete time, so it lands on the target without overshoot), then
    moves the current velocity towards that by at most `a_max * dt`. `target_vel` is not fed forward
    (see the module docstring).
    """

    def reset(self, q: np.ndarray) -> None:
        self._q = self._check("q", q).copy()
        self._v = np.zeros(self.dofs)
        self._stopping = False
        self._ready = True

    def _braking_speed(self, dist: np.ndarray) -> np.ndarray:
        # In units of u = a*dt^2 (distance) and a*dt (speed): moving at w and then braking by 1 per period
        # (w, w-1, ..., w-m with m = floor(w)) covers (m+1)*w - m*(m+1)/2. The largest w covering at most
        # x = dist/u has m = floor((sqrt(1+8x) - 1) / 2) and w = (x + m*(m+1)/2) / (m+1); for x < 1
        # that is w = x, i.e. land on the target in one period.
        dv = self.a_max * self.dt
        x = dist / (dv * self.dt)
        m = np.floor((np.sqrt(1.0 + 8.0 * x) - 1.0) / 2.0)
        return dv * (x + m * (m + 1.0) / 2.0) / (m + 1.0)

    def step(self, target: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        target, _ = self._check_step(target, target_vel)
        if self._stopping:
            v_des = np.zeros(self.dofs)
        else:
            err = target - self._q
            v_des = np.clip(np.sign(err) * self._braking_speed(np.abs(err)), -self.v_max, self.v_max)
        dv_max = self.a_max * self.dt
        self._v = self._v + np.clip(v_des - self._v, -dv_max, dv_max)
        self._q = self._q + self._v * self.dt
        return self._q.copy()


class RuckigSmoother(_Base):
    """Jerk-limited online trajectory generation with ruckig (time-synchronised across the joints).

    Every period ruckig re-plans from its current state to rest at `target`; only the first period of
    the plan is executed, so a moving target is followed at its own speed (lagging by about the braking
    distance) and an abrupt stop lands exactly on it. `target_vel` is not fed forward (see the module
    docstring). `stop()` switches ruckig to its velocity interface with a zero target: a jerk-limited
    brake that never turns back.
    """

    def __init__(self, v_max, a_max, j_max, dt: float):
        import ruckig  # optional dependency; make_smoother falls back when it is missing

        super().__init__(v_max, a_max, dt)
        self.j_max = _limits("j_max", j_max, self.dofs)
        self._ruckig = ruckig
        self._otg = ruckig.Ruckig(self.dofs, self.dt)
        self._inp = ruckig.InputParameter(self.dofs)
        self._out = ruckig.OutputParameter(self.dofs)
        self._inp.max_velocity = self.v_max.tolist()
        self._inp.max_acceleration = self.a_max.tolist()
        self._inp.max_jerk = self.j_max.tolist()

    def reset(self, q: np.ndarray) -> None:
        q = self._check("q", q)
        zeros = [0.0] * self.dofs
        self._otg.reset()
        self._inp.control_interface = self._ruckig.ControlInterface.Position
        self._inp.current_position = q.tolist()
        self._inp.current_velocity = zeros
        self._inp.current_acceleration = zeros
        self._inp.target_position = q.tolist()
        self._inp.target_velocity = zeros
        self._inp.target_acceleration = zeros
        self._q = q.copy()
        self._stopping = False
        self._ready = True

    def stop(self) -> None:
        super().stop()
        self._inp.control_interface = self._ruckig.ControlInterface.Velocity
        self._inp.target_velocity = [0.0] * self.dofs
        self._inp.target_acceleration = [0.0] * self.dofs

    def step(self, target: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        target, _ = self._check_step(target, target_vel)
        if not self._stopping:
            self._inp.target_position = target.tolist()
        try:
            result = self._otg.update(self._inp, self._out)
        except Exception as e:  # ruckig raises RuckigError on some invalid inputs
            raise SmootherError(f"ruckig update failed: {e}") from e
        if result not in (self._ruckig.Result.Working, self._ruckig.Result.Finished):
            raise SmootherError(
                f"ruckig returned {getattr(result, 'name', result)} "
                f"(position {self._inp.current_position}, target {self._inp.target_position})"
            )
        self._out.pass_to_input(self._inp)
        self._q = np.array(self._out.new_position, dtype=np.float64)
        return self._q.copy()


def make_smoother(cfg: SmootherConfig, dt: float) -> Smoother:
    """The configured smoother with every limit scaled by `cfg.velocity_scale`; `RateLimiter` (with a
    warning) if ruckig is requested but not importable."""
    scale = cfg.velocity_scale
    v_max, a_max, j_max = (np.asarray(x, dtype=np.float64) * scale for x in (cfg.v_max, cfg.a_max, cfg.j_max))
    if cfg.impl == "ruckig":
        try:
            return RuckigSmoother(v_max, a_max, j_max, dt)
        except ImportError as e:
            logger.warning("smoother.impl is 'ruckig' but ruckig is not importable (%s); using RateLimiter", e)
    return RateLimiter(v_max, a_max, dt)
