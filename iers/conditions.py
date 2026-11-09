"""Test conditions: reset noise and per-episode dynamics changes."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


# Base of the DeepMind Control reset noise. The standard deviation is
# the multiplier times this value. It matches the Gymnasium HalfCheetah default.
DMC_NOISE_BASE = 0.1


@dataclass(frozen=True)
class Condition:
    name: str
    noise_mult: float = 1.0  # a label on the noise size, Q4 is analysed against measured start distance
    dyn_low: float | None = None  # lower factor for per-episode dynamics scaling
    dyn_high: float | None = None


CONDITIONS = {
    "default": Condition("default"),
    "noise5": Condition("noise5", noise_mult=5.0),
    "noise10": Condition("noise10", noise_mult=10.0),
    "dyn20": Condition("dyn20", dyn_low=0.8, dyn_high=1.2),
}


class DynamicsScaler:
    """Scales every body mass and every geom friction by independent per-episode factors.

    A body's inertia tensor is scaled with its mass, so each scaled body stays physically
    consistent. Each geom's three friction coefficients share one factor.
    """

    def __init__(self, model, data, low: float, high: float, rng: np.random.Generator):
        self.model, self.data = model, data
        self.low, self.high = low, high
        self.rng = rng
        self.base_mass = np.array(model.body_mass)
        self.base_inertia = np.array(model.body_inertia)
        self.base_friction = np.array(model.geom_friction)

    def apply(self) -> dict:
        mass_f = self.rng.uniform(self.low, self.high, size=self.model.nbody)
        friction_f = self.rng.uniform(self.low, self.high, size=self.model.ngeom)
        self.model.body_mass[:] = self.base_mass * mass_f
        self.model.body_inertia[:] = self.base_inertia * mass_f[:, None]
        self.model.geom_friction[:] = self.base_friction * friction_f[:, None]
        mujoco.mj_setConst(self.model, self.data)
        return {"mass": mass_f, "friction": friction_f}
