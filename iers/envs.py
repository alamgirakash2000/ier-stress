"""Environments, built on the upstream OpenAIEnvironment and DMCSEnvironment classes.

The wrappers add what the study needs and keep upstream behaviour otherwise:
- the test conditions (reset noise, dynamics changes)
- a record of each episode's start state and dynamics factors
- per-seed random streams
- exact save and restore of the simulator, for resuming runs
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import gymnasium as gym
import mujoco
import numpy as np

from . import upstream
from .conditions import CONDITIONS, DMC_NOISE_BASE, Condition, DynamicsScaler
from .specs import SPREADS_FILE, TASKS_FILE, load_spreads, load_tasks  # noqa: F401

upstream.setup()

from environments.dm_control.dmcs_environment import DMCSEnvironment  # noqa: E402
from environments.mujoco.openai_environment import OpenAIEnvironment  # noqa: E402
from utils.configurations import GymEnvironmentConfig  # noqa: E402
from utils.helpers import denormalize, normalize  # noqa: E402,F401

_MJ_STATE = mujoco.mjtState.mjSTATE_INTEGRATION
_MODEL_FIELDS = ("body_mass", "body_inertia", "geom_friction", "site_pos", "site_size")


@dataclass
class StartInfo:
    """Everything recorded about the start of one episode."""

    obs: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    factors: dict | None


class _ManagedMixin:
    condition: Condition
    last_start: StartInfo

    def _mj(self):
        raise NotImplementedError

    def _setup_condition(self, condition: Condition, rng: np.random.Generator) -> None:
        self.condition = condition
        self.cond_rng = rng
        self.scaler = None
        if condition.dyn_low is not None:
            model, data = self._mj()
            self.scaler = DynamicsScaler(model, data, condition.dyn_low, condition.dyn_high, rng)
        self.factors = None

    def _before_reset(self) -> None:
        if self.scaler is not None:
            self.factors = self.scaler.apply()

    def _record_start(self, obs: np.ndarray) -> np.ndarray:
        _, data = self._mj()
        self.last_start = StartInfo(
            obs=np.array(obs), qpos=np.array(data.qpos), qvel=np.array(data.qvel), factors=self.factors
        )
        return obs

    def _sim_state(self) -> dict:
        model, data = self._mj()
        state = np.empty(mujoco.mj_stateSize(model, _MJ_STATE))
        mujoco.mj_getState(model, data, state, _MJ_STATE)
        return {
            "mj_state": state,
            "model": {name: np.array(getattr(model, name)) for name in _MODEL_FIELDS},
            "cond_rng": self.cond_rng.bit_generator.state,
            "factors": self.factors,
            "last_start": self.last_start,
        }

    def _restore_sim(self, saved: dict) -> None:
        model, data = self._mj()
        for name, value in saved["model"].items():
            getattr(model, name)[:] = value
        mujoco.mj_setConst(model, data)
        mujoco.mj_setState(model, data, saved["mj_state"], _MJ_STATE)
        mujoco.mj_forward(model, data)
        self.cond_rng.bit_generator.state = saved["cond_rng"]
        self.factors = saved["factors"]
        self.last_start = saved["last_start"]


class GymEnv(_ManagedMixin, OpenAIEnvironment):
    """Gymnasium MuJoCo task. Reset noise scales the task's own `reset_noise_scale`."""

    def __init__(self, task: str, condition: Condition, seed: int, action_seed: int, cond_rng):
        OpenAIEnvironment.__init__(self, GymEnvironmentConfig(task=task, gym="openai"))
        self.default_noise_scale = self.env.unwrapped._reset_noise_scale
        self.env.unwrapped._reset_noise_scale = self.default_noise_scale * condition.noise_mult
        self._setup_condition(condition, cond_rng)
        # Same seeding calls as the upstream set_seed.
        self.env.reset(seed=seed)
        self.env.action_space.seed(action_seed)

    def _mj(self):
        unwrapped = self.env.unwrapped
        return unwrapped.model, unwrapped.data

    def reset(self) -> np.ndarray:
        self._before_reset()
        return self._record_start(OpenAIEnvironment.reset(self))

    def _time_limit(self):
        wrapper = self.env
        while not isinstance(wrapper, gym.wrappers.TimeLimit):
            wrapper = wrapper.env
        return wrapper

    def get_state(self) -> dict:
        saved = self._sim_state()
        saved["np_random"] = self.env.unwrapped.np_random.bit_generator.state
        saved["action_rng"] = self.env.action_space.np_random.bit_generator.state
        saved["elapsed"] = self._time_limit()._elapsed_steps
        return saved

    def set_state(self, saved: dict) -> None:
        self._restore_sim(saved)
        self.env.unwrapped.np_random.bit_generator.state = saved["np_random"]
        self.env.action_space.np_random.bit_generator.state = saved["action_rng"]
        self._time_limit()._elapsed_steps = saved["elapsed"]


class DmcEnv(_ManagedMixin, DMCSEnvironment):
    """DeepMind Control task, with the reset noise of the noise conditions.

    The built-in reset runs first. Then qvel gets noise N(0, (m x 0.1)^2) on every degree of freedom,
    qpos gets the same noise on unlimited joints only, and limited joints are clipped into their limits.
    """

    def __init__(self, domain: str, task: str, condition: Condition, seed: int, action_seed: int, cond_rng):
        DMCSEnvironment.__init__(self, GymEnvironmentConfig(task=task, gym="dmcs", domain=domain))
        # Same call as the upstream set_seed.
        self.set_seed(seed)
        self.action_rng = np.random.default_rng(action_seed)
        self._setup_condition(condition, cond_rng)
        self.noise_std = None
        if condition.noise_mult != 1.0:
            self.noise_std = condition.noise_mult * DMC_NOISE_BASE
            model, _ = self._mj()
            limited = np.asarray(model.jnt_limited, dtype=bool)
            kinds = np.asarray(model.jnt_type)
            simple = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            if not np.isin(kinds, simple).all():
                raise ValueError(f"{domain}-{task} has free or ball joints, which this noise rule does not cover.")
            adr = np.asarray(model.jnt_qposadr)
            self.free_adr = adr[~limited]
            self.limit_adr = adr[limited]
            self.limit_low = np.asarray(model.jnt_range)[limited, 0]
            self.limit_high = np.asarray(model.jnt_range)[limited, 1]

    def _mj(self):
        physics = self.env.physics
        return physics.model.ptr, physics.data.ptr

    def sample_action(self) -> np.ndarray:
        # Upstream draws from the global numpy generator. A per-seed generator gives the
        # same distribution and keeps seeds independent inside a batched group.
        return self.action_rng.uniform(self.min_action_value, self.max_action_value, size=self.action_num)

    def reset(self) -> np.ndarray:
        self._before_reset()
        obs = DMCSEnvironment.reset(self)
        if self.noise_std is not None:
            physics = self.env.physics
            qpos, qvel = physics.data.qpos, physics.data.qvel
            qpos[self.free_adr] += self.cond_rng.normal(0.0, self.noise_std, size=len(self.free_adr))
            qpos[self.limit_adr] = np.clip(qpos[self.limit_adr], self.limit_low, self.limit_high)
            qvel[:] += self.cond_rng.normal(0.0, self.noise_std, size=len(qvel))
            physics.after_reset()
            obs = np.hstack(list(self.env.task.get_observation(physics).values()))
        return self._record_start(obs)

    def get_state(self) -> dict:
        saved = self._sim_state()
        saved["task_random"] = self.env.task.random.get_state()
        saved["action_rng"] = self.action_rng.bit_generator.state
        saved["step_count"] = self.env._step_count
        saved["reset_next"] = self.env._reset_next_step
        return saved

    def set_state(self, saved: dict) -> None:
        self._restore_sim(saved)
        self.env.task.random.set_state(saved["task_random"])
        self.action_rng.bit_generator.state = saved["action_rng"]
        self.env._step_count = saved["step_count"]
        self.env._reset_next_step = saved["reset_next"]


def make_env(task_name: str, condition_name: str, seed: int, action_seed: int, cond_rng):
    tasks = load_tasks()
    spec = tasks[task_name]
    condition = CONDITIONS[condition_name]
    if spec["backend"] == "gym":
        return GymEnv(spec["task"], condition, seed, action_seed, cond_rng)
    return DmcEnv(spec["domain"], spec["task"], condition, seed, action_seed, cond_rng)


def env_dims(env) -> tuple[int, int]:
    return int(env.observation_space), int(env.action_num)
