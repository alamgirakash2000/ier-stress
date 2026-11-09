"""Checks the environment: name, CUDA, task availability, upstream imports and rliable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import current_env_name, require_ierstress  # noqa: E402

require_ierstress()


def main() -> int:
    failures = []

    def check(name, func):
        try:
            print(f"[ok]   {name}: {func()}")
        except Exception as exc:  # Report every failure, not only the first.
            failures.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    print(f"environment: {current_env_name()}, python {sys.version.split()[0]}")

    def cuda():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        x = torch.ones(1000, 1000, device="cuda")
        (x @ x).sum().item()
        return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, CUDA runtime {torch.version.cuda}"

    def gym_task(name):
        def run():
            import gymnasium as gym

            env = gym.make(name)
            env.reset(seed=0)
            obs, *_ = env.step(env.action_space.sample())
            return f"obs {obs.shape}, reset_noise_scale {env.unwrapped._reset_noise_scale}"

        return run

    def dmc_task(domain, task):
        def run():
            from dm_control import suite

            env = suite.load(domain, task, task_kwargs={"random": 0})
            env.reset()
            step = env.step(env.action_spec().minimum)
            size = sum(int(v.size) for v in step.observation.values())
            return f"obs {size}, act {env.action_spec().shape}, nq {env.physics.model.nq}, nv {env.physics.model.nv}"

        return run

    def upstream_imports():
        from iers import upstream

        upstream.setup()
        from environments.environment_factory import EnvironmentFactory  # noqa: F401
        from memory import EpisodicMemory  # noqa: F401
        from networks.network_factory import NetworkFactory  # noqa: F401
        from train_loops.ier import episode_reward_loop  # noqa: F401
        from utils.record import Record  # noqa: F401

        return "ok"

    def mj_state_api():
        import mujoco

        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='1'/></body></worldbody></mujoco>")
        data = mujoco.MjData(model)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        import numpy as np

        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        mujoco.mj_setState(model, data, state, spec)
        return f"mujoco {mujoco.__version__}, integration state size {state.size}"

    def rliable_run():
        import numpy as np
        from rliable import library as rly
        from rliable import metrics

        scores = {"a": np.random.default_rng(0).normal(1.0, 0.2, size=(10, 4))}
        point, ci = rly.get_interval_estimates(
            scores, lambda x: np.array([metrics.aggregate_iqm(x)]), reps=200
        )
        return f"IQM {point['a'][0]:.3f}, CI {np.round(ci['a'].ravel(), 3).tolist()}"

    def references():
        import sb3_contrib
        import stable_baselines3

        return f"stable-baselines3 {stable_baselines3.__version__}, sb3-contrib {sb3_contrib.__version__}"

    check("CUDA", cuda)
    check("HalfCheetah-v4", gym_task("HalfCheetah-v4"))
    check("Hopper-v4", gym_task("Hopper-v4"))
    check("walker-walk", dmc_task("walker", "walk"))
    check("finger-turn_hard", dmc_task("finger", "turn_hard"))
    check("upstream imports", upstream_imports)
    check("MuJoCo state API", mj_state_api)
    check("rliable", rliable_run)
    check("reference learners", references)

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
