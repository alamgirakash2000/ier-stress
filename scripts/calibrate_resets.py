"""Measures the spreads that normalize the start distance on every task (Sections 5.4 and 5.5).

Writes configs/tasks/reset_spreads.yaml with two kinds of numbers:
- the spread of 1000 default resets, as a description of each task's built-in reset
- `obs_scale`: the spread of each observation dimension over random-action rollouts under `default`

The start distance divides by `obs_scale`. The spread of default resets cannot do this job. It is zero
on every velocity dimension of the DeepMind Control tasks, and the version 2 noise acts on exactly
those dimensions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from iers.envs import SPREADS_FILE, load_tasks, make_env  # noqa: E402

N_RESETS = 1000
N_ROLLOUT_STEPS = 20_000
CALIBRATION_SEED = 987654


def main() -> None:
    out = {}
    for name in load_tasks():
        env = make_env(name, "default", CALIBRATION_SEED, CALIBRATION_SEED + 1,
                       np.random.default_rng(CALIBRATION_SEED + 2))
        obs, qpos, qvel = [], [], []
        for _ in range(N_RESETS):
            env.reset()
            start = env.last_start
            obs.append(start.obs)
            qpos.append(start.qpos)
            qvel.append(start.qvel)
        visited = []
        env.reset()
        while len(visited) < N_ROLLOUT_STEPS:
            state, _, done, truncated = env.step(env.sample_action())
            visited.append(state)
            if done or truncated:
                env.reset()
        out[name] = {
            "n_resets": N_RESETS,
            "n_rollout_steps": N_ROLLOUT_STEPS,
            "calibration_seed": CALIBRATION_SEED,
            "qpos_std": np.std(qpos, axis=0).round(8).tolist(),
            "qvel_std": np.std(qvel, axis=0).round(8).tolist(),
            "obs_std": np.std(obs, axis=0).round(8).tolist(),
            "obs_scale": np.std(visited, axis=0).round(8).tolist(),
        }
        print(name)
        for key in ("qpos_std", "qvel_std", "obs_std", "obs_scale"):
            print(f"  {key}: {np.round(out[name][key], 4).tolist()}")
    SPREADS_FILE.write_text(yaml.safe_dump(out, sort_keys=True), encoding="utf-8")
    print(f"wrote {SPREADS_FILE}")


if __name__ == "__main__":
    main()
