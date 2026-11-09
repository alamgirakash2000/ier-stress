"""Replay check: do stored best episodes replay exactly, and how much do they keep
from fresh starts?

    python scripts/replay_check.py collect --tasks HalfCheetah-v4 Hopper-v4   short IER3 runs, every stored best
                                                                               episode saved with its start state
    python scripts/replay_check.py replay                                      replays, writes the CSV

The main runs did not keep the stored actions or full start states (they lived in checkpoints, which are deleted
when a run finishes). So `collect` trains IER3, `default`, 200k steps, seeds 300 and 301 (outside the main seeds
0 to 19 and the pilot seeds 100 to 104), with the unchanged training code. A thin wrapper around each environment
saves the full simulator state (MuJoCo integration state, random streams, step counters) after every reset. At
each trigger the episode that just ended is saved: its start state, its actions as stored by the repetition
controller, its observation path and its return.

`replay` plays each stored episode open-loop:
- from the identical start state, restored with the same `set_state` that resuming a run uses. This must give
  back the stored return exactly.
- from 20 fresh `default` resets of a separate environment.
It stops when the stored actions run out or the episode ends. Per-step deviation is the L2 norm of the
observation difference against the stored path, the measure the main runs use for `repeat_deviation`.

Writes results/aggregated/replay_check.csv (one row per replay, with the deviation at selected steps) and
results/aggregated/replay_check_deviation.npz (the full per-step deviation of every replay, keyed by row).
"""

import argparse
import copy
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

import numpy as np  # noqa: E402

from iers import ROOT  # noqa: E402

STORE = ROOT / "results" / "replay_check"
OUT = ROOT / "results" / "aggregated" / "replay_check.csv"
SEEDS = (300, 301)
BUDGET = 200_000
FRESH = 20
DEV_STEPS = (1, 10, 50, 100, 250, 500, 1000)


class Recorder:
    """Passes everything through to the environment, and keeps the full state saved right after the last reset."""

    def __init__(self, env):
        self.env = env
        self.last_reset_state = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        self.last_reset_state = copy.deepcopy(self.env.get_state())
        return obs


def collect(tasks: list[str]) -> int:
    import torch

    from iers.guard import ResourceGuard
    from iers.logging import get_logger
    from iers.runner import GroupRunner, Hardware, TrainConfig, default_env_factory, make_spec
    from iers.specs import load_tasks

    torch.set_num_threads(1)
    task_rn = load_tasks()
    logger = get_logger("replay_check")
    STORE.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        specs = [make_spec("replay_check", task, "default", "sac", "IER3", s, BUDGET, task_rn[task]["rn"]) for s in SEEDS]
        # No evaluations: they do not change what is stored, and skipping them halves the time.
        runner = GroupRunner(specs, TrainConfig(eval_every=10**9), Hardware("cuda", "cuda", label="replay check"),
                             logger=logger,
                             env_factory=lambda spec, streams: Recorder(default_env_factory(spec, streams)),
                             results_root=STORE / "unused", ckpt_root=STORE / "unused", write_outputs=False)
        # Heat and memory pauses as in every run, with a stop file of its own, so a pause of the main runs
        # (logs/STOP_REQUESTED) does not stop it.
        guard = ResourceGuard(log=logger.info, watch_gpu=True, stop_file=STORE / "STOP_REQUESTED")
        episodes, state = [], {}
        start = time.time()

        def tick(t: int) -> bool:
            for i, seed in enumerate(runner.seeds):
                s = state.setdefault(i, {"start": seed.env.last_reset_state, "ep": seed.ep_index,
                                         "triggers": len(seed.trigger_log)})
                if len(seed.trigger_log) > s["triggers"]:  # the episode that just ended set a new best
                    when, ret, length, kind = seed.trigger_log[-1]
                    episodes.append({"task": task, "seed": seed.spec.seed, "trigger": len(seed.trigger_log) - 1,
                                     "total_steps": when, "kind": kind, "stored_return": float(ret),
                                     "length": int(length), "actions": np.array(seed.ctrl.stored_actions),
                                     "path": np.array(seed.best_path), "start_state": s["start"]})
                    s["triggers"] = len(seed.trigger_log)
                if seed.ep_index != s["ep"]:  # a new training episode began; its reset was the last one
                    s["start"], s["ep"] = seed.env.last_reset_state, seed.ep_index
            return guard.check()

        runner.run(tick=tick)
        with open(STORE / f"episodes_{task}.pkl", "wb") as handle:
            pickle.dump(episodes, handle)
        logger.info("replay check %s: %d stored episodes from seeds %s in %.0f s", task, len(episodes), SEEDS,
                    time.time() - start)
        print(f"{task}: {len(episodes)} stored episodes, {time.time() - start:.0f} s", flush=True)
    return 0


def play(env, actions, start_state=None):
    """Open-loop replay. Returns the start observation, the observation path, the return and whether it ended."""
    from iers.envs import denormalize

    env.reset()
    if start_state is not None:
        env.set_state(copy.deepcopy(start_state))
        start_obs = start_state["last_start"].obs
    else:
        start_obs = env.last_start.obs
    path, total, ended = [], 0.0, False
    for action in actions:
        obs, reward, done, truncated = env.step(denormalize(action, env.max_action_value, env.min_action_value))
        path.append(np.array(obs))
        total += reward
        if done or truncated:
            ended = True
            break
    return np.asarray(start_obs), np.array(path), total, ended


def replay(tasks: list[str] | None = None) -> int:
    """With tasks, writes per-task files (replay_check_<task>.csv and .npz) for `merge`. Without, the merged ones."""
    import pandas as pd

    from iers.envs import make_env
    from iers.runner import Streams

    rows, curves = [], {}
    paths = sorted(STORE.glob("episodes_*.pkl"))
    if tasks:
        paths = [p for p in paths if p.stem.removeprefix("episodes_") in tasks]
    for path in paths:
        with open(path, "rb") as handle:
            episodes = pickle.load(handle)
        task = episodes[0]["task"] if episodes else path.stem.removeprefix("episodes_")
        started = time.time()
        for ep in episodes:
            streams = Streams.from_seed(ep["seed"])
            # The identical-start replay uses an environment built like the training one, then the saved state.
            same = make_env(task, "default", streams.env_seed, streams.action_seed, streams.cond)
            fresh = make_env(task, "default", 900_000 + 1000 * ep["seed"] + ep["trigger"], 0, np.random.default_rng(0))
            stored_start = ep["start_state"]["last_start"].obs
            for k in range(FRESH + 1):
                env, label = (same, "identical") if k == 0 else (fresh, f"fresh_{k:02d}")
                start_obs, obs_path, ret, ended = play(env, ep["actions"], ep["start_state"] if k == 0 else None)
                n = min(len(obs_path), len(ep["path"]))
                deviation = np.linalg.norm(obs_path[:n] - ep["path"][:n], axis=1)
                key = f"{task}|{ep['seed']}|{ep['trigger']}|{label}"
                curves[key] = deviation.astype(np.float32)
                row = {"task": task, "seed": ep["seed"], "trigger": ep["trigger"], "trigger_at_step": ep["total_steps"],
                       "episode_kind": ep["kind"], "stored_return": ep["stored_return"], "stored_length": ep["length"],
                       "start": label, "start_distance": float(np.linalg.norm(start_obs - stored_start)),
                       "replay_return": ret, "replay_length": len(obs_path),
                       "ended_before_actions_ran_out": bool(ended and len(obs_path) < len(ep["actions"])),
                       "return_difference": ret - ep["stored_return"],
                       "retention_ratio": ret / ep["stored_return"] if ep["stored_return"] else np.nan,
                       "exact_return": bool(ret == ep["stored_return"]),
                       "max_deviation": float(deviation.max()) if n else np.nan,
                       "mean_deviation": float(deviation.mean()) if n else np.nan}
                for step in DEV_STEPS:
                    row[f"deviation_step_{step}"] = float(deviation[step - 1]) if n >= step else np.nan
                rows.append(row)
        print(f"{task}: {len(episodes)} episodes replayed {FRESH + 1} times each in {time.time() - started:.0f} s",
              flush=True)
    table = pd.DataFrame(rows)
    out = OUT if not tasks else OUT.with_name(f"replay_check_{'_'.join(tasks)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")
    np.savez_compressed(out.with_name(out.stem.replace("replay_check", "replay_check_deviation") + ".npz"), **curves)
    print(f"{len(table)} replays written to {out}")
    return 0


def merge() -> int:
    """Joins the per-task files into results/aggregated/replay_check.csv and replay_check_deviation.npz."""
    import pandas as pd

    parts = sorted(OUT.parent.glob("replay_check_*.csv"))
    parts = [p for p in parts if not p.stem.startswith("replay_check_deviation")]
    table = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    table.to_csv(OUT, index=False, lineterminator="\n")
    curves = {}
    for p in sorted(OUT.parent.glob("replay_check_deviation_*.npz")):
        with np.load(p) as data:
            curves.update({k: data[k] for k in data.files})
    np.savez_compressed(OUT.with_name("replay_check_deviation.npz"), **curves)
    for p in parts + sorted(OUT.parent.glob("replay_check_deviation_*.npz")):
        p.unlink()
    print(f"{len(table)} replays from {len(parts)} task files merged into {OUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["collect", "replay", "merge"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="folder for a new check, default results. Keeps the saved replay check intact")
    args = parser.parse_args()
    if args.out is not None:
        global STORE, OUT
        STORE, OUT = args.out / "replay_check", args.out / "aggregated" / "replay_check.csv"
    if args.command == "collect":
        return collect(args.tasks or ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"])
    return replay(args.tasks) if args.command == "replay" else merge()


if __name__ == "__main__":
    sys.exit(main())
