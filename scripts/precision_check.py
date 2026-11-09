"""Precision check (exploratory): how much does a rounding-level change to the stored actions change a replay?

    python scripts/precision_check.py --tasks HalfCheetah-v4 Hopper-v4 walker-walk finger-turn_hard
    python scripts/precision_check.py --summary

For each task this trains one short IER3 run under `default` on the CPU (seed 0, 100k steps, or 60k steps on
finger-turn_hard, where the trigger stops early) with the unchanged training code, and keeps every stored best
episode with the full simulator state at its start. For up to 12 stored episodes spread over training it then
replays the stored actions open-loop:
- from the identical start state, once with the actions as stored and once with them cast to float64 before
  scaling to the action range, a change at the level of rounding,
- from 20 fresh `default` resets.
Deviation is the root mean square of the observation difference to the replay from the identical start, per
dimension in units of the spread `obs_scale` (Equation 3 of the paper).

Writes results/precision_check/<task>_s<seed>_rows.json, _curves.json and _captured.pkl, and with --summary
results/precision_check/summary.json. The figure code reads these files (iers/plotting/paper/sim.py).
"""

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

import torch  # noqa: E402

from iers import runner as R  # noqa: E402
from iers.envs import denormalize, make_env  # noqa: E402
from iers.metrics import obs_scale  # noqa: E402
from iers.specs import Hardware, TrainConfig, load_tasks, make_spec  # noqa: E402

OUT = ROOT / "results" / "precision_check"
BUDGET = {"HalfCheetah-v4": 100_000, "Hopper-v4": 100_000, "walker-walk": 100_000, "finger-turn_hard": 60_000}
K = 20       # fresh resets per stored episode
N_TEST = 12  # stored episodes tested per run, spread over training


def run_task(task: str, seed: int, out: Path) -> None:
    torch.set_num_threads(1)
    captured = []
    orig_reset = R.GroupRunner._safe_reset
    orig_end = R.GroupRunner._end_episode

    def _safe_reset(self, i):
        obs = orig_reset(self, i)
        s = self.seeds[i]
        if not s.failed:
            s._start_state = copy.deepcopy(s.env.get_state())
        return obs

    def _end_episode(self, i, t):
        s = self.seeds[i]
        before = s.ctrl.stored_actions
        ret, first_src = float(s.ep_return), s.ep_first_source
        start_state = getattr(s, "_start_state", None)
        orig_end(self, i, t)
        if s.ctrl.stored_actions is not before:
            captured.append(dict(t=t + 1, ret=ret, first_src=first_src,
                                 actions=[np.array(a, copy=True) for a in s.ctrl.stored_actions], start=start_state))

    R.GroupRunner._safe_reset = _safe_reset
    R.GroupRunner._end_episode = _end_episode
    tasks = load_tasks()
    spec = make_spec("precisioncheck", task, "default", "sac", "IER3", seed, BUDGET[task], tasks[task]["rn"])
    train = TrainConfig(eval_every=10**9, checkpoint_every=10**9)
    hw = Hardware(device="cpu", replay_device="cpu")
    t0 = time.time()
    group = R.GroupRunner([spec], train, hw, results_root=out / "raw", ckpt_root=out / "ckpt", write_outputs=False)
    group.run()
    R.GroupRunner._safe_reset, R.GroupRunner._end_episode = orig_reset, orig_end
    print(f"{task}: trained {BUDGET[task]} steps in {time.time() - t0:.0f} s, {len(captured)} stored episodes", flush=True)

    scale = obs_scale(task)
    used = scale > 0

    def run_actions(env, actions):
        obs, ret = [], 0.0
        for a in actions:
            nxt, r, done, trunc = env.step(denormalize(a, env.max_action_value, env.min_action_value))
            obs.append(np.array(nxt, dtype=np.float64))
            ret += float(r)
            if done or trunc:
                break
        return ret, np.array(obs)

    def dev(a, b):
        n = min(len(a), len(b))
        d = (a[:n, used] - b[:n, used]) / scale[used]
        return np.sqrt(np.mean(d ** 2, axis=1))

    picks = sorted(set(np.linspace(0, len(captured) - 1, min(N_TEST, len(captured))).round().astype(int).tolist()))
    check_env = make_env(task, "default", 10_000 + seed, 20_000 + seed, np.random.default_rng(30_000 + seed))
    fresh_env = make_env(task, "default", 40_000 + seed, 50_000 + seed, np.random.default_rng(60_000 + seed))
    rows, curves = [], []
    for k in picks:
        c = captured[k]
        check_env.reset()
        check_env.set_state(copy.deepcopy(c["start"]))
        same_ret, ref_obs = run_actions(check_env, c["actions"])
        check_env.reset()
        check_env.set_state(copy.deepcopy(c["start"]))
        f64_ret, f64_obs = run_actions(check_env, [np.asarray(a, dtype=np.float64) for a in c["actions"]])
        f64_dev = dev(f64_obs, ref_obs)
        stored_start = c["start"]["last_start"].obs
        for j in range(K):
            start = fresh_env.reset()
            ret, obs = run_actions(fresh_env, c["actions"])
            sd = float(np.sqrt(np.mean(((np.asarray(start)[used] - np.asarray(stored_start)[used]) / scale[used]) ** 2)))
            rows.append(dict(task=task, seed=seed, trigger_step=c["t"], trigger_source=c["first_src"],
                             stored_return=c["ret"], identical_return=same_ret,
                             identical_abs_err=abs(same_ret - c["ret"]), f64_return=f64_ret,
                             f64_abs_err=abs(f64_ret - c["ret"]), stored_length=len(c["actions"]),
                             action_dtype=str(np.asarray(c["actions"][0]).dtype), fresh=j, fresh_return=ret,
                             fresh_length=len(obs), ratio=ret / c["ret"] if c["ret"] != 0 else float("nan"),
                             start_distance=sd))
            curves.append(dict(task=task, trigger_step=c["t"], fresh=j, dev=dev(obs, ref_obs).tolist()))
        curves.append(dict(task=task, trigger_step=c["t"], fresh=-1, dev=f64_dev.tolist()))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task}_s{seed}_rows.json").write_text(json.dumps(rows))
    (out / f"{task}_s{seed}_curves.json").write_text(json.dumps(curves))
    with open(out / f"{task}_s{seed}_captured.pkl", "wb") as f:
        pickle.dump(captured, f)
    print(f"{task}: tested {len(picks)} stored episodes in {time.time() - t0:.0f} s", flush=True)


def summary(out: Path) -> dict:
    """Median relative return change under the float64 cast, and the first step past one spread unit."""
    result = {}
    for rows_file in sorted(out.glob("*_rows.json")):
        rows = json.loads(rows_file.read_text())
        curves = json.loads(rows_file.with_name(rows_file.name.replace("_rows", "_curves")).read_text())
        task = rows[0]["task"]
        first = {}
        for r in rows:
            first.setdefault(r["trigger_step"], r)
        policy = [r for r in first.values() if r["action_dtype"] == "float32"]
        rel = [r["f64_abs_err"] / max(abs(r["stored_return"]), 1e-9) for r in policy]
        steps = []
        for c in curves:
            if c["fresh"] < 0 and len(c["dev"]):
                d = np.array(c["dev"])
                steps.append(float(np.argmax(d > 1.0) + 1) if (d > 1.0).any() else float("nan"))
        steps = np.array(steps)
        result[task] = dict(stored_episodes_tested=len(first), policy_episodes=len(policy),
                            median_relative_return_change=float(np.median(rel)) if rel else None,
                            share_past_one_spread_unit=float(np.isfinite(steps).mean()) if len(steps) else None,
                            median_first_step_past_one_spread_unit=float(np.nanmedian(steps)) if np.isfinite(steps).any() else None)
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    for task in args.tasks:
        run_task(task, args.seed, args.out)
    if args.summary:
        print(json.dumps(summary(args.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
