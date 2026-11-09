"""One independent reference run for Gate G1: stable-baselines3 SAC or sb3-contrib CrossQ.

Writes eval.csv in the same format as our runs, so one analysis reads both. Resumes from its own
checkpoint, written every 50k steps. A resumed run restarts its current episode, which is recorded in
resume_count. These runs exist only for the validity gate.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import CrossQ  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN, Actor, SACPolicy  # noqa: E402
from torch import nn  # noqa: E402

from iers.codehash import code_hash, write_manifest  # noqa: E402
from iers.guard import (  # noqa: E402
    EXIT_RETRY,
    GuardLimits,
    ResourceGuard,
    cap_gpu_memory,
    hold_run_lock,
    keep_awake,
    lower_priority,
)
from iers.logging import get_logger  # noqa: E402
from iers.runner import atomic_write_csv, atomic_write_text, package_versions  # noqa: E402

CHECKPOINT_EVERY = 50_000
EVAL_EVERY = 10_000
EVAL_EPISODES = 10


class UpstreamLogStdActor(Actor):
    """SB3's SAC actor with one change: the log std of the upstream IER actor.

    Upstream squashes log std with tanh into [-20, 2], so a fresh policy starts near log std -9. SB3 clips it to
    the same range, so a fresh policy starts near log std 0. Only for the SAC follow-up check of gate_g1.py.
    """

    def get_action_dist_params(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        log_std = torch.tanh(self.log_std(latent_pi))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean_actions, log_std, {}


class UpstreamLogStdPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return UpstreamLogStdActor(**actor_kwargs).to(self.device)


def build_model(algo: str, env, seed: int, budget: int, device: str, actor: str = "sb3"):
    shared = dict(buffer_size=min(budget, 1_000_000), learning_starts=1000, batch_size=256, gamma=0.99, train_freq=1,
                  gradient_steps=1, ent_coef="auto", target_entropy="auto", seed=seed, device=device, verbose=0)
    if algo == "sac":
        # Matched to Table 5 of the IER paper.
        policy = UpstreamLogStdPolicy if actor == "upstream" else "MlpPolicy"
        return SAC(policy, env, learning_rate=3e-4, tau=0.005,
                   policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.ReLU), **shared)
    if algo == "crossq":
        # [1024, 1024] critics, the sb3-contrib default and the width of our CrossQ.
        return CrossQ("MlpPolicy", env, learning_rate=1e-3, policy_delay=3,
                      policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[1024, 1024]), activation_fn=nn.ReLU,
                                         batch_norm_momentum=0.01, batch_norm_eps=1e-3,
                                         renorm_warmup_steps=100_000), **shared)
    raise ValueError(f"Unknown algo: {algo}")


class Tracker(BaseCallback):
    def __init__(self, eval_env, out: Path, ckpt: Path, state: dict, guard: ResourceGuard, logger, args):
        super().__init__()
        self.eval_env, self.out, self.ckpt, self.state = eval_env, out, ckpt, state
        self.guard, self.run_log = guard, logger
        self.eval_every, self.checkpoint_every = args.eval_every, args.checkpoint_every
        self.stop_after = args.stop_after
        self.stopped = False

    def _save(self) -> None:
        self.ckpt.mkdir(parents=True, exist_ok=True)
        tmp_model = self.ckpt / "model_tmp"
        tmp_replay = self.ckpt / "replay_tmp"
        self.model.save(tmp_model)
        self.model.save_replay_buffer(tmp_replay)
        os.replace(self.ckpt / "model_tmp.zip", self.ckpt / "model.zip")
        os.replace(self.ckpt / "replay_tmp.pkl", self.ckpt / "replay.pkl")
        self.state["wall_before"] = self.state["wall_before"] + time.time() - self.state["t0"]
        self.state["t0"] = time.time()
        self.state["paused_s"] = self.state["paused_before"] + self.guard.paused_s
        atomic_write_text(self.ckpt / "state.json", json.dumps({k: v for k, v in self.state.items() if k != "t0"}))
        self.run_log.info("checkpoint at step %d", self.num_timesteps)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t % self.eval_every == 0 and t > self.state["last_eval"]:
            returns, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True,
                                         return_episode_rewards=True, warn=False)
            for episode, value in enumerate(returns):
                self.state["eval_log"].append([t, episode + 1, float(value)])
            self.state["last_eval"] = t
            self.run_log.info("step %d evaluation mean %.1f", t, sum(returns) / len(returns))
        if t % self.checkpoint_every == 0 and t < self.state["budget"]:
            self._save()
        if self.guard.check() or (self.stop_after and t >= self.stop_after):
            self.run_log.info("stopping at step %d: %s", t, self.guard.exit_reason or "--stop-after")
            self._save()
            self.stopped = True
            return False
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["sac", "crossq"])
    parser.add_argument("--task", default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=300_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment", default="g1_reference")
    parser.add_argument("--results-root", default="results/raw")
    parser.add_argument("--ckpt-root", default="results/checkpoints")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--stop-after", type=int, default=0, help="simulates a clean stop, for tests")
    parser.add_argument("--actor", default="sb3", choices=["sb3", "upstream"],
                        help="SAC only: 'upstream' uses the upstream IER log std, for the Gate G1 debug run")
    args = parser.parse_args()
    if args.actor == "upstream" and args.algo != "sac":
        parser.error("--actor upstream applies to SAC only")

    torch.set_num_threads(1)
    lower_priority()
    keep_awake()
    variant ="" if args.actor == "sb3" else "_upstream_logstd"
    run_id = f"{args.task}__default__sb3_{args.algo}{variant}__BASE__seed{args.seed}"
    out = Path(args.results_root) / args.experiment / run_id
    ckpt = Path(args.ckpt_root) / args.experiment / run_id
    logger = get_logger(f"{args.experiment}_{args.algo}_{args.seed}", console=True)
    hold_run_lock(Path(args.ckpt_root) / args.experiment / f"{run_id}.lock", log=logger.info)
    if (out / "record.json").exists():
        logger.info("%s is already finished", run_id)
        return 0
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        cap_gpu_memory(1800)

    env = Monitor(gym.make(args.task))
    eval_env = Monitor(gym.make(args.task))
    eval_env.reset(seed=args.seed + 10_000)
    state = {"budget": args.budget, "eval_log": [], "last_eval": 0, "wall_before": 0.0, "resume_count": 0,
             "paused_s": 0.0, "paused_before": 0.0}
    if (ckpt / "state.json").exists():
        state = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        state["resume_count"] += 1
        state["paused_before"] = state["paused_s"]
        algo_class = {"sac": SAC, "crossq": CrossQ}[args.algo]
        model = algo_class.load(ckpt / "model.zip", env=env, device=args.device)
        model.load_replay_buffer(ckpt / "replay.pkl")
        logger.info("resumed %s at step %d", run_id, model.num_timesteps)
    else:
        model = build_model(args.algo, env, args.seed, args.budget, args.device, args.actor)
    state["t0"] = time.time()
    start_hash = write_manifest()  # taken at start, so later edits cannot change what the record says ran
    state["code_hashes"] = state.get("code_hashes", []) + [start_hash]
    logger.info("%s starts, device of networks %s", run_id, next(model.policy.parameters()).device)

    guard = ResourceGuard(GuardLimits(), log=logger.info, watch_gpu=args.device == "cuda")
    tracker = Tracker(eval_env, out, ckpt, state, guard, logger, args)
    remaining = args.budget - model.num_timesteps
    model.learn(total_timesteps=remaining, callback=tracker, reset_num_timesteps=False)
    if tracker.stopped:
        return EXIT_RETRY

    wall = state["wall_before"] + time.time() - state["t0"]
    atomic_write_csv(out / "eval.csv", ["total_steps", "episode", "return"], state["eval_log"])
    record = {
        "run_id": run_id,
        "status": "finished",
        "reason": "",
        "algo": args.algo,
        "actor": args.actor,
        "task": args.task,
        "seed": args.seed,
        "budget": args.budget,
        "steps_reached": int(model.num_timesteps),
        "device_networks": str(next(model.policy.parameters()).device),
        "wall_time_s": wall,
        "guard_paused_s": state["paused_before"] + guard.paused_s,
        "resume_count": state["resume_count"],
        "code_hash": start_hash,
        "code_hashes_by_session": state["code_hashes"],
        "code_hash_at_finish": code_hash(),
        "package_versions": package_versions(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_text(out / "record.json", json.dumps(record, indent=1))
    shutil.rmtree(ckpt, ignore_errors=True)
    logger.info("%s finished, wall %.1f s", run_id, wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
