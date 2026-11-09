"""Runs a grid of (task, condition, learner, arm, seed) cells in groups, with resume.

    python scripts/run_experiment.py --experiment smoke --tasks HalfCheetah-v4 --arms BASE IER \
        --seeds 0 1 --budget 5000

Finished runs are skipped unless --force is given. Interrupted groups resume from their
checkpoint. A group that keeps crashing is recorded with its reason in each run's record.json.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

import yaml  # noqa: E402

from iers.grid import run_grid  # noqa: E402
from iers.specs import RESULTS_CKPT, RESULTS_RAW, TrainConfig, load_tasks, make_spec  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML file with the same keys as these arguments")
    parser.add_argument("--experiment")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--conditions", nargs="+", default=["default"])
    parser.add_argument("--learners", nargs="+", default=["sac"])
    parser.add_argument("--arms", nargs="+", default=["BASE"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--dup-rn", type=int, help="RN of DUP, to match the arm it is compared with")
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--device")
    parser.add_argument("--replay-device")
    parser.add_argument("--exploration", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--results-root", default=str(RESULTS_RAW))
    parser.add_argument("--ckpt-root", default=str(RESULTS_CKPT))
    parser.add_argument("--monitor-interval", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            for key, value in yaml.safe_load(handle).items():
                setattr(args, key.replace("-", "_"), value)
    return args


def build_specs(args) -> list:
    tasks = load_tasks()
    return [
        make_spec(args.experiment, task, condition, learner, arm, seed, args.budget, tasks[task]["rn"],
                  rn=args.dup_rn if arm == "DUP" else None)
        for task in args.tasks
        for condition in args.conditions
        for learner in args.learners
        for arm in args.arms
        for seed in args.seeds
    ]


def main() -> int:
    args = parse_args()
    train = TrainConfig(exploration_steps=args.exploration, eval_every=args.eval_every,
                        eval_episodes=args.eval_episodes, checkpoint_every=args.checkpoint_every)
    return run_grid(args.experiment, build_specs(args), train, group_size=args.group_size, processes=args.processes,
                    device=args.device, replay_device=args.replay_device, results_root=Path(args.results_root),
                    ckpt_root=Path(args.ckpt_root), force=args.force, retries=args.retries,
                    monitor_interval=args.monitor_interval)


if __name__ == "__main__":
    raise SystemExit(main())
