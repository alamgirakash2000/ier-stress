"""Metrics of Sections 5.5 and 5.6, computed from the files a run writes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import load_spreads

R_RET_WINDOW = 50_000
FINAL_EVALS = 5


def eval_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Steps and mean return of each evaluation, in order."""
    table = pd.read_csv(Path(run_dir) / "eval.csv")
    grouped = table.groupby("total_steps", sort=True)["return"].mean()
    return grouped.index.to_numpy(dtype=np.float64), grouped.to_numpy(dtype=np.float64)


def final_return(steps: np.ndarray, returns: np.ndarray) -> float:
    """Mean of the last 5 evaluations."""
    return float(np.mean(returns[-FINAL_EVALS:]))


def auc(steps: np.ndarray, returns: np.ndarray, normalizer: float) -> float:
    """Area under the evaluation curve, as a mean over the step range, divided by the task normalizer.

    An AUC of 1 means the run averaged the BASE SAC final return over the whole curve.
    """
    if len(steps) < 2:
        return float("nan")
    area = np.trapezoid(returns, steps) / (steps[-1] - steps[0])
    return float(area / normalizer)


def obs_scale(task: str, spreads: dict | None = None) -> np.ndarray:
    spreads = load_spreads() if spreads is None else spreads
    return np.asarray(spreads[task]["obs_scale"], dtype=np.float64)


def start_distance(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Root mean square difference between start observations, per dimension in units of `scale`.

    Dimensions that never vary in the calibration rollouts have a scale of zero and are left out.
    """
    used = scale > 0
    difference = (np.asarray(a)[..., used] - np.asarray(b)[..., used]) / scale[used]
    return np.sqrt(np.mean(difference**2, axis=-1))


@dataclass
class RepetitionSummary:
    trigger_count: int
    repeat_count: int
    replay_steps: int
    total_steps: int
    repetition_share: float
    r_ret: float  # retention in the first 50k steps, the Q7 diagnostic
    r_ret_all: float
    repeats_in_window: int
    start_distance_mean: float


def repetition_summary(run_dir: Path, task: str, spreads: dict | None = None) -> RepetitionSummary:
    run_dir = Path(run_dir)
    train = pd.read_csv(run_dir / "train.csv")
    triggers = pd.read_csv(run_dir / "triggers.csv")
    repeats = pd.read_csv(run_dir / "repeats.csv")
    total = int(train["length"].sum())
    replay_steps = int(train["replay_steps"].sum())
    ratio = repeats["return"] / repeats["stored_return"] if len(repeats) else pd.Series(dtype=float)
    window = ratio[repeats["total_steps"] <= R_RET_WINDOW] if len(repeats) else ratio
    distance = float("nan")
    if len(repeats):
        with np.load(run_dir / "episodes.npz") as data:
            starts, stored = data["repeat_start_obs"], data["repeat_stored_start_obs"]
        distance = float(np.mean(start_distance(starts, stored, obs_scale(task, spreads))))
    return RepetitionSummary(
        trigger_count=len(triggers),
        repeat_count=len(repeats),
        replay_steps=replay_steps,
        total_steps=total,
        repetition_share=replay_steps / total if total else float("nan"),
        r_ret=float(window.mean()) if len(window) else float("nan"),
        r_ret_all=float(ratio.mean()) if len(ratio) else float("nan"),
        repeats_in_window=len(window),
        start_distance_mean=distance,
    )


def read_record(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "record.json").read_text(encoding="utf-8"))
