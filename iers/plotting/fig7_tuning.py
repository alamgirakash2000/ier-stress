"""Figure 7: tuning. IER with the per-task RN against IER3, with BASE for reference, SAC, `default`, 1M steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis import LONG, TASK_ORDER
from .style import ARM_STYLE, DOUBLE, TASK_LABEL, cli, errorbar, read_csv, save

NAME = "fig7_tuning"
ARMS = ["BASE", "IER", "IER3"]


def select(agg: Path):
    cells = read_csv(agg, "cells.csv")
    if cells is None:
        return None
    chosen = cells[(cells.learner == "sac") & (cells.condition == "default") & (cells.budget == LONG)
                   & cells.arm.isin(ARMS)]
    return chosen if (chosen.arm == "IER3").any() else None


def make(agg: Path, out: Path):
    data = select(agg)
    if data is None:
        return None
    tasks = [t for t in TASK_ORDER if t in set(data[data.arm == "IER3"].task)]
    fig, ax = plt.subplots(figsize=(DOUBLE / 2, 2.2))
    for i, arm in enumerate(ARMS):
        part = data[data.arm == arm].set_index("task").reindex(tasks)
        x = np.arange(len(tasks)) + (i - 1) * 0.2
        ok = part["auc_mean"].notna().to_numpy()
        errorbar(ax, x[ok], part["auc_mean"][ok], part["auc_low"][ok], part["auc_high"][ok], ARM_STYLE[arm], label=arm)
    ax.set_xticks(np.arange(len(tasks)), [TASK_LABEL[t] for t in tasks], rotation=15)
    ax.set_ylabel("normalized AUC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    return ("Normalized AUC over 1M environment steps for SAC alone (BASE), IER with the per-task RN of the paper "
            "(HalfCheetah 4, Hopper 2, walker-walk 5, finger-turn_hard 5), and IER with RN = 3 on every task. "
            "Points show the mean over seeds and error bars 95 percent percentile bootstrap intervals of the mean.")


if __name__ == "__main__":
    cli(make, caption)
