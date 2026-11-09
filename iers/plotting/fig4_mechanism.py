"""Figure 4: mechanism. IER3 against DUP against BASE, SAC, per task and condition, 300k steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis import CONDITION_ORDER, SHORT, TASK_ORDER
from .style import ARM_STYLE, DOUBLE, TASK_LABEL, cli, errorbar, panel_label, read_csv, save

NAME = "fig4_mechanism"
ARMS = ["BASE", "IER3", "DUP"]


def select(agg: Path):
    cells = read_csv(agg, "cells.csv")
    if cells is None:
        return None
    chosen = cells[(cells.learner == "sac") & (cells.budget == SHORT) & cells.arm.isin(ARMS)
                   & cells.condition.isin(["default", "noise5"])]
    if not (chosen.arm == "DUP").any():
        return None
    return chosen


def make(agg: Path, out: Path):
    data = select(agg)
    if data is None:
        return None
    tasks = [t for t in TASK_ORDER if t in set(data[data.arm == "DUP"].task)]
    conditions = [c for c in CONDITION_ORDER if c in set(data.condition)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(DOUBLE, 1.9), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for i, arm in enumerate(ARMS):
            part = data[(data.task == task) & (data.arm == arm)].set_index("condition").reindex(conditions)
            x = np.arange(len(conditions)) + (i - 1) * 0.2
            ok = part["auc_mean"].notna().to_numpy()
            errorbar(ax, x[ok], part["auc_mean"][ok], part["auc_low"][ok], part["auc_high"][ok], ARM_STYLE[arm],
                     label=arm if task == tasks[0] else None)
        ax.set_xticks(np.arange(len(conditions)), conditions)
        ax.set_xlim(-0.6, len(conditions) - 0.4)
        panel_label(ax, TASK_LABEL[task])
    axes[0][0].set_ylabel("normalized AUC")
    axes[0][0].legend(loc="lower left")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    return ("Normalized AUC over 300k environment steps for SAC alone (BASE), SAC with IER and RN = 3, and the "
            "buffer duplication control (DUP), which adds RN copies of the stored episode to the replay buffer "
            "with no extra environment steps. Points show the mean over seeds and error bars 95 percent percentile "
            "bootstrap intervals of the mean.")


if __name__ == "__main__":
    cli(make, caption)
