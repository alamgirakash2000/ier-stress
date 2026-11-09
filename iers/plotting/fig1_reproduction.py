"""Figure 1: reproduction curves. SAC BASE against IER on the four tasks, `default`, 1M steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis import LONG, TASK_ORDER
from .style import (ARM_STYLE, DOUBLE, TASK_LABEL, bootstrap_band, cli, line, panel_label, read_csv, save,
                    seed_matrix, steps_axis)

NAME = "fig1_reproduction"
ARMS = ["BASE", "IER"]


def select(agg: Path):
    curves = read_csv(agg, "curves.csv")
    if curves is None:
        return None
    chosen = curves[(curves.learner == "sac") & (curves.condition == "default") & (curves.budget == LONG)
                    & curves.arm.isin(ARMS)]
    return chosen if len(chosen) else None


def make(agg: Path, out: Path):
    data = select(agg)
    if data is None:
        return None
    tasks = [t for t in TASK_ORDER if t in set(data.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(DOUBLE, 1.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for arm in ARMS:
            part = data[(data.task == task) & (data.arm == arm)]
            if not len(part):
                continue
            matrix = seed_matrix(part)
            mean, low, high = bootstrap_band(matrix.to_numpy())
            style = ARM_STYLE[arm]
            line(ax, matrix.columns, mean, style, label=arm if task == tasks[0] else None)
            ax.fill_between(matrix.columns, low, high, color=style["color"], alpha=0.2, linewidth=0)
        panel_label(ax, TASK_LABEL[task])
        steps_axis(ax)
        ax.set_xlabel("environment steps")
    axes[0][0].set_ylabel("evaluation return")
    axes[0][0].legend(loc="lower right")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    data = select(agg)
    seeds = "" if data is None else f" with {data.groupby(['task', 'arm']).seed.nunique().min()} or more seeds per arm"
    return ("Evaluation return of SAC alone (BASE) and SAC with IER under the default condition over 1M environment "
            f"steps{seeds}. Lines show the mean over seeds and bands a 95 percent percentile bootstrap interval of "
            "the mean. Yamani et al. report gains for IER over SAC on MuJoCo and DeepMind Control Suite tasks.")


if __name__ == "__main__":
    cli(make, caption)
