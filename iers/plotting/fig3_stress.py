"""Figure 3: IER3 gain over BASE against condition, per task, for SAC and CrossQ, 300k steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis import CONDITION_ORDER, SHORT, TASK_ORDER
from .style import DOUBLE, LEARNER_STYLE, OKABE_ITO, TASK_LABEL, cli, errorbar, panel_label, read_csv, save

NAME = "fig3_stress"


def select(agg: Path):
    gains = read_csv(agg, "gains.csv")
    if gains is None:
        return None
    chosen = gains[(gains.arm == "IER3") & (gains.budget == SHORT)]
    return chosen if len(chosen) else None


def make(agg: Path, out: Path):
    data = select(agg)
    if data is None:
        return None
    tasks = [t for t in TASK_ORDER if t in set(data.task)]
    conditions = [c for c in CONDITION_ORDER if c in set(data.condition)]
    learners = [name for name in LEARNER_STYLE if name in set(data.learner)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(DOUBLE, 1.9), squeeze=False, sharey=True)
    for ax, task in zip(axes[0], tasks):
        for i, learner in enumerate(learners):
            part = data[(data.task == task) & (data.learner == learner)].set_index("condition").reindex(conditions)
            x = np.arange(len(conditions)) + (i - (len(learners) - 1) / 2) * 0.18
            ok = part["n"].notna().to_numpy()
            errorbar(ax, x[ok], part["estimate"][ok], part["low"][ok], part["high"][ok], LEARNER_STYLE[learner],
                     label=LEARNER_STYLE[learner]["label"] if task == tasks[0] else None)
        ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.6, linestyle=":")
        ax.set_xticks(np.arange(len(conditions)), conditions)
        panel_label(ax, TASK_LABEL[task])
    axes[0][0].set_ylabel("IER3 gain in normalized AUC")
    axes[0][0].legend(loc="lower left")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    data = select(agg)
    seeds = "" if data is None else f" Each point uses {int(data.n.min())} to {int(data.n.max())} paired seeds."
    return ("Gain of IER with RN = 3 over the same learner without it, in normalized AUC over 300k environment "
            "steps, per task and condition. Error bars are 95 percent percentile bootstrap intervals of the mean "
            f"per-seed difference.{seeds} The dotted line marks no gain.")


if __name__ == "__main__":
    cli(make, caption)
