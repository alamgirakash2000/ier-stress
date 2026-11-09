"""Figure 2: IQM of normalized AUC and the probability of improvement, for A1 (SAC) and A2 (CrossQ)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .style import ARM_STYLE, DOUBLE, LEARNER_STYLE, OKABE_ITO, budget_label, cli, panel_label, read_json, save

NAME = "fig2_rliable"


def make(agg: Path, out: Path):
    data = read_json(agg, "rliable.json")
    if not data:
        return None
    keys = sorted(data, key=lambda k: (k.split("_")[0] != "sac", k))
    fig, axes = plt.subplots(len(keys), 2, figsize=(DOUBLE, 0.75 * len(keys) + 0.5), squeeze=False)
    for row, key in enumerate(keys):
        entry = data[key]
        learner = LEARNER_STYLE[entry["learner"]]["label"]
        ax = axes[row][0]
        for y, arm in enumerate(["BASE", "IER"]):
            v = entry["iqm"][arm]
            ax.errorbar([v["estimate"]], [y], xerr=[[v["estimate"] - v["low"]], [v["high"] - v["estimate"]]],
                        color=ARM_STYLE[arm]["color"], marker=ARM_STYLE[arm]["marker"], capsize=2, linestyle="none")
        ax.set_yticks([0, 1], ["BASE", "IER"])
        ax.set_ylim(-0.6, 1.6)
        panel_label(ax, f"{learner}, {budget_label(entry['budget'])} steps, {len(entry['seeds'])} seeds")
        ax.set_xlabel("IQM of normalized AUC")
        ax = axes[row][1]
        p = entry["probability_of_improvement"]
        ax.errorbar([p["estimate"]], [0], xerr=[[p["estimate"] - p["low"]], [p["high"] - p["estimate"]]],
                    color=OKABE_ITO["blue"], marker="^", capsize=2, linestyle="none")
        ax.axvline(0.5, color=OKABE_ITO["black"], linewidth=0.6, linestyle=":")
        ax.set_xlim(-0.03, 1.03)
        ax.set_yticks([0], ["P(IER > BASE)"])
        ax.set_xlabel("probability of improvement")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    return ("Left: interquartile mean of normalized AUC across tasks, with 95 percent stratified bootstrap intervals "
            "(10,000 resamples). Right: probability that a run with IER has a higher normalized AUC than a run "
            "without it on a randomly chosen task, with its 95 percent interval. The dotted line marks 0.5.")


if __name__ == "__main__":
    cli(make, caption)
