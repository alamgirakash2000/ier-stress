"""Figure 8a, appendix: performance profiles of normalized final return, BASE against IER."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import ARM_STYLE, DOUBLE, LEARNER_STYLE, budget_label, cli, line, panel_label, read_json, save

NAME = "fig8a_profiles"


def make(agg: Path, out: Path):
    data = read_json(agg, "profiles.json")
    if not data:
        return None
    keys = sorted(data, key=lambda k: (k.split("_")[0] != "sac", k))
    fig, axes = plt.subplots(1, len(keys), figsize=(DOUBLE if len(keys) > 1 else DOUBLE / 2, 2.0), squeeze=False)
    for ax, key in zip(axes[0], keys):
        entry = data[key]
        taus = np.asarray(entry["taus"])
        for arm in entry["profiles"]:
            style = ARM_STYLE[arm]
            line(ax, taus, entry["profiles"][arm], style, label=arm)
            ax.fill_between(taus, entry["low"][arm], entry["high"][arm], color=style["color"], alpha=0.2, linewidth=0)
        learner, budget = key.split("_")
        panel_label(ax, f"{LEARNER_STYLE[learner]['label']}, {budget_label(int(budget[:-1]) * 1000)} steps")
        ax.set_xlabel("normalized final return threshold")
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("fraction of runs above")
    axes[0][0].legend(loc="lower left")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    return ("Performance profiles: the fraction of runs, pooled over tasks, whose normalized final return exceeds "
            "each threshold. Bands are 95 percent stratified bootstrap intervals.")


if __name__ == "__main__":
    cli(make, caption)
