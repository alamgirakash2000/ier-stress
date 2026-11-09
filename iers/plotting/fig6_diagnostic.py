"""Figure 6: early diagnostic. R_ret in the first 50k steps against the final IER gain, one point per cell."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..analysis import REPEAT_ARMS
from .style import ARM_STYLE, OKABE_ITO, SINGLE, cli, read_csv, save

NAME = "fig6_diagnostic"


def select(agg: Path):
    gains = read_csv(agg, "gains.csv")
    if gains is None:
        return None, None
    chosen = gains[gains.arm.isin(REPEAT_ARMS) & gains.r_ret.notna()]
    contrasts = read_csv(agg, "contrasts.csv")
    p8 = None if contrasts is None else contrasts[contrasts.id == "P8"]
    return (chosen if len(chosen) else None), (p8.iloc[0] if p8 is not None and len(p8) else None)


def make(agg: Path, out: Path):
    data, p8 = select(agg)
    if data is None:
        return None
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for arm in REPEAT_ARMS:
        part = data[data.arm == arm]
        for learner, fill in (("sac", True), ("crossq", False)):
            sub = part[part.learner == learner]
            if not len(sub):
                continue
            style = ARM_STYLE[arm]
            ax.errorbar(sub.r_ret, sub.estimate, yerr=[sub.estimate - sub.low, sub.high - sub.estimate],
                        color=style["color"], marker=style["marker"], linestyle="none", capsize=1.5, elinewidth=0.6,
                        markerfacecolor=style["color"] if fill else "white",
                        label=f"{arm}, {'SAC' if learner == 'sac' else 'CrossQ'}")
    ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.6, linestyle=":")
    if p8 is not None:
        ax.text(0.02, 0.98, f"Spearman {p8.estimate:.2f} [{p8.low:.2f}, {p8.high:.2f}]", transform=ax.transAxes,
                ha="left", va="top", fontsize=7)
    ax.set_xlabel("R_ret, first 50k steps")
    ax.set_ylabel("gain in normalized AUC")
    ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    data, p8 = select(agg)
    cells = "" if data is None else f" There are {len(data)} cells."
    return ("Mean retention ratio of repeats in the first 50k environment steps (R_ret) against the gain of the "
            "IER-type arm over BASE in normalized AUC, one point per task, condition, learner, arm and budget. "
            "Filled markers are SAC and open markers CrossQ. Error bars are 95 percent percentile bootstrap "
            "intervals of the mean per-seed gain. The text gives the Spearman correlation across cells with a "
            f"95 percent bootstrap interval over seeds.{cells}")


if __name__ == "__main__":
    cli(make, caption)
