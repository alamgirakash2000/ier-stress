"""Figure 5: selection bias. Retention ratio of every repeat against its start distance, all IER-type runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis import TASK_ORDER
from .style import DOUBLE, OKABE_ITO, TASK_LABEL, TASK_STYLE, cli, panel_label, read_csv, save

NAME = "fig5_selection"
Y_RANGE = (-1.0, 2.0)
BINS = 8


def select(agg: Path):
    return read_csv(agg, "repeats.csv")


def binned(x: np.ndarray, y: np.ndarray, bins: int = BINS):
    """Median and interquartile range of y in equal-count bins of x."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    centers, med, q1, q3 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (x >= lo) & (x <= hi)
        if inside.sum() < 5:
            continue
        centers.append(np.median(x[inside]))
        med.append(np.median(y[inside]))
        q1.append(np.quantile(y[inside], 0.25))
        q3.append(np.quantile(y[inside], 0.75))
    return map(np.asarray, (centers, med, q1, q3))


def make(agg: Path, out: Path):
    data = select(agg)
    if data is None:
        return None
    tasks = [t for t in TASK_ORDER if t in set(data.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(DOUBLE, 1.9), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        part = data[data.task == task]
        x, y = part.start_distance.to_numpy(float), part.ratio.to_numpy(float)
        style = TASK_STYLE[task]
        ax.scatter(x, np.clip(y, *Y_RANGE), s=2, color=style["color"], alpha=0.15, linewidths=0, rasterized=True)
        centers, med, q1, q3 = binned(x, y)
        ax.plot(centers, med, color=OKABE_ITO["black"], marker=style["marker"], linewidth=1.0)
        ax.fill_between(centers, q1, q3, color=OKABE_ITO["black"], alpha=0.15, linewidth=0)
        ax.axhline(1.0, color=OKABE_ITO["black"], linewidth=0.6, linestyle=":")
        ax.set_ylim(Y_RANGE[0] - 0.05, Y_RANGE[1] + 0.05)
        ax.set_xlabel("start distance")
        panel_label(ax, f"{TASK_LABEL[task]}, {len(part)} repeats")
    axes[0][0].set_ylabel("retention ratio")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    data = select(agg)
    clipped = "" if data is None else (
        f" Of {len(data)} repeats, {int(((data.ratio < Y_RANGE[0]) | (data.ratio > Y_RANGE[1])).sum())} fall outside "
        f"the shown range and are drawn at its edge.")
    return ("Return of each repeated episode divided by the return of the stored episode it replays, against the "
            "distance between their start states, over all IER-type runs. The line shows the median and the band "
            f"the interquartile range in {BINS} bins with equal counts. The dotted line marks a repeat that keeps "
            f"the full stored return.{clipped}")


if __name__ == "__main__":
    cli(make, caption)
