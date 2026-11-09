"""Figure 8c, appendix: mean GPU and CPU use during each logged grid."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import OKABE_ITO, SINGLE, cli, read_csv, save

NAME = "fig8c_resources"


def summary(agg: Path):
    data = read_csv(agg, "resources.csv")
    if data is None:
        return None
    return data.groupby("grid", sort=True).agg(gpu=("gpu_util_pct", "mean"), cpu=("cpu_util_pct", "mean"),
                                               samples=("time", "size"))


def make(agg: Path, out: Path):
    table = summary(agg)
    if table is None:
        return None
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(SINGLE, 0.35 * len(table) + 0.8))
    ax.barh(y - 0.18, table.gpu, height=0.34, color=OKABE_ITO["blue"], label="GPU")
    ax.barh(y + 0.18, table.cpu, height=0.34, color="white", edgecolor=OKABE_ITO["vermillion"], hatch="////",
            label="CPU")
    ax.set_yticks(y, table.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("mean utilization, percent")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    table = summary(agg)
    count = "" if table is None else f" Each grid has between {int(table.samples.min())} and {int(table.samples.max())} samples."
    return ("Mean GPU utilization (solid) and CPU utilization (hatched) during each grid, sampled every 60 seconds "
            f"on the machine that ran each grid.{count}")


if __name__ == "__main__":
    cli(make, caption)
