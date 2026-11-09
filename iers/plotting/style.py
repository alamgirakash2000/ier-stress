"""Shared figure style: matplotlib only, Okabe-Ito colors, a fixed color and marker per arm,
8 pt text, 3.4 in or 7.0 in wide, no titles, PDF and 300 dpi PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import ROOT  # noqa: E402

AGGREGATED = ROOT / "results" / "aggregated"
FIGURES = ROOT / "paper_assets" / "figures"
SINGLE, DOUBLE = 3.4, 7.0

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}
# Identity never rests on color alone: every arm also has its own marker and line style.
ARM_STYLE = {
    "BASE": {"color": OKABE_ITO["black"], "marker": "o", "linestyle": "-"},
    "IER": {"color": OKABE_ITO["blue"], "marker": "^", "linestyle": "-"},
    "IER3": {"color": OKABE_ITO["vermillion"], "marker": "s", "linestyle": "-."},
    "IER_REL": {"color": OKABE_ITO["sky_blue"], "marker": "v", "linestyle": "--"},
    "IER3_REL": {"color": OKABE_ITO["orange"], "marker": "D", "linestyle": "--"},
    "DUP": {"color": OKABE_ITO["bluish_green"], "marker": "P", "linestyle": ":"},
}
LEARNER_STYLE = {
    "sac": {"color": OKABE_ITO["blue"], "marker": "o", "linestyle": "-", "label": "SAC"},
    "crossq": {"color": OKABE_ITO["reddish_purple"], "marker": "s", "linestyle": "--", "label": "CrossQ"},
}
TASK_STYLE = {
    "HalfCheetah-v4": {"color": OKABE_ITO["orange"], "marker": "o"},
    "Hopper-v4": {"color": OKABE_ITO["sky_blue"], "marker": "^"},
    "walker-walk": {"color": OKABE_ITO["bluish_green"], "marker": "s"},
    "finger-turn_hard": {"color": OKABE_ITO["reddish_purple"], "marker": "D"},
}
TASK_LABEL = {"HalfCheetah-v4": "HalfCheetah", "Hopper-v4": "Hopper", "walker-walk": "walker-walk",
              "finger-turn_hard": "finger-turn_hard"}
CONDITION_LABEL = {"default": "default", "noise5": "noise5", "noise10": "noise10", "dyn20": "dyn20"}


def apply() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "legend.frameon": False, "axes.titlesize": 8,
    })


def save(fig, out: Path, name: str) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def panel_label(ax, text: str) -> None:
    """A short label just above the panel, clear of the data. Figures carry no titles."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7)


def budget_label(budget: int) -> str:
    return f"{budget / 1e6:g}M" if budget >= 1_000_000 else f"{budget // 1000}k"


def line(ax, x, y, style: dict, label: str | None = None, marks: int = 6) -> None:
    """A line with a few markers on it, so the legend shows both."""
    ax.plot(x, y, color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
            markevery=max(1, len(x) // marks), label=label)


def steps_axis(ax) -> None:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k"))


EVAL_EVERY = 10_000


def seed_matrix(part: pd.DataFrame) -> pd.DataFrame:
    """Seeds by evaluation point. Points are aligned to their 10k boundary, because upstream evaluates at the first
    episode end after each boundary, so on Hopper the exact step differs between seeds.
    Only points every seed has are kept."""
    nominal = (part["step"] // EVAL_EVERY) * EVAL_EVERY
    table = part.assign(step=nominal).drop_duplicates(["seed", "step"])
    return table.pivot(index="seed", columns="step", values="return").dropna(axis=1)


def bootstrap_band(matrix: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean over seeds (rows) and a percentile bootstrap 95 percent band, per column."""
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    if len(matrix) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(matrix), size=(reps, len(matrix)))
    means = matrix[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5], axis=0)
    return mean, low, high


def read_csv(agg: Path, name: str) -> pd.DataFrame | None:
    path = Path(agg) / name
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return table if len(table) else None


def read_json(agg: Path, name: str):
    path = Path(agg) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def errorbar(ax, x, mean, low, high, style: dict, label: str | None = None, **kwargs) -> None:
    yerr = np.vstack([np.asarray(mean) - np.asarray(low), np.asarray(high) - np.asarray(mean)])
    ax.errorbar(x, mean, yerr=yerr, color=style["color"], marker=style["marker"], linestyle="none",
                capsize=2, elinewidth=0.8, label=label, **kwargs)


def cli(make, caption) -> None:
    """Command line for one figure script: --agg reads, --out writes."""
    from ..envcheck import require_ierstress

    require_ierstress()
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", type=Path, default=AGGREGATED)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    apply()
    written = make(args.agg, args.out)
    if written:
        print("written:", ", ".join(p.name for p in written))
        print(caption(args.agg))
    else:
        print("no data for this figure yet")
