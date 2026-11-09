"""Shared paths, style, loaders and statistics for the paper figures and tables.

Every number comes from the aggregated result files of the study (results/aggregated), the stored replay-check
episodes, the exploratory multiple-comparison table and the precision check (scripts/precision_check.py). Nothing
here changes a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parents[3]
AGG = ROOT / "results" / "aggregated"
REPLAY_STORE = ROOT / "results" / "replay_check"
PRECISION = ROOT / "results" / "precision_check"
MULTI = AGG / "multiple_comparisons.csv"  # written by derived.exact_family()
FIG_OUT = ROOT / "paper_assets" / "figures" / "main"
TAB_OUT = ROOT / "paper_assets" / "tables" / "main"
CACHE = ROOT / "results" / "paper_cache"

WIDTH = 6.5  # TMLR text width, inches

# ---------------------------------------------------------------- names and styles

TASKS = ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"]
TASKS3 = ["HalfCheetah-v4", "walker-walk", "finger-turn_hard"]
TASK_LABEL = {"HalfCheetah-v4": "HalfCheetah", "Hopper-v4": "Hopper", "walker-walk": "walker-walk",
              "finger-turn_hard": "finger-turn_hard"}
TASK_TEX = {k: v.replace("_", r"\_") for k, v in TASK_LABEL.items()}
ARM_LABEL = {"BASE": "BASE", "IER": "IER", "IER_REL": "IER-REL", "IER3": "IER3", "IER3_REL": "IER3-REL",
             "DUP": "DUP", "IER3_RANDOM": "IER3-RANDOM"}
LEARNER_LABEL = {"sac": "SAC", "crossq": "CrossQ", "both": "both", "all": "all"}
CONDITIONS = ["default", "noise5", "noise10", "dyn20"]

OI = {"black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7"}
# Same colors, markers and line styles per arm as the project's iers/plotting/style.py.
ARM_STYLE = {
    "BASE": {"color": OI["black"], "marker": "o", "linestyle": "-"},
    "IER": {"color": OI["blue"], "marker": "^", "linestyle": "-"},
    "IER3": {"color": OI["vermillion"], "marker": "s", "linestyle": "-."},
    "IER_REL": {"color": OI["sky_blue"], "marker": "v", "linestyle": "--"},
    "IER3_REL": {"color": OI["orange"], "marker": "D", "linestyle": "--"},
    "DUP": {"color": OI["bluish_green"], "marker": "P", "linestyle": ":"},
}
TASK_STYLE = {
    "HalfCheetah-v4": {"color": OI["orange"], "marker": "o"},
    "Hopper-v4": {"color": OI["sky_blue"], "marker": "^"},
    "walker-walk": {"color": OI["bluish_green"], "marker": "s"},
    "finger-turn_hard": {"color": OI["reddish_purple"], "marker": "D"},
}
# Conditions (fig_stress): more noise is a darker blue, the dynamics change stands apart.
COND_STYLE = {"default": OI["black"], "noise5": OI["sky_blue"], "noise10": OI["blue"], "dyn20": OI["vermillion"]}
GRAY = "#9a9a9a"
GRID = "#e3e3e3"
LW = 1.2
BAND_ALPHA = 0.2

# ---------------------------------------------------------------- fonts and rcParams

LM_DIRS = [ROOT / "paper_assets" / "fonts", Path("/usr/share/texmf/fonts/opentype/public/lm"),
           Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm")]


def apply_style() -> None:
    # Latin Modern Roman in its 8 pt optical design (what lmodern uses for 8 pt text), named explicitly so the choice
    # does not depend on which optical sizes the font cache lists first.
    for name in ("lmroman8-regular", "lmroman8-italic", "lmroman8-bold"):
        for folder in LM_DIRS:
            path = folder / f"{name}.otf"
            if path.exists():
                fm.fontManager.addfont(str(path))
                break
    family = "LM Roman 8"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [family], "mathtext.fontset": "cm",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.5,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": LW, "lines.markersize": 3.5,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.0, "legend.borderpad": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
        "axes.unicode_minus": True, "axes.formatter.use_mathtext": False,
    })


# ---------------------------------------------------------------- layout helpers (absolute inches)

MARGIN = {"left": 0.50, "right": 0.14, "bottom": 0.34, "top_legend": 0.40, "top": 0.20}
GAP = 0.36  # horizontal gap between panels in a strip, inches


def strip(height: float, ratios, legend: bool = True, gaps=None, left=None, right=None, bottom=None, top=None):
    """A full-width strip of panels with fixed margins in inches, so every figure shares them."""
    fig = plt.figure(figsize=(WIDTH, height))
    left = MARGIN["left"] if left is None else left
    right = MARGIN["right"] if right is None else right
    bottom = MARGIN["bottom"] if bottom is None else bottom
    top = (MARGIN["top_legend"] if legend else MARGIN["top"]) if top is None else top
    n = len(ratios)
    gaps = [GAP] * (n - 1) if gaps is None else list(gaps)
    usable = WIDTH - left - right - sum(gaps)
    widths = [usable * r / sum(ratios) for r in ratios]
    axes, x = [], left
    for i, w in enumerate(widths):
        ax = fig.add_axes([x / WIDTH, bottom / height, w / WIDTH, (height - bottom - top) / height])
        axes.append(ax)
        x += w + (gaps[i] if i < n - 1 else 0)
    return fig, axes


def panel_label(ax, letter: str, text: str = "", dx: float = 0.0) -> None:
    """'(a) Task' just above the top-left corner of the axes, left-aligned with the y tick labels."""
    label = f"({letter})" + (f" {text}" if text else "")
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-14 + dx, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=8)


def top_legend(fig, handles, labels, ncol=None, y=None, height=None, fontsize=7):
    height = fig.get_figheight() if height is None else height
    y = 1 - 0.05 / height if y is None else y
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5 + (MARGIN["left"] - MARGIN["right"]) / 2 / WIDTH, y),
                      ncol=ncol or len(handles), frameon=False, fontsize=fontsize, handlelength=1.8,
                      columnspacing=1.2, handletextpad=0.4)


def steps_formatter(x, _=None) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    return f"{x / 1e3:g}k"


def steps_axis(ax, budget: int) -> None:
    if budget >= 1_000_000:
        ax.set_xticks([0, 250_000, 500_000, 750_000, 1_000_000])
        ax.set_xticklabels(["0", "250k", "500k", "", "1M"])
    else:
        ax.set_xticks([0, 100_000, 200_000, 300_000])
        ax.set_xticklabels(["0", "100k", "200k", "300k"])
    ax.set_xlim(0, budget)


def num_formatter(ax, axis="y") -> None:
    """Thousands as 2k, 10k on return axes."""
    def fmt(v, _):
        if abs(v) >= 1000:
            return f"{v / 1000:g}k".replace("-", "−")
        return f"{v:g}".replace("-", "−")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(mticker.FuncFormatter(fmt))


def save(fig, name: str, png: bool = True) -> list[Path]:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    paths = [FIG_OUT / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=300)
    if png:
        fig.savefig(FIG_OUT / f"{name}.png", dpi=300)
        paths.append(FIG_OUT / f"{name}.png")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- data

_cache: dict = {}


def load(name: str) -> pd.DataFrame:
    if name not in _cache:
        _cache[name] = pd.read_csv(AGG / name)
    return _cache[name]


def load_json(name: str):
    return json.loads((AGG / name).read_text(encoding="utf-8"))


def normalizer(task: str, budget: int) -> float:
    for row in load_json("normalizers.json"):
        if row["task"] == task and int(row["budget"]) == int(budget):
            return float(row["value"])
    raise KeyError((task, budget))


EVAL_EVERY = 10_000


def seed_matrix(part: pd.DataFrame, budget: int | None = None) -> pd.DataFrame:
    """Seeds by evaluation point, aligned to the 10k boundary (Hopper evaluates at the first episode end after
    each boundary, as the project's plotting code notes). Only points every seed has are kept."""
    if budget is not None:
        part = part[part["step"] <= budget + EVAL_EVERY - 1]
    nominal = (part["step"] // EVAL_EVERY) * EVAL_EVERY
    table = part.assign(step=nominal).drop_duplicates(["seed", "step"])
    m = table.pivot(index="seed", columns="step", values="return").dropna(axis=1)
    if budget is not None:
        m = m.loc[:, m.columns <= budget]
    return m


def curve_part(task, condition, learner, arm, budget, experiment_budget=None):
    """Curves of a cell. experiment_budget selects the runs (e.g. A1 at 1M) when the cell is cut at `budget`."""
    c = load("curves.csv")
    src_budget = experiment_budget or budget
    return c[(c.task == task) & (c.condition == condition) & (c.learner == learner) & (c.arm == arm)
             & (c.budget == src_budget)]


BOOT_REPS = 10_000


def bootstrap_band(matrix, reps: int = BOOT_REPS, seed: int = 0):
    """Mean over seeds (rows) with a 95 percent percentile bootstrap band per column (10,000 resamples).

    Resampling seeds with replacement is done with multinomial weights, the same distribution as index draws."""
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=reps) / n  # (reps, n)
    means = weights @ m  # (reps, cols)
    low, high = np.percentile(means, [2.5, 97.5], axis=0)
    return m.mean(axis=0), low, high


def plot_band(ax, x, mean, low, high, color, linestyle="-", lw=LW, label=None, alpha=BAND_ALPHA, zorder=2):
    ax.fill_between(x, low, high, color=color, alpha=alpha, linewidth=0, zorder=zorder)
    return ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=lw, label=label, zorder=zorder + 0.5)[0]


def fmt_num(v, d=3, signed=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
    if s.startswith("-0.") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "$-$") if "$" not in s else s


def tex_minus(s: str) -> str:
    return s.replace("-", "$-$")
