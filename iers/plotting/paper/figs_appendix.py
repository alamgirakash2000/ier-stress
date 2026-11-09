"""Appendix figures: every learning curve per experiment, and the rliable aggregates."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from .common import (ARM_LABEL, ARM_STYLE, LW, TASK_LABEL, TASKS, TASKS3, WIDTH, load_json, num_formatter,
                    panel_label, save, steps_axis, top_legend)
from .derived import cell_scores
from .figs_main import arm_curve, legend_line, legend_marker, ylabel

LONG, SHORT = 1_000_000, 300_000

# experiment: learner, budget, tasks, conditions, [(arm, source experiment or None for own runs)], conditions per arm
EXPERIMENTS = {
    "A1": dict(learner="sac", budget=LONG, tasks=TASKS, conds=["default"], arms=[("BASE", None), ("IER", None)]),
    "R1": dict(learner="sac", budget=LONG, tasks=TASKS, conds=["default"],
               arms=[("IER_REL", None), ("BASE", "A1"), ("IER", "A1")]),
    "D1": dict(learner="sac", budget=LONG, tasks=TASKS, conds=["default"],
               arms=[("IER3", None), ("BASE", "A1"), ("IER", "A1")]),
    "B1": dict(learner="sac", budget=SHORT, tasks=TASKS3, conds=["default", "noise5", "dyn20"],
               arms=[("BASE", None), ("IER3", None)]),
    "B3": dict(learner="sac", budget=SHORT, tasks=TASKS3, conds=["noise10"], arms=[("BASE", None), ("IER3", None)]),
    "C1": dict(learner="sac", budget=SHORT, tasks=TASKS3, conds=["default", "noise5"],
               arms=[("DUP", None), ("BASE", "B1"), ("IER3", "B1")]),
    "A2": dict(learner="crossq", budget=SHORT, tasks=TASKS, conds=["default"], arms=[("BASE", None), ("IER", None)]),
    "B2": dict(learner="crossq", budget=SHORT, tasks=TASKS3, conds=["default", "noise5"],
               arms=[("IER3", None), ("BASE", None)]),
}


def _source(exp, arm, cond):
    """Experiment that holds the runs of this arm and condition (B2's CrossQ BASE under default is A2's)."""
    if exp == "B2" and arm == "BASE" and cond == "default":
        return "A2"
    for a, src in EXPERIMENTS[exp]["arms"]:
        if a == arm:
            return src or exp
    return exp


def fig_curves(exp: str):
    spec = EXPERIMENTS[exp]
    nrow, ncol = len(spec["conds"]), len(spec["tasks"])
    row_h = 1.30
    left, right, bottom, top = 0.50, 0.14, 0.34, 0.40
    gap_x, gap_y = 0.36, 0.36
    H = top + bottom + nrow * row_h + (nrow - 1) * gap_y
    fig = plt.figure(figsize=(WIDTH, H))
    w = (WIDTH - left - right - (ncol - 1) * gap_x) / ncol
    letters = "abcdefghijkl"
    k = 0
    learner_name = "SAC" if spec["learner"] == "sac" else "CrossQ"
    refs = set()
    for i, cond in enumerate(spec["conds"]):
        for j, task in enumerate(spec["tasks"]):
            x0 = left + j * (w + gap_x)
            y0 = H - top - (i + 1) * row_h - i * gap_y
            ax = fig.add_axes([x0 / WIDTH, y0 / H, w / WIDTH, row_h / H])
            for z, (arm, src) in enumerate(spec["arms"]):
                origin = _source(exp, arm, cond)
                if origin != exp:
                    refs.add((arm, origin))
                arm_curve(ax, task, cond, spec["learner"], arm, spec["budget"], zorder=2 + z)
            steps_axis(ax, spec["budget"])
            num_formatter(ax)
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            text = TASK_LABEL[task] + (f", {cond}" if nrow > 1 or cond != "default" else "")
            panel_label(ax, letters[k], text)
            k += 1
            if j == 0:
                ylabel(ax, "Return")
            if i == nrow - 1:
                ax.set_xlabel("Environment steps")
    handles = []
    for arm, _ in spec["arms"]:
        origins = sorted({_source(exp, arm, c) for c in spec["conds"]})
        label = f"{learner_name} {ARM_LABEL[arm]}"
        if origins != [exp]:
            label += " (" + ", ".join(o for o in origins) + ")"
        h, _ = legend_line(arm)
        handles.append((h, label))
    top_legend(fig, [h for h, _ in handles], [l for _, l in handles], height=H)
    return save(fig, f"fig_curves_{exp}"), {"height_in": H, "references": sorted(refs)}


def _perf_profile(learner, budget, reps=10_000):
    from rliable import library as rly

    arms = ["BASE", "IER"]
    mats = {a: np.column_stack([cell_scores(t, "default", learner, a, budget).final.to_numpy() for t in TASKS])
            for a in arms}
    upper = max(1.5, float(max(m.max() for m in mats.values())))
    taus = np.linspace(0.0, upper, 61)
    np.random.seed(0)
    profiles, cis = rly.create_performance_profile(mats, taus, reps=reps)
    return taus, profiles, cis


def fig_rliable():
    rl = load_json("rliable.json")
    H = 2.2
    fig = plt.figure(figsize=(WIDTH, H))
    left, right, bottom, top = 0.80, 0.14, 0.36, 0.40
    widths = [1.0, 1.35, 1.35, 0.85]
    gaps = [0.55, 0.30, 0.85]
    scale = (WIDTH - left - right - sum(gaps)) / sum(widths)
    widths = [w * scale for w in widths]
    xs, x = [], left
    for i, w in enumerate(widths):
        xs.append(x)
        x += w + (gaps[i] if i < len(gaps) else 0)
    axes = [fig.add_axes([xs[i] / WIDTH, bottom / H, widths[i] / WIDTH, (H - top - bottom) / H]) for i in range(4)]
    groups = [("sac_1000k", "SAC, 1M"), ("crossq_300k", "CrossQ, 300k")]
    # (a) IQM of AUC
    ax = axes[0]
    for k, (key, name) in enumerate(groups):
        y = 1 - k
        for arm, dy in (("BASE", 0.15), ("IER", -0.15)):
            v = rl[key]["iqm"][arm]
            st = ARM_STYLE[arm]
            ax.plot([v["low"], v["high"]], [y + dy] * 2, color=st["color"], linewidth=1.0)
            ax.plot([v["estimate"]], [y + dy], marker=st["marker"], color=st["color"], markersize=4.2, linestyle="none")
    ax.set_yticks([1, 0]); ax.set_yticklabels([g[1] for g in groups]); ax.set_ylim(-0.6, 1.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(False); ax.grid(True, axis="x", color="#e3e3e3", linewidth=0.5)
    ax.set_xlabel("IQM of AUC")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    panel_label(ax, "a", dx=-40)
    info = {}
    # (b, c) performance profiles of normalized final return
    for idx, (learner, budget, name) in ((1, ("sac", LONG, "SAC, 1M")), (2, ("crossq", SHORT, "CrossQ, 300k"))):
        ax = axes[idx]
        taus, profiles, cis = _perf_profile(learner, budget)
        for arm in ("BASE", "IER"):
            st = ARM_STYLE[arm]
            ax.fill_between(taus, cis[arm][0], cis[arm][1], color=st["color"], alpha=0.2, linewidth=0)
            ax.plot(taus, profiles[arm], color=st["color"], linewidth=LW, linestyle=st["linestyle"])
        ax.set_xlim(0, taus[-1]); ax.set_ylim(0, 1.02)
        if idx == 1:
            ylabel(ax, "Share of runs above $\\tau$")
        panel_label(ax, "bc"[idx - 1], name)
        info[name] = {"tau_max": float(taus[-1])}
    mid = (axes[1].get_position().x0 + axes[2].get_position().x1) / 2
    fig.text(mid, 0.03 / H, "Normalized final return $\\tau$", ha="center", va="bottom", fontsize=8)
    # (d) probability of improvement
    ax = axes[3]
    for k, (key, name) in enumerate(groups):
        y = 1 - k
        p = rl[key]["probability_of_improvement"]
        ax.plot([p["low"], p["high"]], [y, y], color=ARM_STYLE["IER"]["color"], linewidth=1.0)
        ax.plot([p["estimate"]], [y], marker="D", color=ARM_STYLE["IER"]["color"], markersize=3.8, linestyle="none")
    ax.axvline(0.5, color="#555555", linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_yticks([1, 0]); ax.set_yticklabels([g[1] for g in groups]); ax.set_ylim(-0.6, 1.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(False); ax.grid(True, axis="x", color="#e3e3e3", linewidth=0.5)
    ax.set_xlim(0.25, 0.75); ax.set_xticks([0.3, 0.5, 0.7])
    ax.set_xlabel("P(IER > BASE)")
    panel_label(ax, "d", dx=-40)
    handles = [(Line2D([], [], color=ARM_STYLE[a]["color"], marker=ARM_STYLE[a]["marker"], markersize=3.8,
                       linewidth=LW, linestyle=ARM_STYLE[a]["linestyle"]), ARM_LABEL[a]) for a in ("BASE", "IER")]
    top_legend(fig, [h for h, _ in handles], [l for _, l in handles], height=H)
    return save(fig, "fig_rliable"), info


if __name__ == "__main__":
    from .common import apply_style

    apply_style()
    for e in EXPERIMENTS:
        print(e, fig_curves(e)[1])
    print(fig_rliable()[1])
