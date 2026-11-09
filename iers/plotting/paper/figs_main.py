"""Main-text data figures: fig_reproduction, fig_effects, fig_filmstrip, fig_replay, fig_copy, fig_stress, fig_crossq
and fig_random_replay. Each is written as PDF and 300 dpi PNG to paper_assets/figures/main."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .common import (ARM_LABEL, ARM_STYLE, BAND_ALPHA, CACHE, COND_STYLE, GRAY, LW, MULTI, OI, ROOT, TASK_LABEL,
                    TASK_STYLE, TASKS, TASKS3, WIDTH, bootstrap_band, curve_part, load, load_json, num_formatter,
                    panel_label, plot_band, save, seed_matrix, steps_axis, steps_formatter, strip, top_legend)

LONG, SHORT = 1_000_000, 300_000


def family() -> pd.DataFrame:
    return pd.read_csv(MULTI)


# ---------------------------------------------------------------- shared drawing

def arm_curve(ax, task, condition, learner, arm, budget, source_budget=None, color=None, linestyle=None, lw=LW,
              alpha=BAND_ALPHA, zorder=2, label=None):
    part = curve_part(task, condition, learner, arm, budget, source_budget)
    m = seed_matrix(part, budget)
    if len(m) != 20:
        raise AssertionError(f"{task} {condition} {learner} {arm}: {len(m)} seeds")
    mean, low, high = bootstrap_band(m.to_numpy())
    st = ARM_STYLE.get(arm, {})
    return plot_band(ax, m.columns.to_numpy(), mean, low, high, color or st["color"],
                     linestyle or st.get("linestyle", "-"), lw=lw, alpha=alpha, zorder=zorder, label=label)


def dot(ax, x, est, low, high, color, marker, kind, size=4.2, zorder=3):
    """kind: 'filled' (BH < 0.05, or interval excludes 0 where no BH exists), 'hollow' (interval excludes 0,
    BH >= 0.05), 'gray' (interval includes 0)."""
    if kind == "gray":
        c, face, ms, elw = GRAY, GRAY, size * 0.78, 0.8
    elif kind == "hollow":
        c, face, ms, elw = color, "white", size, 1.0
    else:
        c, face, ms, elw = color, color, size, 1.0
    ax.plot([x, x], [low, high], color=c, linewidth=elw, solid_capstyle="butt", zorder=zorder)
    ax.plot([x], [est], marker=marker, markersize=ms, markerfacecolor=face, markeredgecolor=c,
            markeredgewidth=0.9, linestyle="none", zorder=zorder + 0.1)


def marker_kind(low, high, bh=None) -> str:
    excludes = low > 0 or high < 0
    if not excludes:
        return "gray"
    if bh is None or not np.isfinite(bh):
        return "filled"
    return "filled" if bh < 0.05 else "hollow"


def zero_line(ax, y=0.0):
    ax.axhline(y, color="#555555", linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)


def task_xaxis(ax, tasks, labels=None):
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(labels or [TASK_LABEL[t] for t in tasks])
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.tick_params(axis="x", length=0)


def legend_line(arm, label=None):
    st = ARM_STYLE[arm]
    return Line2D([], [], color=st["color"], linestyle=st["linestyle"], linewidth=LW), label or ARM_LABEL[arm]


def legend_marker(color, marker, label, face=None, size=4.2):
    return (Line2D([], [], color=color, marker=marker, linestyle="none", markersize=size,
                   markerfacecolor=face or color, markeredgecolor=color), label)


def xlabel_center(fig, axes, text, y=0.02):
    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    fig.text((left + right) / 2, y, text, ha="center", va="bottom", fontsize=8)


def ylabel(ax, text):
    ax.set_ylabel(text, labelpad=3)


# ---------------------------------------------------------------- reproduction

def fig_reproduction():
    fig, axes = strip(1.8, [1, 1, 1, 1], legend=True)
    arms = ["BASE", "IER", "IER_REL", "IER3"]
    for i, (ax, task) in enumerate(zip(axes, TASKS)):
        for z, arm in enumerate(arms):
            arm_curve(ax, task, "default", "sac", arm, LONG, zorder=2 + z)
        steps_axis(ax, LONG)
        num_formatter(ax)
        ax.set_ylim(bottom=0)
        panel_label(ax, "abcd"[i], TASK_LABEL[task])
    ylabel(axes[0], "Return")
    xlabel_center(fig, axes, "Environment steps")
    handles = [legend_line(a) for a in arms]
    top_legend(fig, [h for h, _ in handles], [l for _, l in handles])
    return save(fig, "fig_reproduction")


# ---------------------------------------------------------------- effects

def bh_lookup(pid, task, condition="default", label=None):
    fam = family()
    rows = fam[(fam.id == pid) & (fam.task == task) & (fam.condition == condition)]
    if label is not None:
        rows = rows[rows.label == label]
    return float(rows.bh.iloc[0]) if len(rows) == 1 else None


def fig_effects():
    fig, axes = strip(2.0, [2.0, 0.62], legend=True, gaps=[0.92])
    ax = axes[0]
    gains = load("gains.csv")
    arms = ["IER", "IER_REL", "IER3"]
    offsets = dict(zip(arms, (-0.22, 0.0, 0.22)))
    used_rules = {}
    for j, task in enumerate(TASKS):
        for arm in arms:
            r = gains[(gains.task == task) & (gains.learner == "sac") & (gains.budget == LONG) & (gains.arm == arm)
                      & (gains.condition == "default")].iloc[0]
            bh = bh_lookup("P1", task) if arm == "IER" else None
            used_rules[arm] = "BH" if arm == "IER" else "interval"
            st = ARM_STYLE[arm]
            dot(ax, j + offsets[arm], r.estimate, r.low, r.high, st["color"], st["marker"], marker_kind(r.low, r.high, bh))
    zero_line(ax)
    task_xaxis(ax, TASKS)
    ylabel(ax, "AUC difference to BASE")
    panel_label(ax, "a")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

    # (b) rliable aggregates: IQM of AUC for BASE and IER, and the probability of improvement.
    rl = load_json("rliable.json")
    pos = axes[1].get_position()
    fig.delaxes(axes[1])
    h_total = pos.height
    gap = 0.40 / fig.get_figheight()
    h_each = (h_total - gap) / 2
    ax_iqm = fig.add_axes([pos.x0, pos.y0 + h_each + gap, pos.width, h_each])
    ax_poi = fig.add_axes([pos.x0, pos.y0, pos.width, h_each])
    groups = [("sac_1000k", "SAC, 1M"), ("crossq_300k", "CrossQ, 300k")]
    for k, (key, name) in enumerate(groups):
        y = len(groups) - 1 - k
        for arm, dy in (("BASE", 0.17), ("IER", -0.17)):
            v = rl[key]["iqm"][arm]
            st = ARM_STYLE[arm]
            ax_iqm.plot([v["low"], v["high"]], [y + dy] * 2, color=st["color"], linewidth=1.0)
            ax_iqm.plot([v["estimate"]], [y + dy], marker=st["marker"], color=st["color"], markersize=4.2,
                        linestyle="none")
        p = rl[key]["probability_of_improvement"]
        ax_poi.plot([p["low"], p["high"]], [y, y], color=ARM_STYLE["IER"]["color"], linewidth=1.0)
        ax_poi.plot([p["estimate"]], [y], marker="D", color=ARM_STYLE["IER"]["color"], markersize=3.8,
                    linestyle="none")
    for a in (ax_iqm, ax_poi):
        a.set_yticks(range(len(groups)))
        a.set_yticklabels([g[1] for g in groups][::-1])
        a.set_ylim(-0.6, len(groups) - 0.4)
        a.grid(False)
        a.grid(True, axis="x", color="#e3e3e3", linewidth=0.5)
        a.tick_params(axis="y", length=0)
    ax_iqm.set_xlabel("IQM of AUC", labelpad=1.5)
    ax_iqm.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax_poi.axvline(0.5, color="#555555", linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    ax_poi.set_xlim(0.25, 0.75)
    ax_poi.set_xticks([0.3, 0.5, 0.7])
    ax_poi.set_xlabel("P(IER > BASE)", labelpad=1.5)
    panel_label(ax_iqm, "b", dx=-50)
    handles = [legend_marker(ARM_STYLE[a]["color"], ARM_STYLE[a]["marker"], ARM_LABEL[a]) for a in ("BASE", "IER", "IER_REL", "IER3")]
    top_legend(fig, [h for h, _ in handles], [l for _, l in handles])
    xlabel_center(fig, [ax], "", y=0.0)
    return save(fig, "fig_effects"), used_rules


# ---------------------------------------------------------------- filmstrip

def fig_filmstrip():
    from . import sim
    meta = json.loads((CACHE / "filmstrip.json").read_text())
    frames = sim.load_frames()
    H = 2.4
    fig = plt.figure(figsize=(WIDTH, H))
    left, right, top = 0.50, 0.02, 0.03
    fw, fh = sim.FILM_PX[0] / 300, sim.FILM_PX[1] / 300
    n = len(sim.FILM_STEPS)
    gap = (WIDTH - left - right - n * fw) / (n - 1)
    cap = 0.118  # caption line under each frame
    block_gap = (H - top - 4 * (fh + cap)) - 0.0
    y = H - top
    for b, task in enumerate(("HalfCheetah-v4", "walker-walk")):
        info = meta["tasks"][task]
        block_top = y
        for row, name in (("stored", "stored"), ("fresh", "replay")):
            y -= fh
            for k, step in enumerate(sim.FILM_STEPS):
                x = left + k * (fw + gap)
                ax = fig.add_axes([x / WIDTH, y / H, fw / WIDTH, fh / H])
                key = f"{task}|{row}|{step}"
                if key in frames:
                    ax.imshow(frames[key], interpolation="none", aspect="auto")
                    ret = info["rows"][row]["running_return"][str(step)]
                    text = f"step {step}, return {ret:.0f}"
                else:
                    text = f"step {step}, ended"
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("#b0b0b0")
                ax.grid(False)
                fig.text((x + fw / 2) / WIDTH, (y - 0.018) / H, text, ha="center", va="top", fontsize=6.5)
            fig.text((left - 0.05) / WIDTH, (y + fh / 2) / H, name, ha="right", va="center", fontsize=7)
            y -= cap
        block_bottom = y + cap * 0.35
        fig.text(0.07 / WIDTH, (block_top + block_bottom) / 2 / H, f"({'ab'[b]}) {TASK_LABEL[task]}", rotation=90,
                 ha="center", va="center", fontsize=8)
        y -= block_gap
    return save(fig, "fig_filmstrip"), meta


# ---------------------------------------------------------------- replay

from .derived import retention_after, trigger_times  # noqa: E402

BIN = 50_000
LATE_STORED = 100_000  # replay-check episodes stored after this many training steps (panel b)


def retention_bins(task, condition="default", learner="sac", arm="IER", budget=LONG):
    rep = load("repeats.csv")
    part = rep[(rep.task == task) & (rep.condition == condition) & (rep.learner == learner) & (rep.arm == arm)
               & (rep.budget == budget)].copy()
    # A replay ending exactly on a bin edge belongs to the bin it ends in, (k*50k, (k+1)*50k].
    part["bin"] = (part.total_steps - 1) // BIN
    per_seed = part.groupby(["bin", "seed"]).ratio.median().reset_index()
    out = per_seed.groupby("bin").ratio.agg(median="median", q25=lambda s: s.quantile(0.25),
                                            q75=lambda s: s.quantile(0.75), seeds="size").reset_index()
    out["x"] = (out["bin"] + 0.5) * BIN
    return out


def deviation_curves(task, late=LATE_STORED, min_share=0.5):
    """Median and quartiles of the official fresh-reset deviation per step, over replays still running, cut where
    fewer than `min_share` of the replays are still running. Also the identical-start maximum."""
    rc = load("replay_check.csv")
    dev = np.load(ROOT / "results" / "aggregated" / "replay_check_deviation.npz")
    rows = rc[(rc.task == task) & (rc.trigger_at_step > late)]
    fresh = [dev[f"{r.task}|{r.seed}|{r.trigger}|{r.start}"] for r in rows.itertuples() if r.start != "identical"]
    ident = [dev[f"{r.task}|{r.seed}|{r.trigger}|{r.start}"] for r in rows.itertuples() if r.start == "identical"]
    return _median_curve(fresh, min_share), float(max((np.max(c) if len(c) else 0.0) for c in ident)), len(ident), len(fresh)


def _median_curve(curves, min_share=0.5):
    L = max(len(c) for c in curves)
    m = np.full((len(curves), L), np.nan)
    for i, c in enumerate(curves):
        m[i, :len(c)] = c
    running = np.isfinite(m).mean(axis=0)
    keep = running >= min_share
    steps = np.arange(1, L + 1)[keep]
    sub = m[:, keep]
    return steps, np.nanmedian(sub, axis=0), np.nanpercentile(sub, 25, axis=0), np.nanpercentile(sub, 75, axis=0)


def f64_curves(task, min_share=0.5):
    from . import sim
    meta = json.loads((CACHE / "f64_raw.json").read_text())
    arr = sim.load_f64_arrays()
    eps = [e for e in meta["episodes"] if e["task"] == task and e["action_dtype"] == "float32"]
    curves = [arr[f"{task}|{e['trigger_step']}|raw"] for e in eps]
    return _median_curve(curves, min_share), len(curves)


def fig_replay():
    fig, axes = strip(1.9, [1, 1, 1, 1], legend=True, gaps=[0.50, 0.50, 0.44], top=0.52)
    info = {}
    # (a) retention over training
    ax = axes[0]
    for task in TASKS:
        b = retention_bins(task)
        c = TASK_STYLE[task]["color"]
        lo, hi = -0.2, 1.1
        ax.fill_between(b.x, b.q25.clip(lo, hi), b.q75.clip(lo, hi), color=c, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(b.x, b["median"].clip(lo, hi), color=c, linewidth=LW, marker="o" if len(b) == 1 else None,
                markersize=2.8)
        info[f"a|{task}"] = b.to_dict("list")
    ax.set_ylim(-0.2, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    steps_axis(ax, LONG)
    ax.set_xlabel("Environment steps")
    ylabel(ax, "Retention ratio")
    panel_label(ax, "a")

    # (b) replay check: distance to the stored path within the episode
    ax = axes[1]
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-7, linscale=0.35)
    for task in ("HalfCheetah-v4", "Hopper-v4", "walker-walk"):
        c = TASK_STYLE[task]["color"]
        (steps, med, q25, q75), ident_max, n_ident, n_fresh = deviation_curves(task)
        ax.fill_between(steps, q25, q75, color=c, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(steps, med, color=c, linewidth=LW)
        (fs, fmed, _, _), n_f64 = f64_curves(task)
        ax.plot(fs, fmed, color=c, linewidth=1.0, linestyle=(0, (1, 1.2)))
        info[f"b|{task}"] = {"fresh_replays": n_fresh, "identical_replays": n_ident, "identical_max": ident_max,
                             "fresh_last_step_shown": int(steps[-1]), "f64_episodes": n_f64,
                             "f64_last_step_shown": int(fs[-1]),
                             "fresh_median_at": {s: float(med[s - 1]) for s in (1, 10, 100, 1000) if s <= len(med)},
                             "f64_median_at": {s: float(fmed[s - 1]) for s in (1, 10, 100, 1000) if s <= len(fmed)}}
    ax.axhline(0, color="black", linestyle=(0, (3, 2)), linewidth=0.9, zorder=4)
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 1e-6, 1e-3, 1, 100])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: "0" if v == 0 else ("1" if v == 1 else ("100" if v == 100 else f"$10^{{{int(np.log10(v))}}}$"))))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Step within the episode")
    ylabel(ax, "Distance to stored path")
    panel_label(ax, "b")

    # (c) cumulative triggers
    ax = axes[2]
    grid = np.arange(0, LONG + 1, 10_000)
    for task in TASKS:
        t = trigger_times(task)
        seeds = sorted(load("runs.csv").query("experiment == 'A1' and arm == 'IER' and task == @task").seed)
        m = np.array([[np.sum(t[t.seed == s].first_replay_step <= g) for g in grid] for s in seeds], dtype=float)
        c = TASK_STYLE[task]["color"]
        ax.fill_between(grid, np.percentile(m, 25, axis=0), np.percentile(m, 75, axis=0), color=c, alpha=BAND_ALPHA,
                        linewidth=0)
        ax.plot(grid, np.median(m, axis=0), color=c, linewidth=LW)
        last = t.groupby("seed").first_replay_step.max()
        info[f"c|{task}"] = {"median_final": float(np.median(m[:, -1])),
                             "median_last_trigger_replay_step": float(last.median())}
    steps_axis(ax, LONG)
    ax.set_xlabel("Environment steps")
    ylabel(ax, "Triggers (cumulative)")
    panel_label(ax, "c")

    # (d) repetition share against retention after 200k steps
    ax = axes[3]
    runs = load("runs.csv")
    for task in TASKS:
        r = runs[(runs.experiment == "A1") & (runs.arm == "IER") & (runs.task == task)].set_index("seed")
        share = r.repetition_share * 100
        ret = retention_after(task, "default", "sac", "IER", LONG)
        st = TASK_STYLE[task]
        info[f"d|{task}"] = {"share_median": float(share.median()), "share_q25": float(share.quantile(0.25)),
                             "share_q75": float(share.quantile(0.75)), "share_mean": float(share.mean()),
                             "retention_seeds": int(len(ret))}
        if len(ret) == 0:
            continue
        info[f"d|{task}"].update({"ret_median": float(ret.median()), "ret_q25": float(ret.quantile(0.25)),
                                  "ret_q75": float(ret.quantile(0.75))})
        x, y = share.median(), ret.median()
        ax.plot([share.quantile(0.25), share.quantile(0.75)], [y, y], color=st["color"], linewidth=1.0)
        ax.plot([x, x], [ret.quantile(0.25), ret.quantile(0.75)], color=st["color"], linewidth=1.0)
        ax.plot([x], [y], marker=st["marker"], color=st["color"], markersize=4.2, linestyle="none")
    zero_line(ax)
    ax.set_ylim(-0.21, 0.19)
    ax.set_yticks([-0.1, 0, 0.1])
    ax.set_xlim(-2, 42)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Replay share (%)")
    ylabel(ax, "Retention after 200k")
    panel_label(ax, "d")
    ax.text(0.06, 0.03, "finger-turn_hard:\nno replays\nafter 200k", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color=TASK_STYLE["finger-turn_hard"]["color"], linespacing=1.05)

    handles = [(Line2D([], [], color=TASK_STYLE[t]["color"], marker=TASK_STYLE[t]["marker"], markersize=3.5,
                       linewidth=LW), TASK_LABEL[t]) for t in TASKS]
    styles = [(Line2D([], [], color="#444444", linewidth=LW), "(b) fresh reset"),
              (Line2D([], [], color="#444444", linewidth=1.0, linestyle=(0, (1, 1.2))), "(b) same start, float64 actions"),
              (Line2D([], [], color="black", linewidth=0.9, linestyle=(0, (3, 2))), "(b) identical start")]
    leg1 = top_legend(fig, [h for h, _ in handles], [l for _, l in handles])
    fig.add_artist(leg1)
    top_legend(fig, [h for h, _ in styles], [l for _, l in styles], y=1 - 0.205 / fig.get_figheight())
    return save(fig, "fig_replay"), info


# ---------------------------------------------------------------- copy control

CONTRAST_COLOR = {"IER3 minus BASE": ARM_STYLE["IER3"]["color"], "DUP minus BASE": ARM_STYLE["DUP"]["color"],
                  "IER3 minus DUP": OI["orange"]}
COND_MARKER = {"default": "o", "noise5": "s"}


def contrast_row(label, task, condition, learner="sac", budget=SHORT):
    if label == "IER3 minus DUP":
        c = load("contrasts.csv")
        r = c[(c.id == "P3") & (c.task == task) & (c.condition == condition)].iloc[0]
        return r, bh_lookup("P3", task, condition)
    g = load("gains.csv")
    r = g[(g.label == label) & (g.task == task) & (g.condition == condition) & (g.learner == learner)
          & (g.budget == budget)].iloc[0]
    return r, None


def fig_copy():
    fig, axes = strip(1.9, [1, 1, 1, 2.75], legend=True, gaps=[0.32, 0.32, 0.55], top=0.52)
    arms = ["BASE", "IER3", "DUP"]
    for i, task in enumerate(TASKS3):
        ax = axes[i]
        for z, arm in enumerate(arms):
            arm_curve(ax, task, "default", "sac", arm, SHORT, zorder=2 + z)
        steps_axis(ax, SHORT)
        ax.set_xticks([0, 150_000, 300_000])
        ax.set_xticklabels(["0", "150k", "300k"])
        num_formatter(ax)
        ax.set_ylim(bottom=0)
        panel_label(ax, "abc"[i], TASK_LABEL[task])
    ylabel(axes[0], "Return")
    xlabel_center(fig, axes[:3], "Environment steps")
    ax = axes[3]
    labels = ["IER3 minus BASE", "DUP minus BASE", "IER3 minus DUP"]
    rules = {}
    for j, task in enumerate(TASKS3):
        for k, label in enumerate(labels):
            for cond, dx in (("default", -0.065), ("noise5", 0.065)):
                r, bh = contrast_row(label, task, cond)
                rules[label] = "BH" if label == "IER3 minus DUP" else "interval"
                dot(ax, j + (k - 1) * 0.27 + dx, r.estimate, r.low, r.high, CONTRAST_COLOR[label], COND_MARKER[cond],
                    marker_kind(r.low, r.high, bh), size=3.8)
    zero_line(ax)
    task_xaxis(ax, TASKS3)
    ax.set_xlim(-0.45, 2.45)
    ylabel(ax, "AUC difference")
    panel_label(ax, "d", dx=-8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    h1 = [legend_line(a) for a in arms]
    leg1 = top_legend(fig, [h for h, _ in h1], [l for _, l in h1])
    fig.add_artist(leg1)
    h2 = [(Line2D([], [], color=CONTRAST_COLOR[l], linewidth=1.0, marker="|", markersize=5), "(d) " + l.replace(" minus ", " − "))
          for l in labels]
    h2 += [legend_marker("black", COND_MARKER[c], f"(d) {c}", size=3.8) for c in ("default", "noise5")]
    top_legend(fig, [h for h, _ in h2], [l for _, l in h2], y=1 - 0.205 / fig.get_figheight())
    return save(fig, "fig_copy"), rules


# ---------------------------------------------------------------- stress

def fig_stress():
    from matplotlib.patches import Patch

    fig, axes = strip(1.9, [2.0, 1.0], legend=True, gaps=[0.62], top=0.52)
    ax = axes[0]
    g = load("gains.csv")
    entries = [("sac", c) for c in ("default", "noise5", "noise10", "dyn20")] + [("crossq", c) for c in ("default", "noise5")]
    offsets = np.array([-0.33, -0.21, -0.09, 0.03, 0.20, 0.32])
    for j, task in enumerate(TASKS3):
        for (learner, cond), dx in zip(entries, offsets):
            r = g[(g.label == "IER3 minus BASE") & (g.task == task) & (g.condition == cond) & (g.learner == learner)
                  & (g.budget == SHORT)].iloc[0]
            marker = TASK_STYLE[task]["marker"] if learner == "sac" else "X"
            dot(ax, j + dx, r.estimate, r.low, r.high, COND_STYLE[cond], marker, marker_kind(r.low, r.high),
                size=4.4 if learner == "crossq" else 4.0)
    zero_line(ax)
    task_xaxis(ax, TASKS3)
    ylabel(ax, "AUC difference to BASE")
    panel_label(ax, "a")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    ax = axes[1]
    c = load("contrasts.csv")
    p5 = c[(c.id == "P5") & (c.learner == "sac") & c.label.str.endswith("all tasks")].iloc[0]
    sac = g[(g.label == "IER3 minus BASE") & (g.learner == "sac") & (g.budget == SHORT)]
    for r in sac.itertuples():
        # default is drawn larger and below, so a default marker hidden under another keeps a visible rim
        big = r.condition == "default"
        ax.plot([r.start_distance_mean], [r.estimate], marker=TASK_STYLE[r.task]["marker"], color=COND_STYLE[r.condition],
                markersize=5.8 if big else 4.4, linestyle="none", markeredgecolor=COND_STYLE[r.condition],
                zorder=2.5 if big else 3)
    zero_line(ax)
    ax.set_xlabel("Mean start distance")
    ylabel(ax, "AUC difference to BASE")
    ax.text(0.03, 0.97, f"Spearman {p5.estimate:.2f}\n[{p5.low:.2f}, {p5.high:.2f}]", transform=ax.transAxes,
            ha="left", va="top", fontsize=7)
    ax.set_xlim(0, 2.4)
    ax.set_ylim(-0.105, 0.14)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    panel_label(ax, "b")
    h1 = [(Patch(facecolor=COND_STYLE[c], edgecolor="none"), c) for c in ("default", "noise5", "noise10", "dyn20")]
    leg1 = top_legend(fig, [h for h, _ in h1], [l for _, l in h1])
    fig.add_artist(leg1)
    h2 = [legend_marker("#333333", TASK_STYLE[t]["marker"], f"SAC, {TASK_LABEL[t]}") for t in TASKS3]
    h2 += [legend_marker("#333333", "X", "CrossQ", size=4.4)]
    top_legend(fig, [h for h, _ in h2], [l for _, l in h2], y=1 - 0.205 / fig.get_figheight())
    return save(fig, "fig_stress"), {"P5": p5.to_dict()}


# ---------------------------------------------------------------- CrossQ

def fig_crossq():
    fig, axes = strip(1.9, [1, 1, 1, 1], legend=True)
    for i, (ax, task) in enumerate(zip(axes, TASKS)):
        arm_curve(ax, task, "default", "sac", "BASE", SHORT, source_budget=LONG, color=GRAY, linestyle="-", lw=0.8,
                  alpha=0.12, zorder=1)
        arm_curve(ax, task, "default", "crossq", "BASE", SHORT, zorder=3)
        arm_curve(ax, task, "default", "crossq", "IER", SHORT, zorder=4)
        steps_axis(ax, SHORT)
        num_formatter(ax)
        ax.set_ylim(bottom=0)
        panel_label(ax, "abcd"[i], TASK_LABEL[task])
    ylabel(axes[0], "Return")
    xlabel_center(fig, axes, "Environment steps")
    handles = [(Line2D([], [], color=ARM_STYLE["BASE"]["color"], linewidth=LW), "CrossQ BASE"),
               (Line2D([], [], color=ARM_STYLE["IER"]["color"], linewidth=LW), "CrossQ IER"),
               (Line2D([], [], color=GRAY, linewidth=0.8), "SAC BASE")]
    top_legend(fig, [h for h, _ in handles], [l for _, l in handles])
    return save(fig, "fig_crossq")


# ---------------------------------------------------------------- random replay (E1, exploratory)

RANDOM_STYLE = {"color": OI["reddish_purple"], "marker": "X", "linestyle": (0, (4, 1.2, 1, 1.2, 1, 1.2))}
E1_RAW = ROOT / "results" / "raw" / "E1"


def e1_matrix():
    rows = {}
    for seed in range(20):
        t = pd.read_csv(E1_RAW / f"walker-walk__default__sac__IER3_RANDOM__seed{seed}" / "eval.csv")
        rows[seed] = t.groupby("total_steps")["return"].mean()
    return pd.DataFrame(rows).T.sort_index()


def fig_random_replay():
    e1 = json.loads((ROOT / "results" / "aggregated" / "e1.json").read_text())
    fig, axes = strip(1.9, [2.0, 1.0], legend=True, gaps=[0.62])
    ax = axes[0]
    for z, arm in enumerate(["BASE", "IER3", "DUP"]):
        arm_curve(ax, "walker-walk", "default", "sac", arm, SHORT, zorder=2 + z)
    m = e1_matrix()
    norm = e1["normalizer"]
    steps = m.columns.to_numpy(float)
    auc = [np.trapezoid(m.loc[s].to_numpy(), steps) / (steps[-1] - steps[0]) / norm for s in m.index]
    check = {"e1_mean_auc_recomputed": float(np.mean(auc)),
             "e1_mean_auc_file": e1["mean_normalized_auc"]["IER3_RANDOM"], "seeds": len(m)}
    mean, low, high = bootstrap_band(m.to_numpy())
    plot_band(ax, steps, mean, low, high, RANDOM_STYLE["color"], RANDOM_STYLE["linestyle"], zorder=6)
    steps_axis(ax, SHORT)
    ax.set_ylim(bottom=0)
    num_formatter(ax)
    ax.set_xlabel("Environment steps")
    ylabel(ax, "Return")
    panel_label(ax, "a", "walker-walk")
    ax = axes[1]
    names = [("IER3_RANDOM minus IER3", "IER3"), ("IER3_RANDOM minus BASE", "BASE"), ("IER3_RANDOM minus DUP", "DUP")]
    for j, (key, short) in enumerate(names):
        r = e1["contrasts"][key]
        dot(ax, j, r["estimate"], r["low"], r["high"], RANDOM_STYLE["color"], RANDOM_STYLE["marker"],
            marker_kind(r["low"], r["high"]), size=4.6)
    zero_line(ax)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s for _, s in names])
    ax.set_xlim(-0.5, 2.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("IER3-RANDOM minus")
    ylabel(ax, "AUC difference")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    panel_label(ax, "b")
    handles = [legend_line(a) for a in ("BASE", "IER3", "DUP")]
    handles.append((Line2D([], [], color=RANDOM_STYLE["color"], linestyle=RANDOM_STYLE["linestyle"], linewidth=LW),
                    "IER3-RANDOM"))
    top_legend(fig, [h for h, _ in handles], [l for _, l in handles])
    return save(fig, "fig_random_replay"), check
