"""Quantities derived from the aggregated files, with checks against the frozen analysis outputs.

- Per-seed normalized AUC and final return of every cell, computed from curves.csv with normalizers.json the way
  iers/metrics.py does (trapezoid rule over the evaluation steps up to the budget, divided by the step span and the
  normalizer). Checked against cells.csv and gains.csv.
- Two-sided sign-flip permutation p-values (all 2^20 sign patterns) on per-seed AUC differences, for gains.csv rows
  outside the 38-contrast family of multiple_comparisons.csv. The same routine is checked on the family rows.
- Per-run retention after 200k steps, and trigger times reconstructed from repeats.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import CACHE, MULTI, load, normalizer

LONG, SHORT = 1_000_000, 300_000
RET_AFTER = 200_000


# ---------------------------------------------------------------- per-seed scores

def _auc(steps, returns, norm):
    return float(np.trapezoid(returns, steps) / (steps[-1] - steps[0]) / norm)


def cell_scores(task, condition, learner, arm, budget, source_budget=None) -> pd.DataFrame:
    """Per-seed normalized AUC and final return (mean of the last 5 evaluations) of a cell."""
    c = load("curves.csv")
    src = source_budget or budget
    part = c[(c.task == task) & (c.condition == condition) & (c.learner == learner) & (c.arm == arm) & (c.budget == src)]
    norm = normalizer(task, budget)
    rows = []
    for seed, g in part.groupby("seed"):
        g = g.sort_values("step")
        g = g[g.step <= budget]
        steps, rets = g.step.to_numpy(float), g["return"].to_numpy(float)
        rows.append({"seed": int(seed), "auc": _auc(steps, rets, norm), "final": float(np.mean(rets[-5:])) / norm,
                     "final_raw": float(np.mean(rets[-5:]))})
    return pd.DataFrame(rows).set_index("seed").sort_index()


def check_cells() -> dict:
    """Recompute auc_mean and final_mean of every cell in cells.csv. Returns the largest absolute differences."""
    cells = load("cells.csv")
    worst_auc = worst_final = 0.0
    for r in cells.itertuples():
        s = cell_scores(r.task, r.condition, r.learner, r.arm, int(r.budget))
        worst_auc = max(worst_auc, abs(s.auc.mean() - r.auc_mean))
        worst_final = max(worst_final, abs(s.final.mean() - r.final_mean))
    return {"cells": len(cells), "max_abs_diff_auc_mean": worst_auc, "max_abs_diff_final_mean": worst_final}


def gain_diffs(task, condition, learner, arm, budget) -> pd.Series:
    a = cell_scores(task, condition, learner, arm, budget).auc
    b = cell_scores(task, condition, learner, "BASE", budget).auc
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).sort_index()


# ---------------------------------------------------------------- sign-flip permutation

def sign_flip_p(diffs) -> float:
    """Two-sided exact sign-flip p: share of all 2^n sign patterns whose |mean| is at least the observed |mean|."""
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    patterns = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.float64) * 2 - 1  # (2^n, n)
    means = np.abs(patterns @ d) / n
    observed = abs(d.mean())
    return float(np.mean(means >= observed - 1e-12))


def gains_with_p(force: bool = False) -> pd.DataFrame:
    """gains.csv with p, Holm and BH. Rows in the 38-contrast family take its values, and the others get an unadjusted
    sign-flip p and no adjusted values."""
    path = CACHE / "gains_p.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    g = load("gains.csv").copy()
    fam = pd.read_csv(MULTI)
    ps, holm, bh, source, check = [], [], [], [], []
    for r in g.itertuples():
        d = gain_diffs(r.task, r.condition, r.learner, r.arm, int(r.budget))
        est_diff = abs(d.mean() - r.estimate)
        if r.learner == "sac" and r.budget == LONG and r.arm == "IER":
            hit = fam[(fam.id == "P1") & (fam.task == r.task)]
        elif r.learner == "crossq" and r.arm == "IER" and r.condition == "default":
            hit = fam[(fam.id == "P6") & (fam.task == r.task) & (fam.label == "IER minus BASE, CrossQ")]
        else:
            hit = fam.iloc[0:0]
        if len(hit) == 1:
            ps.append(float(hit.p.iloc[0])); holm.append(float(hit.holm.iloc[0])); bh.append(float(hit.bh.iloc[0]))
            source.append(f"family {hit.id.iloc[0]}")
            check.append(abs(sign_flip_p(d) - float(hit.p.iloc[0])))
        else:
            ps.append(sign_flip_p(d)); holm.append(np.nan); bh.append(np.nan); source.append("unadjusted")
            check.append(np.nan)
        if est_diff > 1e-9:
            raise AssertionError(f"gain estimate mismatch {r}")
    g["p"], g["holm"], g["bh"], g["p_source"], g["p_check_vs_family"] = ps, holm, bh, source, check
    CACHE.mkdir(parents=True, exist_ok=True)
    g.to_csv(path, index=False)
    return g


def check_family_p() -> dict:
    """Recompute the sign-flip p of the family's P1, P3 and P7 rows from per-seed AUC (checks the routine)."""
    fam = pd.read_csv(MULTI)
    out = {}
    for r in fam.itertuples():
        if r.id == "P1":
            d = gain_diffs(r.task, r.condition, "sac", "IER", LONG)
        elif r.id == "P3":
            a = cell_scores(r.task, r.condition, "sac", "IER3", SHORT).auc
            b = cell_scores(r.task, r.condition, "sac", "DUP", SHORT).auc
            d = a - b
        elif r.id == "P7":
            a = cell_scores(r.task, "default", "sac", "IER3", LONG).auc
            b = cell_scores(r.task, "default", "sac", "IER", LONG).auc
            d = a - b
        else:
            continue
        out[f"{r.id}|{r.task}|{r.condition}"] = {"estimate_diff": abs(d.mean() - r.estimate),
                                                 "p_diff": abs(sign_flip_p(d) - r.p)}
    return out


# ---------------------------------------------------------------- replays

def retention_after(task, condition, learner, arm, budget, after: int = RET_AFTER) -> pd.Series:
    """Per-run median retention ratio of the replays that ended after `after` steps (seeds without any: absent)."""
    rep = load("repeats.csv")
    part = rep[(rep.task == task) & (rep.condition == condition) & (rep.learner == learner) & (rep.arm == arm)
               & (rep.budget == budget) & (rep.total_steps > after)]
    return part.groupby("seed").ratio.median()


def trigger_times(task, condition="default", learner="sac", arm="IER", budget=LONG) -> pd.DataFrame:
    """One row per stored episode that was replayed at least once: its first replay's end step. A new stored
    episode is a new trigger, so the count of distinct stored returns per run counts the triggers that led to a
    replay. Triggers whose replays never ran (in the exploration phase or at the very end) are not visible here."""
    rep = load("repeats.csv")
    part = rep[(rep.task == task) & (rep.condition == condition) & (rep.learner == learner) & (rep.arm == arm)
               & (rep.budget == budget)]
    first = part.groupby(["seed", "stored_return"]).total_steps.min().reset_index()
    return first.rename(columns={"total_steps": "first_replay_step"})


def check_triggers() -> pd.DataFrame:
    runs = load("runs.csv")
    rows = []
    for task in ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"]:
        t = trigger_times(task)
        seen = t.groupby("seed").size()
        logged = runs[(runs.experiment == "A1") & (runs.task == task) & (runs.arm == "IER")].set_index("seed").trigger_count
        rows.append({"task": task, "logged_mean": logged.mean(), "seen_mean": seen.reindex(logged.index).fillna(0).mean(),
                     "runs_equal": int((seen.reindex(logged.index).fillna(0) == logged).sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(json.dumps(check_cells(), indent=1))
    fam = check_family_p()
    print("family check max estimate diff", max(v["estimate_diff"] for v in fam.values()),
          "max p diff", max(v["p_diff"] for v in fam.values()), "rows", len(fam))
    g = gains_with_p(force=True)
    print(g[["label", "task", "learner", "condition", "budget", "estimate", "p", "holm", "bh", "p_source", "p_check_vs_family"]].to_string())
    print(check_triggers().to_string())


# ---------------------------------------------------------------- optional: the family with exact sign-flip p

def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def _bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def _family_diffs(r) -> pd.Series:
    t, cnd = r.task, r.condition

    def gain(task, cond, learner, arm, budget, src=None):
        a = cell_scores(task, cond, learner, arm, budget, src).auc
        b = cell_scores(task, cond, learner, "BASE", budget, src).auc
        return a - b

    def ab(a, b, cond, budget):
        return cell_scores(t, cond, "sac", a, budget).auc - cell_scores(t, cond, "sac", b, budget).auc

    if r.id == "P1":
        return gain(t, "default", "sac", "IER", LONG)
    if r.id == "P2":
        return ab("IER", "IER_REL", "default", LONG)
    if r.id == "P3":
        return ab("IER3", "DUP", cnd, SHORT)
    if r.id == "P7":
        return ab("IER3", "IER", "default", LONG)
    if r.id == "P6" and r.label == "IER minus BASE, CrossQ":
        return gain(t, "default", "crossq", "IER", SHORT)
    if r.id == "P6":
        return gain(t, "default", "crossq", "IER", SHORT) - gain(t, "default", "sac", "IER", SHORT, LONG)
    if r.id == "P4":
        return gain(t, cnd, r.learner, "IER3", SHORT) - gain(t, "default", r.learner, "IER3", SHORT)
    raise KeyError(r.id)


FAMILY_IDS = ("P1", "P2", "P3", "P4", "P6", "P7")


def exact_family() -> "Path":
    """The 38 paired AUC contrasts of the fixed plan (P1 to P4, P6 and P7 in contrasts.csv) with an exact two-sided
    sign-flip p over all 2^20 sign patterns of the per-seed differences, and Holm and Benjamini-Hochberg values
    over the 38. Exploratory, added after the main results. Writes results/aggregated/multiple_comparisons.csv."""
    con = load("contrasts.csv")
    fam = con[con.id.isin(FAMILY_IDS)].reset_index(drop=True)
    if len(fam) != 38:
        raise AssertionError(f"expected 38 plan contrasts, found {len(fam)}")
    p = []
    for r in fam.itertuples():
        d = _family_diffs(r).to_numpy()
        if abs(d.mean() - r.estimate) > 1e-9:
            raise AssertionError((r.id, r.task, r.condition))
        p.append(sign_flip_p(d))
    out = fam.copy()
    out["p"], out["holm"], out["bh"] = p, _holm(p), _bh(p)
    MULTI.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MULTI, index=False)
    return MULTI
