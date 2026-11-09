"""LaTeX tables (booktabs): tab_reproduction, tab_verdicts, tab_all_contrasts (longtable), tab_repetition."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .common import AGG, MULTI, ROOT, TAB_OUT, TASK_TEX, TASKS, TASKS3, load, load_json
from .derived import gains_with_p, retention_after

LONG, SHORT = 1_000_000, 300_000
M = "$-$"
ARM_TEX = {"BASE": "BASE", "IER": "IER", "IER_REL": "IER-REL", "IER3": "IER3", "IER3_REL": "IER3-REL", "DUP": "DUP"}
LEARNER_TEX = {"sac": "SAC", "crossq": "CrossQ", "both": "both", "all": "all"}


def family() -> pd.DataFrame:
    return pd.read_csv(MULTI)


def num(v, d=3, keep_sign=False) -> str:
    """Fixed decimals. A value that rounds to zero drops its minus sign, except interval bounds (keep_sign), where
    the sign tells whether the interval excludes zero."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:.{d}f}"
    if float(s) == 0 and not keep_sign:
        s = s.lstrip("-")
    return s.replace("-", M)


def ci(lo, hi, d=3) -> str:
    return f"[{num(lo, d, True)}, {num(hi, d, True)}]"


def steps_tex(b) -> str:
    b = int(b)
    return "1M" if b == LONG else f"{b // 1000}k" if b else "--"


def pval(p) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def write(name: str, text: str) -> None:
    TAB_OUT.mkdir(parents=True, exist_ok=True)
    (TAB_OUT / name).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- tab_reproduction

def retention_summary(task, condition, learner, arm, budget):
    r = retention_after(task, condition, learner, arm, budget)
    return (float(r.median()) if len(r) else np.nan), len(r)


def tab_reproduction():
    cells = load("cells.csv")
    info = {}
    lines = [r"\begin{tabular}{lccrrr}", r"\toprule",
             r" & BASE & IER & Replay & Triggers & Retention \\",
             r"Task & final return & final return & share (\%) & per run & after 200k \\",
             r"\midrule"]
    for learner, budget, title in (("sac", LONG, "SAC, 1M steps"), ("crossq", SHORT, "CrossQ, 300k steps")):
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}}}} \\")
        for task in TASKS:
            def row(arm):
                return cells[(cells.task == task) & (cells.condition == "default") & (cells.learner == learner)
                             & (cells.arm == arm) & (cells.budget == budget)].iloc[0]
            b, i = row("BASE"), row("IER")
            fmt = lambda r: f"{r.final_mean * r.normalizer:.0f} [{r.final_low * r.normalizer:.0f}, {r.final_high * r.normalizer:.0f}]"
            ret, n = retention_summary(task, "default", learner, "IER", budget)
            lines.append(f"{TASK_TEX[task]} & {fmt(b)} & {fmt(i)} & {i.repetition_share * 100:.1f} & "
                         f"{i.trigger_count:.1f} & {num(ret, 2)} \\\\")
            info[f"{learner}|{task}"] = {"base": fmt(b), "ier": fmt(i), "share_pct": round(i.repetition_share * 100, 2),
                                         "triggers": i.trigger_count, "retention_after_200k_median": ret,
                                         "seeds_with_replays_after_200k": n}
        if learner == "sac":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_reproduction.tex", ""
          + "\n".join(lines) + "\n")
    return info


# ---------------------------------------------------------------- tab_verdicts

UP, DOWN, NONE = r"$\blacktriangle$", r"$\blacktriangledown$", r"$\circ$"


def symbol(row) -> str:
    if row.low > 0:
        return UP
    if row.high < 0:
        return DOWN
    return NONE


def frozen_verdicts():
    sys.path.insert(0, str(ROOT))
    from iers import verdicts as v

    table = load("contrasts.csv")
    rl = load_json("rliable.json")
    return {x.question: (x.verdict, x.reason) for x in v.all_verdicts(table, rl)}


def verdict_word(text: str) -> str:
    if text.startswith("mixed"):
        return "Mixed"
    if text.startswith("the gain depends on the schedule"):
        return "Depends on schedule"
    if text.startswith("no clear dependence"):
        return "No clear dependence"
    if text.startswith("holds"):
        return "Holds"
    if text.startswith("does not hold"):
        return "Does not hold"
    if text.startswith("not available"):
        return "Not available"
    return text


READING = {  # plain reading of each verdict, shown with the formal rule verdict where it differs in wording
    "Q1 gain": ("Task-dependent (mixed)", "Mixed"),
    "Q1 schedule": ("Matters on one task", "Depends on schedule"),
    "Q2": ("Better on walker-walk (holds)", "Holds"),
    "Q3": ("Far below 1 (mixed{a})", "Mixed"),
    "Q4": ("No clear reduction (does not hold)", "Does not hold"),
    "Q5": ("No clear gain (does not hold)", "Does not hold"),
    "Q6": ("No clear loss (holds)", "Holds"),
    "Q7": ("No clear link (does not hold)", "Does not hold"),
}
LABEL = {"Q1 gain": "Q1 Gain over BASE", "Q1 schedule": "Q1 Schedule", "Q2": r"Q2 Re-execution vs.\ copy",
         "Q3": "Q3 Retention", "Q4": "Q4 Gain under noise5", "Q5": "Q5 Gain for CrossQ", "Q6": r"Q6 Fixed $\RN=3$",
         "Q7": "Q7 Early retention"}


def tab_verdicts():
    c = load("contrasts.csv")
    fam = family()
    verdicts = frozen_verdicts()
    for q, (_, word) in READING.items():
        if verdict_word(verdicts[q][0]) != word:
            raise AssertionError(f"{q}: rule verdict {verdict_word(verdicts[q][0])!r}, reading written for {word!r}")
    notes, info = [], {}
    marks = iter("abcdefgh")

    def cell(rows_main, rows_other=None, describe=None):
        """Symbol of the main cell, with a footnote when another cell of the same task has a different symbol."""
        if rows_main is None or not len(rows_main):
            return "--"
        s = symbol(rows_main.iloc[0])
        if rows_other is not None and len(rows_other):
            differ = [r for r in rows_other.itertuples() if symbol(r) != s]
            if differ:
                m = next(marks)
                notes.append((m, describe(rows_main.iloc[0], differ)))
                return s + f"$^{{\\mathrm{{{m}}}}}$"
        return s

    rows = []
    for q, pid in (("Q1 gain", "P1"), ("Q1 schedule", "P2")):
        rows.append((q, [cell(c[(c.id == pid) & (c.task == t)]) for t in TASKS]))
    cells = []
    for t in TASKS:
        main = c[(c.id == "P3") & (c.task == t) & (c.condition == "default")]
        other = c[(c.id == "P3") & (c.task == t) & (c.condition != "default")]
        cells.append(cell(main, other, lambda m, d: f"{TASK_TEX[m.task]}: the noise5 cell differs.") if len(main) else "--")
    rows.append(("Q2", cells))
    cells = []
    for t in TASKS:
        p9 = c[(c.id == "P9") & (c.task == t)]
        main = p9[(p9.learner == "sac") & (p9.condition == "default") & (p9.budget == LONG) & p9.label.str.startswith("IER:")]
        cells.append(cell(main, p9.drop(main.index), lambda m, d: f"{TASK_TEX[m.task]}: other cells differ."))
    rows.append(("Q3", cells))
    p10 = c[c.id == "P10"]
    pos10 = p10[p10.low > 0]
    m3 = next(marks)
    words = {1: "one", 2: "two", 3: "three"}
    notes.append((m3, "The Q3 rule also needs the within-run correlation of start distance and retention to be clearly "
                  f"positive in no cell, and it is in {words.get(len(pos10), str(len(pos10)))}, " + " and ".join(
                      f"{LEARNER_TEX[r.learner]} {ARM_TEX[r.label.split(':')[0]]} on {TASK_TEX[r.task]} under "
                      f"{r.condition}, with a value of {num(r.estimate, 2)}" for r in pos10.itertuples()) + "."))
    cells = []
    for t in TASKS:
        p4 = c[(c.id == "P4") & (c.task == t)]
        if not len(p4):
            cells.append("--")
            continue
        main = p4[(p4.learner == "sac") & (p4.condition == "noise5")]

        def q4_text(m, d):
            clear = [r for r in d if symbol(r) != NONE]
            unclear = [r for r in d if symbol(r) == NONE]
            parts = []
            if unclear:
                parts.append("not clear " + " or ".join(
                    f"under {r.condition}" if r.learner == "sac" else f"for {LEARNER_TEX[r.learner]}" for r in unclear))
            for r in clear:
                f = fam[(fam.id == "P4") & (fam.task == r.task) & (fam.condition == r.condition) & (fam.learner == r.learner)]
                text = f"clear under {r.condition}"
                if len(f) and 0.05 < f.p.iloc[0] < 0.07:
                    text += f", borderline at sign-flip p = {f.p.iloc[0]:.3f}"
                parts.append(text)
            return f"{TASK_TEX[m.task]}: " + ", ".join(parts) + "."
        cells.append(cell(main, p4.drop(main.index), q4_text))
    rows.append(("Q4", cells))
    rows.append(("Q5", [cell(c[(c.id == "P6") & (c.task == t) & (c.label == "IER minus BASE, CrossQ")]) for t in TASKS]))
    rows.append(("Q6", [cell(c[(c.id == "P7") & (c.task == t)]) for t in TASKS]))
    p8 = c[c.id == "P8"].iloc[0]
    rows.append(("Q7", [rf"\multicolumn{{4}}{{c}}{{{symbol(p8)} (one cell across tasks)}}"]))

    head = " & ".join(["Question"] + [TASK_TEX[t] for t in TASKS] + ["Reading (rule)"])
    lines = [r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccl@{}}", r"\toprule", head + r" \\", r"\midrule"]
    for q, cells in rows:
        reading = READING[q][0].replace("{a}", f"$^{{\\mathrm{{{m3}}}}}$")
        lines.append(" & ".join([LABEL[q]] + cells + [reading]) + r" \\")
    lines += [r"\bottomrule", r"\noalign{\vskip 2pt}"]
    first, rest = notes[0], notes[1:]
    order = [n for n in notes if n[0] == m3] + [n for n in notes if n[0] != m3]
    lines.append(rf"\multicolumn{{6}}{{@{{}}p{{\linewidth}}@{{}}}}{{\footnotesize $^{{\mathrm{{{order[0][0]}}}}}$\,{order[0][1]}}} \\")
    if len(order) > 1:
        lines.append(r"\multicolumn{6}{@{}p{\linewidth}@{}}{\footnotesize " + " ".join(
            f"$^{{\\mathrm{{{m}}}}}$\\,{t}" for m, t in order[1:]) + r"} \\")
    lines.append(r"\end{tabular*}")
    write("tab_verdicts.tex", "\n".join(lines) + "\n")
    info["verdicts"] = verdicts
    info["footnotes"] = notes
    return info


# ---------------------------------------------------------------- tab_all_contrasts (longtable)

def _label(text: str) -> str:
    for k, v in ARM_TEX.items():
        text = text.replace(k + " ", v + " ").replace(k + ":", v + ":").replace(" " + k, " " + v)
    text = text.replace("R_ret", r"$R_{\mathrm{ret}}$")
    text = text.replace(" minus ", f" {M} ").replace(" minus 1", f" {M} 1")
    return text


def tab_all_contrasts():
    c = load("contrasts.csv")
    fam = family()
    g = gains_with_p()
    order = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"]
    head = (r"Task & Learner & Condition & Steps & $n$ & Estimate & 95\% CI & Win rate & $p$ & Holm & BH \\")
    out = [r"\begingroup\small", r"\setlength{\tabcolsep}{2.5pt}",
           r"\begin{longtable}{llllrrlrrrr}",
           r"\caption{Every contrast of the study. Estimates are paired differences in normalized AUC over seeds, "
           r"except for P5 and P8, which are rank correlations across cells with $n$ counting cells, P9, the median "
           r"retention ratio of each run minus 1, and P10, the rank correlation within each run. Intervals are 95\% "
           r"percentile bootstrap intervals over seeds, and the win rate is the share of seeds with a positive value. "
           r"For the 38 contrasts of the fixed plan (P1, P2, P3, P4, P6 and P7), $p$, Holm and BH come from the "
           r"exploratory multiple-comparison check. The other gain rows carry an unadjusted two-sided sign-flip "
           r"$p$.}\label{tab:allcontrasts} \\",
           r"\toprule", head, r"\midrule", r"\endfirsthead",
           r"\multicolumn{11}{l}{\footnotesize\textit{continued}} \\", r"\toprule", head, r"\midrule", r"\endhead",
           r"\bottomrule", r"\endfoot"]
    n_rows = 0
    first = True
    for pid in order:
        part = c[c.id == pid]
        for label in part.label.drop_duplicates():
            rows = part[part.label == label]
            q = rows.question.iloc[0]
            if not first:
                out.append(r"\addlinespace")
            first = False
            out.append(rf"\multicolumn{{11}}{{l}}{{\textit{{{pid} ({q}): {_label(label)}}}}} \\*")
            for r in rows.itertuples():
                hit = fam[(fam.id == r.id) & (fam.label == r.label) & (fam.task == r.task) & (fam.condition == r.condition)
                          & (fam.learner == r.learner)]
                p, holm, bh = ((float(hit.p.iloc[0]), float(hit.holm.iloc[0]), float(hit.bh.iloc[0])) if len(hit) == 1
                               else (np.nan, np.nan, np.nan))
                task = TASK_TEX.get(r.task, r.task)
                win = num(r.win_rate, 2) if r.kind == "paired" else "--"
                out.append(" & ".join([task, LEARNER_TEX.get(r.learner, r.learner), r.condition, steps_tex(r.budget),
                                       str(int(r.n)), num(r.estimate), ci(r.low, r.high), win, pval(p), pval(holm),
                                       pval(bh)]) + r" \\")
                n_rows += 1
    # gains.csv: every arm minus BASE
    gg = g.copy()
    gg["key"] = gg.label + "|" + gg.learner + "|" + gg.budget.astype(str)
    for key in gg.key.drop_duplicates():
        rows = gg[gg.key == key]
        r0 = rows.iloc[0]
        out.append(r"\addlinespace")
        out.append(rf"\multicolumn{{11}}{{l}}{{\textit{{Gain over BASE: {_label(r0.label)}, "
                   rf"{LEARNER_TEX[r0.learner]}, {steps_tex(r0.budget)}}}}} \\*")
        for r in rows.itertuples():
            out.append(" & ".join([TASK_TEX[r.task], LEARNER_TEX[r.learner], r.condition, steps_tex(r.budget),
                                   str(int(r.n)), num(r.estimate), ci(r.low, r.high), num(r.win_rate, 2), pval(r.p),
                                   pval(r.holm), pval(r.bh)]) + r" \\")
            n_rows += 1
    out += [r"\end{longtable}", r"\endgroup"]
    matched = sum(1 for line in out if line.count("&") == 10 and "--" not in line.split("&")[-1])
    write("tab_all_contrasts.tex",
          ""
          + "\n".join(out) + "\n")
    return {"rows": n_rows, "contrast_rows": len(c), "gain_rows": len(g), "rows_with_bh": matched}


# ---------------------------------------------------------------- tab_repetition

def tab_repetition():
    cells = load("cells.csv")
    rep = cells[cells.arm.isin(["IER", "IER3", "IER_REL", "IER3_REL"])].copy()
    rep["t"] = rep.task.map({t: i for i, t in enumerate(TASKS)})
    rep["c"] = rep.condition.map({k: i for i, k in enumerate(["default", "noise5", "noise10", "dyn20"])})
    rep["a"] = rep.arm.map({"IER": 0, "IER_REL": 1, "IER3": 2})
    rep["l"] = rep.learner.map({"sac": 0, "crossq": 1})
    rep = rep.sort_values(["l", "budget", "t", "c", "a"], ascending=[True, False, True, True, True])
    lines = [r"\begin{tabular}{llllrrrr}", r"\toprule",
             r"Learner, & & & & Replay & Triggers & Retention & Start \\",
             r"steps & Task & Condition & Arm & share (\%) & per run & after 200k & distance \\",
             r"\midrule"]
    prev = None
    info = []
    for r in rep.itertuples():
        group = (r.learner, r.budget)
        if prev is not None and group != prev:
            lines.append(r"\midrule")
        ret, n = retention_summary(r.task, r.condition, r.learner, r.arm, int(r.budget))
        head = f"{LEARNER_TEX[r.learner]}, {steps_tex(r.budget)}" if group != prev else ""
        prev = group
        lines.append(" & ".join([head, TASK_TEX[r.task], r.condition, ARM_TEX[r.arm], f"{r.repetition_share * 100:.1f}",
                                 f"{r.trigger_count:.1f}", num(ret, 2), num(r.start_distance_mean, 2)]) + r" \\")
        info.append({"cell": f"{r.learner} {steps_tex(r.budget)} {r.task} {r.condition} {r.arm}", "retention": ret,
                     "seeds_with_late_replays": n})
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_repetition.tex",
          ""
          + "\n".join(lines) + "\n")
    return {"cells": len(rep), "detail": info}


if __name__ == "__main__":
    print(json.dumps(tab_reproduction(), indent=1, default=str))
    v = tab_verdicts()
    print(json.dumps(v["footnotes"], indent=1))
    print(tab_all_contrasts())
    print(tab_repetition()["cells"])
