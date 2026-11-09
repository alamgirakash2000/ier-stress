"""The three tables of the main text that are laid out by hand.

tab_grid.tex: the experimental grid (fixed by the study design).
tab_main_reproduction.tex: comparisons (rows) by task (columns), SAC at 1M steps and CrossQ at 300k steps.
tab_main_stress.tex: BASE and IER3 AUC and the IER3 gain per condition (rows) by task (columns), 300k steps.
Bold marks a paired difference whose 95% bootstrap interval excludes zero.
"""

from __future__ import annotations

import pandas as pd

from .common import AGG, TAB_OUT

TASKS = ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"]
TASKS3 = ["HalfCheetah-v4", "walker-walk", "finger-turn_hard"]


def _num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace("-", "$-$")


def _pick(df: pd.DataFrame, **kw):
    m = df
    for k, v in kw.items():
        m = m[m[k] == v]
    return None if m.empty else next(m.itertuples())


def _diff(r) -> str:
    if r is None:
        return "--"
    est = _num(r.estimate)
    est = f"\\textbf{{{est}}}" if r.clear else est
    return f"{est} {{\\footnotesize[{_num(r.low)}, {_num(r.high)}]}}"


def _write(name: str, lines: list[str]) -> None:
    TAB_OUT.mkdir(parents=True, exist_ok=True)
    (TAB_OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def tab_grid() -> None:
    _write("tab_grid.tex", [
        r"\begin{tabular}{lllllrr}", r"\toprule",
        r"Experiment & Learner & Tasks & Arms & Conditions & Steps & Runs \\", r"\midrule",
        r"Reproduction & SAC & all four & BASE, IER & default & 1M & 160 \\",
        r"Released-code schedule & SAC & all four & IER-REL & default & 1M & 80 \\",
        r"Fixed repeat count & SAC & all four & IER3 & default & 1M & 80 \\",
        r"Stress & SAC & all but Hopper & BASE, IER3 & default, noise5, dyn20 & 300k & 360 \\",
        r"Heavy noise & SAC & all but Hopper & BASE, IER3 & noise10 & 300k & 120 \\",
        r"Copy control & SAC & all but Hopper & DUP & default, noise5 & 300k & 120 \\",
        r"Stronger learner & CrossQ & all four & BASE, IER & default & 300k & 160 \\",
        r"Stronger learner, stress & CrossQ & all but Hopper & BASE, IER3 & noise5 (both), default (IER3) & 300k & 180 \\",
        r"\midrule", r"Total & & & & & & 1260 \\", r"\bottomrule", r"\end{tabular}"])


def tab_main_reproduction() -> None:
    cells = pd.read_csv(AGG / "cells.csv")
    con = pd.read_csv(AGG / "contrasts.csv")
    gains = pd.read_csv(AGG / "gains.csv")

    def final(task, learner, arm, budget):
        r = _pick(cells, task=task, learner=learner, arm=arm, budget=budget, condition="default")
        return "--" if r is None else f"{r.final_return_raw_mean:.0f}"

    def row(label, values):
        return label + " & " + " & ".join(values) + r" \\"

    rows = [r"\multicolumn{5}{l}{\textit{SAC, 1M steps}} \\",
            row("Return, BASE", [final(t, "sac", "BASE", 1_000_000) for t in TASKS]),
            row("Return, IER", [final(t, "sac", "IER", 1_000_000) for t in TASKS]),
            r"\addlinespace[2pt]"]
    for label, arm in (("IER $-$ BASE", "IER"), ("IER-REL $-$ BASE", "IER_REL"), ("IER3 $-$ BASE", "IER3")):
        rows.append(row(label, [_diff(_pick(gains, task=t, learner="sac", arm=arm, budget=1_000_000,
                                            condition="default", label=f"{arm} minus BASE")) for t in TASKS]))
    rows.append(row("IER $-$ IER-REL", [_diff(_pick(con, id="P2", task=t)) for t in TASKS]))
    rows.append(row("IER3 $-$ IER", [_diff(_pick(con, id="P7", task=t)) for t in TASKS]))
    rows += [r"\midrule", r"\multicolumn{5}{l}{\textit{CrossQ, 300k steps}} \\",
             row("Return, BASE", [final(t, "crossq", "BASE", 300_000) for t in TASKS]),
             row("Return, IER", [final(t, "crossq", "IER", 300_000) for t in TASKS]),
             r"\addlinespace[2pt]",
             row("IER $-$ BASE", [_diff(_pick(con, id="P6", task=t, label="IER minus BASE, CrossQ")) for t in TASKS]),
             row("BASE, CrossQ $-$ SAC", [_diff(_pick(con, id="P11", task=t)) for t in TASKS])]
    _write("tab_main_reproduction.tex", [
        r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lcccc}", r"\toprule",
        r" & HalfCheetah & Hopper & walker-walk & finger-turn\_hard \\", r"\midrule"]
        + rows + [r"\bottomrule", r"\end{tabular}"])


def tab_main_stress() -> None:
    cells = pd.read_csv(AGG / "cells.csv")
    gains = pd.read_csv(AGG / "gains.csv")

    def auc(task, learner, arm, cond):
        r = _pick(cells, task=task, learner=learner, arm=arm, budget=300_000, condition=cond)
        return "--" if r is None else f"{r.auc_mean:.2f}"

    def gain(task, learner, cond):
        r = _pick(gains, task=task, learner=learner, arm="IER3", budget=300_000, condition=cond, label="IER3 minus BASE")
        if r is None:
            return "--"
        e = _num(r.estimate)
        return f"\\textbf{{{e}}}" if r.clear else e

    rows = []
    for learner, conds in (("SAC", ["default", "noise5", "noise10", "dyn20"]), ("CrossQ", ["default", "noise5"])):
        for c in conds:
            values = []
            for t in TASKS3:
                values += [auc(t, learner.lower(), "BASE", c), auc(t, learner.lower(), "IER3", c), gain(t, learner.lower(), c)]
            rows.append(f"{learner} & {c} & " + " & ".join(values) + r" \\")
        if learner == "SAC":
            rows.append(r"\midrule")
    _write("tab_main_stress.tex", [
        r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llccccccccc}", r"\toprule",
        r" & & \multicolumn{3}{c}{HalfCheetah} & \multicolumn{3}{c}{walker-walk} & \multicolumn{3}{c}{finger-turn\_hard} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Learner & Condition & BASE & IER3 & Gain & BASE & IER3 & Gain & BASE & IER3 & Gain \\", r"\midrule"]
        + rows + [r"\bottomrule", r"\end{tabular}"])


def build_all() -> None:
    tab_grid()
    tab_main_reproduction()
    tab_main_stress()
