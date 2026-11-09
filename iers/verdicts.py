"""The verdict rules of the analysis plan, applied to the contrast table. Torch-free.

Each function returns a Verdict: the verdict, the rule that gave it, and the contrast rows behind it.
A question whose contrasts were not run, for example because an item was cut, gets "not available".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

HOLDS, NOT_HOLD, MIXED, NA = "holds", "does not hold", "mixed", "not available"


@dataclass
class Verdict:
    question: str
    verdict: str
    reason: str
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def signs(rows: pd.DataFrame) -> tuple[int, int, int]:
    """Counts of clear positive, clear negative and all cells. Clear means the CI excludes zero."""
    usable = rows[rows["n"].fillna(0) > 0]
    positive = int((usable["low"] > 0).sum())
    negative = int((usable["high"] < 0).sum())
    return positive, negative, len(usable)


def pick(table: pd.DataFrame, pid: str, label_contains: str | None = None) -> pd.DataFrame:
    rows = table[table["id"] == pid]
    if label_contains is not None:
        rows = rows[rows["label"].str.contains(label_contains, regex=False)]
    return rows


def q1_gain(table: pd.DataFrame, rliable: dict) -> Verdict:
    rows = pick(table, "P1")
    poi = (rliable.get("sac_1000k") or {}).get("probability_of_improvement")
    if not len(rows) or poi is None:
        return Verdict("Q1 gain", NA, "A1 has not produced P1 and the probability of improvement.", rows)
    pos, neg, _ = signs(rows)
    text = f"probability of improvement {poi['estimate']:.3f} [{poi['low']:.3f}, {poi['high']:.3f}], " \
           f"{pos} task(s) clear positive, {neg} clear negative"
    if poi["low"] > 0.5 and neg == 0:
        return Verdict("Q1 gain", HOLDS, text, rows)
    if poi["high"] < 0.5 or (poi["low"] <= 0.5 <= poi["high"] and pos == 0):
        return Verdict("Q1 gain", NOT_HOLD, text, rows)
    return Verdict("Q1 gain", MIXED, text, rows)


def q1_schedule(table: pd.DataFrame) -> Verdict:
    rows = pick(table, "P2")
    if not len(rows):
        return Verdict("Q1 schedule", NA, "R1 has not produced P2.", rows)
    pos, neg, cells = signs(rows)
    if pos or neg:
        return Verdict("Q1 schedule", "the gain depends on the schedule",
                       f"{pos} of {cells} task(s) clear positive and {neg} clear negative for IER minus IER_REL", rows)
    return Verdict("Q1 schedule", "no clear dependence on the schedule", f"no task of {cells} is clear", rows)


def q2(table: pd.DataFrame) -> Verdict:
    rows = pick(table, "P3")
    if not len(rows):
        return Verdict("Q2", NA, "C1 has not produced P3.", rows)
    pos, neg, cells = signs(rows)
    # The practical question at equal environment steps and updates. Why one does better is E1's question, which
    # is exploratory and outside these rules.
    text = f"IER3 minus DUP: {pos} of {cells} cell(s) clear positive, {neg} clear negative"
    if pos and not neg:
        return Verdict("Q2", HOLDS + ": re-executing does better than duplication at equal budgets", text, rows)
    if not pos and not neg:
        return Verdict("Q2", NOT_HOLD + ": no clear difference between re-executing and duplication", text, rows)
    return Verdict("Q2", MIXED, text, rows)


def q3(table: pd.DataFrame) -> Verdict:
    p9, p10 = pick(table, "P9"), pick(table, "P10")
    if not len(p9):
        return Verdict("Q3", NA, "No IER-type run has produced repeats.", p9)
    _, neg9, cells9 = signs(p9)
    pos10, neg10, cells10 = signs(p10)
    text = (f"P9 clear negative in {neg9} of {cells9} cell(s), P10 clear negative in {neg10} and clear positive in "
            f"{pos10} of {cells10}")
    rows = pd.concat([p9, p10])
    if neg9 > cells9 / 2 and neg10 > 0 and pos10 == 0:
        return Verdict("Q3", HOLDS, text, rows)
    if neg9 == 0:
        return Verdict("Q3", NOT_HOLD, text, rows)
    return Verdict("Q3", MIXED, text, rows)


def q4(table: pd.DataFrame) -> Verdict:
    p4 = pick(table, "P4")
    p5 = pick(table, "P5", "all tasks")
    p5 = p5[p5["learner"] == "sac"]
    if not len(p4) and not len(p5):
        return Verdict("Q4", NA, "B1 has not produced P4 or P5.", p4)
    pos4, neg4, cells4 = signs(p4)
    p5_below = bool(len(p5)) and bool(p5["high"].iloc[0] < 0)
    p5_text = (f"SAC Spearman {p5['estimate'].iloc[0]:.3f} [{p5['low'].iloc[0]:.3f}, {p5['high'].iloc[0]:.3f}]"
               if len(p5) else "no P5")
    text = f"{p5_text}, P4 clear negative in {neg4} and clear positive in {pos4} of {cells4} cell(s)"
    rows = pd.concat([p4, p5])
    if p5_below or (neg4 > 0 and pos4 == 0):
        return Verdict("Q4", HOLDS + ": the gain shrinks", text, rows)
    if neg4 == 0 and not p5_below:
        return Verdict("Q4", NOT_HOLD, text, rows)
    return Verdict("Q4", MIXED, text, rows)


def q5(table: pd.DataFrame) -> Verdict:
    rows = pick(table, "P6", "CrossQ")
    rows = rows[rows["label"] == "IER minus BASE, CrossQ"]
    if not len(rows):
        return Verdict("Q5", NA, "A2 has not produced P6.", rows)
    pos, neg, cells = signs(rows)
    text = f"CrossQ gain clear positive in {pos} and clear negative in {neg} of {cells} task(s)"
    if pos and not neg:
        return Verdict("Q5", HOLDS + ": the gain survives", text, rows)
    if not pos:
        return Verdict("Q5", NOT_HOLD, text, rows)
    return Verdict("Q5", MIXED, text, rows)


def q6(table: pd.DataFrame) -> Verdict:
    rows = pick(table, "P7")
    if not len(rows):
        return Verdict("Q6", NA, "D1 has not produced P7.", rows)
    pos, neg, cells = signs(rows)
    text = f"IER3 minus IER clear negative in {neg} and clear positive in {pos} of {cells} task(s)"
    if not neg:
        return Verdict("Q6", HOLDS + ": a fixed RN = 3 transfers across tasks", text, rows)
    if not pos:
        return Verdict("Q6", NOT_HOLD + ": the gain needs the per-task RN", text, rows)
    return Verdict("Q6", MIXED, text, rows)


def q7(table: pd.DataFrame) -> Verdict:
    rows = pick(table, "P8")
    if not len(rows) or not pd.notna(rows["low"].iloc[0]):
        return Verdict("Q7", NA, "Fewer than 3 cells with an IER-type arm and BASE.", rows)
    r = rows.iloc[0]
    # Secondary and descriptive: an association across this study's cells, never a predictor for new tasks.
    text = f"Spearman {r['estimate']:.3f} [{r['low']:.3f}, {r['high']:.3f}] over {int(r['n'])} cells"
    verdict = HOLDS + ": a positive association across these cells" if r["low"] > 0 else NOT_HOLD
    return Verdict("Q7", verdict, text, rows)


def all_verdicts(table: pd.DataFrame, rliable: dict) -> list[Verdict]:
    return [q1_gain(table, rliable), q1_schedule(table), q2(table), q3(table), q4(table), q5(table), q6(table),
            q7(table)]
