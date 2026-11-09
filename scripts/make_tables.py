"""Writes the LaTeX tables (booktabs) to paper_assets/tables/, from results/aggregated/.

    python scripts/make_tables.py
    python scripts/make_tables.py --agg results/aggregated/pilot_check --out results/aggregated/pilot_check/tables

A table without data yet is skipped and listed as such.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from iers import ROOT  # noqa: E402
from iers.analysis import CONDITION_ORDER, LONG, SHORT, TASK_ORDER  # noqa: E402
from iers.specs import atomic_write_text  # noqa: E402

AGGREGATED = ROOT / "results" / "aggregated"
TABLES = ROOT / "paper_assets" / "tables"
TASK_LABEL = {"HalfCheetah-v4": "HalfCheetah", "Hopper-v4": "Hopper", "walker-walk": "walker-walk",
              "finger-turn_hard": "finger-turn\\_hard"}
LEARNER_LABEL = {"sac": "SAC", "crossq": "CrossQ"}


def tex(text) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def ci(mean, low, high, digits: int = 3) -> str:
    if mean is None or not np.isfinite(mean):
        return "n/a"
    return f"{num(mean, digits)} [{num(low, digits)}, {num(high, digits)}]"


def table(columns: list[str], rows: list[list[str]], caption: str, label: str, align: str | None = None) -> str:
    align = align or "l" * 2 + "r" * (len(columns) - 2)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(columns) + " \\\\", "\\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(lines)


def read_csv(agg: Path, name: str) -> pd.DataFrame | None:
    path = agg / name
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame if len(frame) else None


def task_rank(task: str) -> int:
    return TASK_ORDER.index(task) if task in TASK_ORDER else 99


# ---------------------------------------------------------------- the tables


def hyperparameters(agg: Path) -> str | None:
    path = agg / "configs.json"
    configs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not configs:
        return None
    rows = []
    for learner in ("sac", "crossq"):
        for entry in configs.get(learner, []):
            values = entry["config"]["learner"] | {f"train: {k}": v for k, v in entry["config"]["train"].items()}
            for key, value in values.items():
                rows.append([LEARNER_LABEL[learner], tex(key), tex(value), str(entry["runs"])])
        if len(configs.get(learner, [])) > 1:
            rows.append([LEARNER_LABEL[learner], "\\textit{more than one setting was used}", "", ""])
    return table(["Learner", "Setting", "Value", "Runs"], rows,
                 "Settings of every main run, read from the run records. Evaluation steps do not count toward the "
                 "budget. Repeated steps do count.", "tab:hyperparameters", "llrr")


def reproduction(agg: Path) -> str | None:
    cells, contrasts = read_csv(agg, "cells.csv"), read_csv(agg, "contrasts.csv")
    if cells is None or contrasts is None:
        return None
    a1 = cells[(cells.learner == "sac") & (cells.condition == "default") & (cells.budget == LONG)]
    p1 = contrasts[contrasts.id == "P1"].set_index("task")
    if not len(p1):
        return None
    rows = []
    for task in sorted(p1.index, key=task_rank):
        base = a1[(a1.task == task) & (a1.arm == "BASE")].iloc[0]
        ier = a1[(a1.task == task) & (a1.arm == "IER")].iloc[0]
        c = p1.loc[task]
        rows.append([TASK_LABEL.get(task, tex(task)), num(base.final_return_raw_mean, 0), num(ier.final_return_raw_mean, 0),
                     ci(c.estimate, c.low, c.high), "yes" if c.clear else "no", num(c.win_rate, 2), str(int(c.n)),
                     num(ier.get("repetition_share", np.nan), 3)])
    return table(["Task", "BASE final", "IER final", "IER gain, AUC [95\\% CI]", "Clear", "Win rate", "Seeds",
                  "Repetition share"], rows,
                 "Reproduction with SAC under the default condition over 1M environment steps. Final return is the "
                 "mean of the last 5 evaluations. The gain is the mean per-seed difference in normalized AUC with a "
                 "percentile bootstrap interval. An effect is clear when its interval excludes zero.",
                 "tab:reproduction", "lrrrrrrr")


def stress(agg: Path) -> str | None:
    gains = read_csv(agg, "gains.csv")
    if gains is None:
        return None
    chosen = gains[(gains.arm == "IER3") & (gains.budget == SHORT)]
    if not len(chosen):
        return None
    rows = []
    chosen = chosen.assign(t=chosen.task.map(task_rank), c=chosen.condition.map(
        lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 9)).sort_values(["learner", "t", "c"])
    for r in chosen.itertuples():
        rows.append([LEARNER_LABEL.get(r.learner, tex(r.learner)), TASK_LABEL.get(r.task, tex(r.task)), tex(r.condition),
                     num(r.start_distance_mean, 2), ci(r.estimate, r.low, r.high), "yes" if r.clear else "no",
                     num(r.win_rate, 2), str(int(r.n))])
    return table(["Learner", "Task", "Condition", "Start distance", "IER3 gain [95\\% CI]", "Clear", "Win rate", "Seeds"],
                 rows,
                 "Gain of IER with RN = 3 over the same learner without it, in normalized AUC over 300k environment "
                 "steps. Start distance is the mean distance between a repeat's start state and the stored "
                 "episode's start state.", "tab:stress", "lllrrrrr")


def mechanism(agg: Path) -> str | None:
    contrasts, gains = read_csv(agg, "contrasts.csv"), read_csv(agg, "gains.csv")
    if contrasts is None or gains is None:
        return None
    p3 = contrasts[contrasts.id == "P3"]
    if not len(p3):
        return None
    rows = []
    for r in p3.assign(t=p3.task.map(task_rank)).sort_values(["t", "condition"]).itertuples():
        def gain(arm):
            g = gains[(gains.task == r.task) & (gains.condition == r.condition) & (gains.learner == "sac")
                      & (gains.arm == arm) & (gains.budget == SHORT)]
            return ci(g.estimate.iloc[0], g.low.iloc[0], g.high.iloc[0]) if len(g) else "n/a"

        rows.append([TASK_LABEL.get(r.task, tex(r.task)), tex(r.condition), gain("IER3"), gain("DUP"),
                     ci(r.estimate, r.low, r.high), "yes" if r.clear else "no", str(int(r.n))])
    return table(["Task", "Condition", "IER3 gain", "DUP gain", "IER3 minus DUP", "Clear", "Seeds"], rows,
                 "Mechanism with SAC over 300k environment steps. DUP adds RN copies of the stored episode to the "
                 "replay buffer with no extra environment steps. All values are normalized AUC with 95 percent "
                 "percentile bootstrap intervals over seeds.", "tab:mechanism", "llrrrrr")


def gates(agg: Path) -> str | None:
    rows = []
    # Every G1 record, so a failed pair is never dropped: gate_g1.json, or the per-learner files and the SAC debug.
    g1_paths = [agg / "gate_g1.json"] if (agg / "gate_g1.json").exists() else sorted(agg.glob("gate_g1_*.json"))
    for g1_path in g1_paths:
        g1 = json.loads(g1_path.read_text(encoding="utf-8"))
        pairs = {"debug": g1["pair"]} if "pair" in g1 else g1.get("pairs", {})
        for learner, v in pairs.items():
            check = ("SAC against SB3 SAC with the upstream log std (debug)" if learner == "debug"
                     else f"{LEARNER_LABEL.get(learner, learner)} against the reference")
            rows.append(["G1", check,
                         f"{num(v['ours_mean'], 0)} against {num(v['reference_mean'], 0)}, "
                         f"difference {num(100 * v['relative_difference'], 1)} percent",
                         "pass" if v["passed"] else "fail"])
    g2_path = agg / "gate_g2.json"
    if g2_path.exists():
        g2 = json.loads(g2_path.read_text(encoding="utf-8"))
        for task, by_name in g2["contrasts"].items():
            for name, r in by_name.items():
                rows.append(["G2", f"{name}, {TASK_LABEL.get(task, tex(task))}", ci(r["mean"], r["low"], r["high"]),
                             "clear" if r["clear"] else "not clear"])
        rows.append(["G2", "decision", tex(g2["decision"]["reason"]), g2["decision"]["verdict"]])
    if not rows:
        return None
    return table(["Gate", "Check", "Value", "Result"], rows,
                 "Implementation checks on pilot seeds, which do not enter any other result. G1 compares mean final return "
                 "over 3 seeds at 300k steps on HalfCheetah. G2 gives paired contrasts in normalized AUC over 5 "
                 "seeds at 300k steps.", "tab:gates", "llp{7cm}l")


TABLES_TO_MAKE = {"hyperparameters": hyperparameters, "reproduction": reproduction, "stress": stress,
                  "mechanism": mechanism, "gates": gates}


def build(agg: Path, out: Path) -> dict[str, list[str]]:
    made, skipped = [], []
    for name, builder in TABLES_TO_MAKE.items():
        text = builder(agg)
        if text is None:
            skipped.append(name)
            continue
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out / f"{name}.tex", text)
        made.append(name)
    return {"made": made, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agg", type=Path, default=AGGREGATED)
    parser.add_argument("--out", type=Path, default=TABLES)
    args = parser.parse_args()
    result = build(args.agg, args.out)
    print(f"made {len(result['made'])} tables: {', '.join(result['made']) or 'none'}")
    if result["skipped"]:
        print(f"no data yet for: {', '.join(result['skipped'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
