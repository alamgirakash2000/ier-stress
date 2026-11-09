"""Writes RESULTS.md from results/aggregated/, with a verdict per question by the fixed rules of iers/verdicts.py.

    python scripts/make_results.py
    python scripts/make_results.py --agg results/aggregated/pilot_check --out results/aggregated/pilot_check/RESULTS.md

Every number comes from the files it names.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from iers import ROOT  # noqa: E402
from iers.specs import atomic_write_text  # noqa: E402
from iers.verdicts import all_verdicts  # noqa: E402

AGGREGATED = ROOT / "results" / "aggregated"
OUT = ROOT / "RESULTS.md"
QUESTIONS = {
    "Q1 gain": ("Q1 Reproduction", "Do the IER-SAC gains hold with more seeds and rliable statistics?", "P1"),
    "Q1 schedule": ("Q1 Reproduction, schedule", "Does the gain depend on the schedule, Algorithm 1 or the released code?",
                    "P2"),
    "Q2": ("Q2 Mechanism", "Does re-executing the actions matter, or does re-adding the best episode to the buffer "
           "give the same gain?", "P3"),
    "Q3": ("Q3 Selection bias", "Do replays keep less than the stored return, and less as start distance grows?",
           "P9, P10"),
    "Q4": ("Q4 Stress", "Does the gain shrink as starts spread out, or under per-episode dynamics changes?", "P4, P5"),
    "Q5": ("Q5 Stronger learner", "Does the gain survive with CrossQ?", "P6"),
    "Q6": ("Q6 Tuning fairness", "Does the gain survive a fixed RN = 3?", "P7"),
    "Q7": ("Q7 Early diagnostic", "Does R_ret in the first 50k steps predict the final gain?", "P8"),
}


def num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def rows_table(rows: pd.DataFrame) -> list[str]:
    if not len(rows):
        return ["No contrast rows."]
    lines = ["| Contrast | What | Task | Learner | Condition | Budget | Estimate [95% CI] | Win rate | Seeds or cells | "
             "Missing seeds | Runs left out |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows.itertuples():
        budget = "" if not r.budget else f"{int(r.budget) // 1000}k"
        lines.append(f"| {r.id} | {r.label} | {r.task} | {r.learner} | {r.condition} | {budget} | "
                     f"{num(r.estimate)} [{num(r.low)}, {num(r.high)}] | {num(r.win_rate, 2)} | {int(r.n)} | "
                     f"{'' if pd.isna(r.missing_seeds) else r.missing_seeds} | "
                     f"{'' if pd.isna(r.excluded_runs) else r.excluded_runs} |")
    return lines


def prereg_lines() -> list[str]:
    config = yaml.safe_load((ROOT / "configs" / "prereg.yaml").read_text(encoding="utf-8")) \
        if (ROOT / "configs" / "prereg.yaml").exists() else None
    if config is None:
        return ["The analysis plan has not been fixed."]
    return [f"The analysis plan was fixed and hashed before the main runs (SHA256 "
            f"`{config['prereg_sha256']}`, fixed on {config['frozen_at']}), with {config['main_seeds']} main seeds."]


def limitation_lines(agg: Path) -> list[str]:
    lines = ["- CrossQ uses a critic width of 1024 in place of the paper's 2048, and A2 runs for 300k steps "
             "to fit the compute budget. Results for CrossQ say nothing about width 2048 or 1M steps."]
    g1 = read_json(agg / "gate_g1_sac.json")
    if g1:
        v = g1["pairs"]["sac"]
        lines.append(f"- Our SAC is the upstream IER SAC. On HalfCheetah at 300k steps it reached {num(v['ours_mean'], 1)} "
                     f"against {num(v['reference_mean'], 1)} for stable-baselines3 SAC, which failed the 20 percent rule "
                     f"of the implementation check (`results/aggregated/gate_g1_sac.json`).")
    debug = read_json(agg / "gate_g1_sac_debug.json")
    if debug:
        v = debug["pair"]
        lines.append(f"- With the upstream log std, stable-baselines3 SAC reached {num(v['reference_mean'], 1)}, "
                     f"{'within' if v['passed'] else 'outside'} the rule of our {num(v['ours_mean'], 1)} "
                     f"(`results/aggregated/gate_g1_sac_debug.json`).")
    lines += ["- Main results stayed blind until each item's main runs had finished.",
              "- Main runs ran on RTX 3090 GPUs, and all arms compared within a task on the same GPU type. CrossQ "
              "learned faster in its first 50k steps on these machines than on the pilot machine, with identical code "
              "and seeds, so pilot and main CrossQ numbers are not compared.",
              "- With 20 seeds per cell, some CIs stay wide, and a verdict of \"does not hold\" can mean too little "
              "data."]
    return lines


def compute_lines(summary: dict) -> list[str]:
    compute, resources = summary.get("compute", {}), summary.get("resources", {})
    lines = [f"- Runs: {compute.get('runs', 0)}, finished {compute.get('finished', 0)}, not finished "
             f"{compute.get('not_finished', 0)} (`results/aggregated/excluded.csv`).",
             f"- Wall time of all groups: {num(compute.get('group_wall_hours'), 1)} hours, of which the resource guard "
             f"held training back for {num(compute.get('guard_pause_hours'), 1)} hours.",
             f"- Code hashes: {', '.join('`' + h[:12] + '`' for h in compute.get('code_hashes', [])) or 'none'}, "
             "each with a per-file manifest in `results/code_manifests/`."]
    for grid, r in sorted(resources.items()):
        lines.append(f"- {grid}: GPU {num(r['gpu_util_mean_pct'], 1)} percent and CPU {num(r['cpu_util_mean_pct'], 1)} "
                     f"percent on average over {r['samples']} samples, peak GPU memory {num(r['gpu_mem_peak_mb'], 0)} MB.")
    return lines


def build(agg: Path) -> str:
    contrasts = pd.read_csv(agg / "contrasts.csv", dtype={"missing_seeds": str, "excluded_runs": str, "seeds": str}) if (agg / "contrasts.csv").exists() else pd.DataFrame(
        columns=["id", "label", "task", "learner", "condition", "budget", "n", "estimate", "low", "high", "win_rate",
                 "missing_seeds", "excluded_runs"])
    rliable = read_json(agg / "rliable.json") or {}
    summary = read_json(agg / "summary.json") or {}
    verdicts = all_verdicts(contrasts, rliable)
    lines = ["# Results", "",
             f"Generated by `scripts/make_results.py` on {time.strftime('%Y-%m-%d %H:%M')} from `results/aggregated/`. "
             "Main seeds only. Verdicts follow the fixed rules of `iers/verdicts.py`.",
             "", *prereg_lines(), "", "## Verdicts", "", "| Question | Verdict | Basis |", "|---|---|---|"]
    lines += [f"| {QUESTIONS[v.question][0]} | {v.verdict} | {v.reason} |" for v in verdicts]
    for v in verdicts:
        title, question, ids = QUESTIONS[v.question]
        lines += ["", f"## {title}", "", f"*{question}*", "", f"**Verdict: {v.verdict}.** {v.reason[:1].upper() + v.reason[1:]}.", "",
                  f"Source: `results/aggregated/contrasts.csv`, rows {ids}"
                  + (", and `results/aggregated/rliable.json`" if v.question == "Q1 gain" else "") + ".", ""]
        lines += rows_table(v.rows)
    if "sac_1000k" in rliable:
        r = rliable["sac_1000k"]
        lines += ["", "**A1 aggregates across tasks** (`results/aggregated/rliable.json`):", ""]
        lines += [f"- IQM of normalized AUC, {arm}: {num(v['estimate'])} [{num(v['low'])}, {num(v['high'])}]"
                  for arm, v in r["iqm"].items()]
    lines += ["", "## Exploratory findings", "",
              "The replay check (`results/aggregated/replay_check.csv`) and the random-episode replay control E1 "
              "(`results/aggregated/e1.json`, `scripts/e1_analysis.py`) were added after unblinding.",
              "", "## Limitations", "", *limitation_lines(agg), "", "## Compute and resource use", "",
              *compute_lines(summary), ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agg", type=Path, default=AGGREGATED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    atomic_write_text(args.out, build(args.agg))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
