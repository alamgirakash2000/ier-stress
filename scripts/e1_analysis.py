"""E1, IER3_RANDOM, an exploratory control. Written before any E1 run existed.

    python scripts/e1_analysis.py
    python scripts/e1_analysis.py --raw check/raw --out check/aggregated/e1.json   E1 retrained into check/raw,
                                                                      compared with the saved B1 and C1 runs

Compares E1 (IER3_RANDOM, walker-walk, default, 300k, seeds 0 to 19) with B1's IER3 and BASE and C1's DUP (RN = 3)
on the same task, condition, budget and seeds, paired by seed. The metric and statistics are those of the main
analysis: normalized AUC with the main normalizer (mean final return of BASE SAC, walker-walk, default, 300k, from B1), and the
paired percentile bootstrap of iers/stats.py (10,000 resamples, 95 percent CI). Also the median retention ratio
minus 1 (as P9), here measured against each drawn episode's own return.

Stated before the run: a CI of IER3_RANDOM minus IER3 below zero means the gain needs the best episode. A CI
around zero means re-executing any past episode does about as well. Writes results/aggregated/e1.json. Runs only
once logs/UNBLIND_OK.md names E1.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from iers import ROOT  # noqa: E402
from iers.analysis import Cell, Store, load_runs, paired, read_repeats, retention_rows  # noqa: E402
from iers.specs import RESULTS_RAW, atomic_write_text, load_spreads  # noqa: E402
from iers.stats import paired_bootstrap  # noqa: E402

TASK, CONDITION, BUDGET = "walker-walk", "default", 300_000
OUT = ROOT / "results" / "aggregated" / "e1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RESULTS_RAW, help="folder holding the E1 runs")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    unblind = ROOT / "logs" / "UNBLIND_OK.md"
    if not unblind.exists() or "E1" not in re.findall(r"\b[A-Z][0-9]\b", unblind.read_text(encoding="utf-8")):
        print("logs/UNBLIND_OK.md does not name E1. E1 stays blind.")
        return 2
    spreads = load_spreads()
    # B1 and C1 always come from the saved results. E1 comes from --raw, so a retrained E1 can be checked alone.
    runs, curves, repeats = load_runs(RESULTS_RAW, spreads, experiments=["B1", "C1"])
    e1_runs, e1_curves, e1_repeats = load_runs(args.raw, spreads, experiments=["E1"])
    runs = pd.concat([runs, e1_runs], ignore_index=True)
    curves.update(e1_curves)
    repeats = pd.concat([f for f in (repeats, e1_repeats) if len(f)], ignore_index=True) if len(e1_repeats) else repeats
    runs = runs[(runs.task == TASK) & (runs.condition == CONDITION) & (runs.budget == BUDGET)]
    # The main analysis reads repeats only for its own arms, so E1's are read here with the same helper.
    extra = [read_repeats(args.raw / row["run_id"], row, spreads)
             for row in runs[(runs.arm == "IER3_RANDOM") & (runs.status == "finished")].to_dict("records")]
    repeats = pd.concat([repeats, *extra], ignore_index=True) if extra else repeats
    store = Store(runs, curves, repeats)
    cells = {arm: Cell(TASK, CONDITION, "sac", arm, BUDGET) for arm in ("IER3_RANDOM", "IER3", "BASE", "DUP")}
    scores = {arm: store.scores(cell) for arm, cell in cells.items()}
    result = {"task": TASK, "condition": CONDITION, "budget": BUDGET, "normalizer": store.normalizer(TASK, BUDGET),
              "runs": {arm: int(len(store.finished(cell))) for arm, cell in cells.items()},
              "excluded": {arm: store.excluded(cell) for arm, cell in cells.items()},
              "mean_normalized_auc": {arm: float(s.mean()) for arm, s in scores.items()}, "contrasts": {}}
    for other in ("IER3", "BASE", "DUP"):
        diffs, missing = paired(scores["IER3_RANDOM"], scores[other])
        r = paired_bootstrap(diffs.to_numpy())
        result["contrasts"][f"IER3_RANDOM minus {other}"] = {
            "n": r.n, "estimate": r.mean, "low": r.low, "high": r.high, "win_rate": r.win_rate, "clear": r.clear,
            "missing_seeds": missing}
    for row in retention_rows(store, cells["IER3_RANDOM"]) + retention_rows(store, cells["IER3"]):
        result.setdefault("retention", []).append({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                                   for k, v in vars(row).items() if k != "extra"})
    atomic_write_text(args.out, json.dumps(result, indent=1, default=str))
    for name, c in result["contrasts"].items():
        print(f"{name}: {c['estimate']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}], n={c['n']}, win rate {c['win_rate']:.2f}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
