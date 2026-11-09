"""Builds results/aggregated/ and paper_assets/numbers.json from the raw run folders.

    python scripts/aggregate.py                                   main seeds (0 to 99), every unblinded experiment
    python scripts/aggregate.py --pilot --experiments g2 --out results/aggregated/pilot_check
                                                                  pilot seeds, only to check this code

Every number in the paper comes from the files this script writes. Pilot seeds never enter the main
output, and the pilot mode refuses to write into paper_assets/.
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
from iers.analysis import (  # noqa: E402
    MAIN_EXPERIMENTS,
    REPEAT_ARMS,
    Cell,
    Store,
    condition_key,
    contrasts,
    curves_table,
    gain_table,
    load_runs,
    performance_profile,
    reference_curves,
    rliable_summary,
    run_configs,
    task_sort_key,
    wall_clock_table,
)
from iers.specs import RESULTS_RAW, atomic_write_text, load_spreads  # noqa: E402
from iers.stats import paired_bootstrap  # noqa: E402

AGGREGATED = ROOT / "results" / "aggregated"
NUMBERS = ROOT / "paper_assets" / "numbers.json"
# Main results stay blind until this file names the item, for example "A1 B1", or "all". It is written once an
# item's main runs have finished.
UNBLIND = ROOT / "logs" / "UNBLIND_OK.md"


def unblinded_items() -> list[str]:
    if not UNBLIND.exists():
        return []
    text = UNBLIND.read_text(encoding="utf-8")
    if re.search(r"\ball\b", text, flags=re.IGNORECASE):
        return list(MAIN_EXPERIMENTS)
    words = set(re.findall(r"\b[A-Z][0-9]\b", text))
    return [item for item in MAIN_EXPERIMENTS if item in words]


def blind_check(wanted: list[str] | None = None) -> str | None:
    """Main results stay unseen until logs/UNBLIND_OK.md names their item. Returns why aggregation must wait, or None."""
    if not UNBLIND.exists():
        return ("logs/UNBLIND_OK.md does not exist. Main seeds stay blind until it names the items whose main runs "
                "have finished, for example 'A1 B1', or 'all'.")
    allowed = unblinded_items()
    if not allowed:
        return f"logs/UNBLIND_OK.md names no main item ({' '.join(MAIN_EXPERIMENTS)}) and does not say 'all'."
    blocked = sorted(set(wanted or []) - set(allowed))
    if blocked:
        return f"logs/UNBLIND_OK.md does not name {' '.join(blocked)}. Those items stay blind."
    return None


def cell_summary(store: Store) -> pd.DataFrame:
    """One row per (task, condition, learner, arm, budget): seeds, AUC and final return with CIs, IER quantities."""
    runs = store.runs
    rows = []
    for key, group in runs.groupby(["task", "condition", "learner", "arm", "budget"], sort=False):
        task, condition, learner, arm, budget = key
        cell = Cell(task, condition, learner, arm, int(budget))
        done = group[group.status == "finished"]
        row = {"task": task, "condition": condition, "learner": learner, "arm": arm, "budget": int(budget),
               "runs": len(group), "finished": len(done), "not_finished": len(group) - len(done),
               "experiments": " ".join(sorted(group.experiment.unique())),
               "normalizer": store.normalizer(task, int(budget))}
        for metric in ("auc", "final"):
            values = store.scores(cell, metric)
            values = values[np.isfinite(values)]
            if len(values):
                result = paired_bootstrap(values.to_numpy())
                row.update({f"{metric}_mean": result.mean, f"{metric}_low": result.low, f"{metric}_high": result.high})
        row["final_return_raw_mean"] = float(done.final_return.mean()) if len(done) else np.nan
        if arm != "BASE":
            for column in ("trigger_count", "repetition_share", "r_ret", "r_ret_all", "start_distance_mean"):
                row[f"{column}"] = float(done[column].mean()) if len(done) else np.nan
        row["wall_time_s_mean"] = float(done.wall_time_s.mean()) if len(done) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["learner", "budget", "task", "condition", "arm"],
                                  key=lambda c: c.map(task_sort_key) if c.name == "task"
                                  else c.map(condition_key) if c.name == "condition" else c)
    return table.reset_index(drop=True)


def resource_summary(logs: Path) -> dict:
    """Mean GPU and CPU use over every logged grid."""
    out = {}
    for path in sorted(logs.glob("resources_*.csv")):
        table = pd.read_csv(path)
        if not len(table):
            continue
        out[path.stem.removeprefix("resources_")] = {
            "samples": int(len(table)),
            "gpu_util_mean_pct": float(table.gpu_util_pct.mean()),
            "gpu_mem_peak_mb": float(table.gpu_mem_used_mb.max()),
            "cpu_util_mean_pct": float(table.cpu_util_pct.mean()),
            "ram_available_min_mb": float(table.ram_available_mb.min()),
        }
    return out


def resource_series(logs: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(logs.glob("resources_*.csv")):
        table = pd.read_csv(path)
        if len(table):
            table.insert(0, "grid", path.stem.removeprefix("resources_"))
            parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["grid", "time"])


def compute_summary(runs: pd.DataFrame) -> dict:
    """Wall time counted once per group, since the seeds of a group share one process."""
    done = runs.dropna(subset=["wall_time_s"])
    groups = done.drop_duplicates(subset=["experiment", "group_seeds", "task", "learner", "wall_time_s"])
    return {
        "runs": int(len(runs)),
        "finished": int((runs.status == "finished").sum()),
        "not_finished": int((runs.status != "finished").sum()),
        "group_wall_hours": float(groups.wall_time_s.sum() / 3600),
        "guard_pause_hours": float(groups.guard_paused_s.fillna(0).sum() / 3600),
        "code_hashes": sorted(h for h in runs.code_hash.dropna().unique() if h),
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def aggregate(raw: Path, out: Path, numbers: Path | None, pilot: bool, experiments: list[str] | None,
              logs: Path, reps: int) -> dict:
    runs, curves, repeats = load_runs(raw, load_spreads(), pilot=pilot, experiments=experiments)
    store = Store(runs, curves, repeats)
    out.mkdir(parents=True, exist_ok=True)
    table = contrasts(store)
    cells = cell_summary(store)
    rliable = {}
    profiles = {}
    for learner in ("sac", "crossq"):
        budgets = sorted(runs[(runs.learner == learner) & (runs.arm == "IER")].budget.unique()) if len(runs) else []
        for budget in budgets:
            key = f"{learner}_{int(budget) // 1000}k"
            summary = rliable_summary(store, learner, int(budget), reps=reps)
            if summary:
                rliable[key] = summary
            profile = performance_profile(store, learner, int(budget), reps=reps)
            if profile:
                profiles[key] = profile
    excluded = runs[runs.status != "finished"][["experiment", "run_id", "status", "reason"]]

    write_csv(runs, out / "runs.csv")
    write_csv(curves_table(runs, curves), out / "curves.csv")
    write_csv(repeats, out / "repeats.csv")
    write_csv(cells, out / "cells.csv")
    write_csv(table, out / "contrasts.csv")
    write_csv(excluded, out / "excluded.csv")
    write_csv(gain_table(store), out / "gains.csv")
    write_csv(wall_clock_table(runs), out / "wall_clock.csv")
    write_csv(reference_curves(raw), out / "gate_curves.csv")
    write_csv(resource_series(logs), out / "resources.csv")
    atomic_write_text(out / "normalizers.json", json.dumps(jsonable(list(store.normalizers.values())), indent=1))
    atomic_write_text(out / "rliable.json", json.dumps(jsonable(rliable), indent=1))
    atomic_write_text(out / "profiles.json", json.dumps(jsonable(profiles), indent=1))
    atomic_write_text(out / "configs.json", json.dumps(jsonable(run_configs(raw, pilot, experiments)), indent=1))

    result = {
        "pilot": pilot,
        "experiments": experiments if experiments is not None else sorted(runs.experiment.unique()),
        "compute": compute_summary(runs) if len(runs) else {},
        "resources": resource_summary(logs),
        "normalizers": list(store.normalizers.values()),
        "rliable": {k: {kk: vv for kk, vv in v.items() if kk != "left_out"} for k, v in rliable.items()},
        "contrasts": table.to_dict(orient="records"),
        "repetition": cells[cells.arm.isin(REPEAT_ARMS)].to_dict(orient="records") if len(cells) else [],
        "excluded": excluded.to_dict(orient="records"),
    }
    atomic_write_text(out / "summary.json", json.dumps(jsonable(result), indent=1))
    if numbers is not None:
        numbers.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(numbers, json.dumps(jsonable(result), indent=1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RESULTS_RAW)
    parser.add_argument("--out", type=Path, default=AGGREGATED)
    parser.add_argument("--pilot", action="store_true", help="pilot seeds only, to check this code")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--reps", type=int, default=10_000, help="bootstrap resamples for rliable")
    args = parser.parse_args()
    if args.pilot and args.out.resolve() == AGGREGATED.resolve():
        parser.error("pilot output must go to its own folder, given with --out")
    if not args.pilot and args.raw.resolve() == RESULTS_RAW.resolve():
        reason = blind_check(args.experiments)
        if reason:
            print(reason)
            return 2
        if args.experiments is None:
            args.experiments = unblinded_items()
    # paper_assets/numbers.json belongs to the saved results. A retrained check folder never overwrites it.
    numbers = None if args.pilot or args.raw.resolve() != RESULTS_RAW.resolve() else NUMBERS
    result = aggregate(args.raw, args.out, numbers, args.pilot, args.experiments, ROOT / "logs", args.reps)
    compute = result["compute"]
    print(f"{compute.get('runs', 0)} runs, {compute.get('finished', 0)} finished, "
          f"{len(result['contrasts'])} contrast rows, written to {args.out}")
    if compute.get("not_finished"):
        print(f"{compute['not_finished']} runs did not finish. They are listed in excluded.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
