"""Aggregation of the main experiments, and every contrast of the analysis plan (P1 to P11).

Torch-free. Every number comes from the files the runs wrote. Pilot seeds (100 and up) never enter the
main analysis. A run that did not finish is never dropped silently: it is listed with its reason next to
every contrast it would have entered.

A cell is (task, condition, learner, arm, budget). Seeds pair across cells by their number.
When a cell has no runs at the asked budget, longer runs of the same cell are cut at that budget, and the
contrast says so.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import auc, eval_curve, final_return, obs_scale, repetition_summary, start_distance
from .stats import N_RESAMPLES, paired_bootstrap

PILOT_SEED_MIN = 100
REPEAT_ARMS = ("IER", "IER3", "IER_REL", "IER3_REL")
TASK_ORDER = ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"]
GYM_TASKS = ("HalfCheetah-v4", "Hopper-v4")
CONDITION_ORDER = ["default", "noise5", "noise10", "dyn20"]
LONG, SHORT = 1_000_000, 300_000
SKIP_EXPERIMENTS = {"g1_reference", "upstream_smoke"}  # written in other formats
# The main analysis reads only the main items, so no stray folder under results/raw can enter.
MAIN_EXPERIMENTS = ("A1", "B1", "C1", "A2", "R1", "D1", "B2", "B3", "A3")
MIN_REPEATS_FOR_RANK = 3
SOURCES = ("explore", "policy", "replay", "copy")  # as iers/batched.py, which this torch-free module cannot import

RUN_COLUMNS = ["experiment", "run_id", "task", "condition", "learner", "arm", "seed", "budget", "rn", "schedule",
               "status", "reason", "code_hash", "group_seeds", "wall_time_s", "guard_paused_s", "resume_count",
               "machine_type", "gpu", "processes_per_gpu",
               "n_evals", "final_return", "trigger_count", "repeat_count", "repetition_share", "r_ret", "r_ret_all",
               "repeats_in_window", "start_distance_mean",
               *[f"sampled_share_{s}" for s in SOURCES], "max_share_replay", "max_share_copy",
               "evicted_replay", "evicted_copy"]
REPEAT_COLUMNS = ["task", "condition", "learner", "arm", "budget", "seed", "run_id", "total_steps", "return",
                  "stored_return", "ratio", "start_distance"]


def task_sort_key(task: str) -> int:
    return TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)


def tasks_in(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["task"].unique(), key=task_sort_key)


# ---------------------------------------------------------------- reading the runs


def load_runs(raw_root: Path, spreads: dict, pilot: bool = False, experiments: list[str] | None = None):
    """Every run record under raw_root, with its metrics, curve and repeats.

    Returns (runs, curves, repeats): a table with one row per run, a dict from run_id (experiment/folder) to
    (steps, returns),
    and a table with one row per repeated episode of every IER-type run.
    """
    if experiments is None and not pilot:
        experiments = list(MAIN_EXPERIMENTS)
    rows, curves, repeat_tables = [], {}, []
    for record_path in sorted(Path(raw_root).glob("*/*/record.json")):
        run_dir = record_path.parent
        experiment = run_dir.parent.name
        if experiment in SKIP_EXPERIMENTS or (experiments is not None and experiment not in experiments):
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        spec = record["spec"]
        seed = int(spec["seed"])
        if pilot != (seed >= PILOT_SEED_MIN):
            continue
        row = {key: spec[key] for key in ("task", "condition", "learner", "arm", "budget", "rn", "schedule")}
        # Run folder names leave out the experiment and the budget, so A1 and B1 share names. The key adds the experiment.
        row.update(experiment=experiment, run_id=f"{experiment}/{record.get('run_id', run_dir.name)}", seed=seed,
                   status=record.get("status", "unknown"), reason=record.get("reason", ""),
                   code_hash=record.get("code_hash", ""), group_seeds=" ".join(map(str, record.get("group_seeds", []))),
                   wall_time_s=record.get("wall_time_s", np.nan), guard_paused_s=record.get("guard_paused_s", 0.0),
                   resume_count=record.get("resume_count", 0))
        machine = record.get("machine") or {}
        row.update(machine_type=machine.get("machine_type", ""), gpu=machine.get("gpu", ""),
                   processes_per_gpu=machine.get("processes_per_gpu", np.nan))
        if row["status"] == "finished":
            steps, returns = eval_curve(run_dir)
            if not (np.isfinite(returns).all() and len(steps)):
                row["status"], row["reason"] = "failed", "non-finite or empty evaluation curve"
            else:
                curves[row["run_id"]] = (steps, returns)
                row.update(n_evals=len(steps), final_return=final_return(steps, returns))
                row.update(source_summary(run_dir))
                if row["arm"] != "BASE":
                    summary = repetition_summary(run_dir, row["task"], spreads)
                    row.update(trigger_count=summary.trigger_count, repeat_count=summary.repeat_count,
                               repetition_share=summary.repetition_share, r_ret=summary.r_ret,
                               r_ret_all=summary.r_ret_all, repeats_in_window=summary.repeats_in_window,
                               start_distance_mean=summary.start_distance_mean)
                if row["arm"] in REPEAT_ARMS:
                    repeat_tables.append(read_repeats(run_dir, row, spreads))
        rows.append(row)
    runs = pd.DataFrame(rows).reindex(columns=RUN_COLUMNS)
    duplicated = runs[runs.duplicated(["task", "condition", "learner", "arm", "budget", "seed"], keep=False)]
    if len(duplicated):
        raise ValueError("The same cell and seed appear in more than one run folder:\n"
                         + duplicated[["experiment", "run_id"]].to_string())
    repeats = pd.concat(repeat_tables, ignore_index=True) if repeat_tables else pd.DataFrame(columns=REPEAT_COLUMNS)
    return runs, curves, repeats


def source_summary(run_dir: Path) -> dict:
    """Share of all sampled minibatch rows from each source over the run, the largest share of replayed and of
    copied rows in any one minibatch, and the replayed and copied rows evicted from the buffer by the end."""
    path = run_dir / "sources.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if not len(table):
        return {}
    sampled = table[[f"sampled_{s}" for s in SOURCES]].sum()
    total = float(sampled.sum())
    out = {f"sampled_share_{s}": float(sampled[f"sampled_{s}"]) / total if total else np.nan for s in SOURCES}
    out.update(max_share_replay=float(table.max_share_replay.max()), max_share_copy=float(table.max_share_copy.max()),
               evicted_replay=int(table.evicted_replay.iloc[-1]), evicted_copy=int(table.evicted_copy.iloc[-1]))
    return out


def wall_clock_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Wall-clock time per arm, one row per cell and machine type, counted once per group. The repetition share of environment steps is a separate column, so the two are never mixed up."""
    done = runs[runs.status == "finished"].dropna(subset=["wall_time_s"])
    if not len(done):
        return pd.DataFrame()
    rows = []
    keys = ["experiment", "task", "condition", "learner", "arm", "budget", "machine_type"]
    for key, cell in done.groupby(keys, sort=False, dropna=False):
        groups = cell.drop_duplicates(subset=["group_seeds", "wall_time_s"])
        sizes = groups.group_seeds.astype(str).str.split().str.len()
        row = dict(zip(keys, key))
        row.update(runs=len(cell), groups=len(groups), seeds_per_group_mean=float(sizes.mean()),
                   processes_per_gpu=float(groups.processes_per_gpu.mean()),
                   group_wall_hours_mean=float(groups.wall_time_s.mean() / 3600),
                   guard_pause_hours_mean=float(groups.guard_paused_s.fillna(0).mean() / 3600),
                   env_steps_per_s_per_group=float((cell.budget.iloc[0] * sizes / groups.wall_time_s).mean()),
                   repetition_share_of_env_steps=float(cell.repetition_share.mean()) if cell.arm.iloc[0] != "BASE"
                   else 0.0)
        for source in ("replay", "copy"):
            row[f"sampled_share_{source}"] = float(cell[f"sampled_share_{source}"].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values(["learner", "budget", "task", "condition", "arm"],
                             key=lambda c: c.map(task_sort_key) if c.name == "task"
                             else c.map(condition_key) if c.name == "condition" else c).reset_index(drop=True)


def run_configs(raw_root: Path, pilot: bool = False, experiments: list[str] | None = None) -> dict:
    """Every distinct (learner config, train config) pair the runs used, per learner, with a run count."""
    if experiments is None and not pilot:
        experiments = list(MAIN_EXPERIMENTS)
    found: dict[str, dict[str, dict]] = {}
    for record_path in sorted(Path(raw_root).glob("*/*/record.json")):
        experiment = record_path.parent.parent.name
        if experiment in SKIP_EXPERIMENTS or (experiments is not None and experiment not in experiments):
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        spec = record.get("spec") or {}
        if "learner_config" not in record or pilot != (int(spec.get("seed", -1)) >= PILOT_SEED_MIN):
            continue
        config = {"learner": record["learner_config"], "train": record["train_config"]}
        key = json.dumps(config, sort_keys=True)
        entry = found.setdefault(spec["learner"], {}).setdefault(key, {"config": config, "runs": 0})
        entry["runs"] += 1
    return {learner: list(entries.values()) for learner, entries in found.items()}


def read_repeats(run_dir: Path, row: dict, spreads: dict) -> pd.DataFrame:
    table = pd.read_csv(run_dir / "repeats.csv")
    if not len(table):
        return pd.DataFrame(columns=REPEAT_COLUMNS)
    with np.load(run_dir / "episodes.npz") as data:
        starts, stored = data["repeat_start_obs"], data["repeat_stored_start_obs"]
    if len(starts) != len(table):
        raise ValueError(f"{run_dir}: {len(table)} repeats but {len(starts)} start states")
    table["ratio"] = table["return"] / table["stored_return"]
    table["start_distance"] = start_distance(starts, stored, obs_scale(row["task"], spreads))
    for key in ("task", "condition", "learner", "arm", "budget", "seed", "run_id"):
        table[key] = row[key]
    return table.reindex(columns=REPEAT_COLUMNS)


def curves_table(runs: pd.DataFrame, curves: dict) -> pd.DataFrame:
    parts = []
    for row in runs[runs["run_id"].isin(curves.keys())].itertuples():
        steps, returns = curves[row.run_id]
        parts.append(pd.DataFrame({"task": row.task, "condition": row.condition, "learner": row.learner,
                                   "arm": row.arm, "budget": row.budget, "seed": row.seed, "run_id": row.run_id,
                                   "step": steps.astype(np.int64), "return": returns}))
    columns = ["task", "condition", "learner", "arm", "budget", "seed", "run_id", "step", "return"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------- cells, normalizers and scores


@dataclass
class Cell:
    task: str
    condition: str
    learner: str
    arm: str
    budget: int
    # Take the runs of this budget and cut them at `budget`, even if runs at `budget` itself exist. P6 and P11 use
    # it, because their SAC side comes from A1 cut at 300k, not from the 300k runs of B1.
    from_budget: int | None = None

    def label(self) -> str:
        return f"{self.learner} {self.arm} {self.task} {self.condition} {self.budget // 1000}k"


class Store:
    """Answers questions about cells, in normalized AUC."""

    def __init__(self, runs: pd.DataFrame, curves: dict, repeats: pd.DataFrame):
        self.runs, self.curves, self.repeats = runs, curves, repeats
        self.normalizers: dict[tuple[str, int], dict] = {}

    def rows(self, cell: Cell) -> pd.DataFrame:
        """Runs of the cell at its budget, or else at the smallest longer budget, to be cut."""
        r = self.runs
        same = r[(r.task == cell.task) & (r.condition == cell.condition) & (r.learner == cell.learner)
                 & (r.arm == cell.arm)]
        if cell.from_budget is not None:
            return same[same.budget == cell.from_budget]
        exact = same[same.budget == cell.budget]
        if len(exact):
            return exact
        longer = same[same.budget > cell.budget]
        return longer[longer.budget == longer.budget.min()] if len(longer) else longer

    def finished(self, cell: Cell) -> pd.DataFrame:
        rows = self.rows(cell)
        return rows[rows.status == "finished"]

    def cut(self, cell: Cell) -> bool:
        rows = self.rows(cell)
        return bool(len(rows)) and int(rows.budget.iloc[0]) != cell.budget

    def excluded(self, cell: Cell) -> list[str]:
        rows = self.rows(cell)
        return [f"{r.run_id} ({r.status}: {r.reason})" for r in rows.itertuples() if r.status != "finished"]

    def curve(self, run_id: str, budget: int) -> tuple[np.ndarray, np.ndarray]:
        steps, returns = self.curves[run_id]
        keep = steps <= budget
        return steps[keep], returns[keep]

    def normalizer(self, task: str, budget: int) -> float:
        """Mean final return of BASE SAC under default on the task at the budget."""
        key = (task, budget)
        if key not in self.normalizers:
            cell = Cell(task, "default", "sac", "BASE", budget)
            done = self.finished(cell)
            finals = [final_return(*self.curve(run_id, budget)) for run_id in done.run_id]
            self.normalizers[key] = {"task": task, "budget": budget, "value": float(np.mean(finals)) if finals else np.nan,
                                     "runs": len(finals), "cut_from_longer_runs": self.cut(cell)}
        return self.normalizers[key]["value"]

    def scores(self, cell: Cell, metric: str = "auc") -> pd.Series:
        """Per-seed normalized AUC, or normalized final return, of the cell."""
        norm = self.normalizer(cell.task, cell.budget)
        values = {}
        for row in self.finished(cell).itertuples():
            steps, returns = self.curve(row.run_id, cell.budget)
            if metric == "auc":
                values[row.seed] = auc(steps, returns, norm)
            elif metric == "final":
                values[row.seed] = final_return(steps, returns) / norm
            else:
                raise ValueError(metric)
        return pd.Series(values, dtype=float).sort_index()

    def run_values(self, cell: Cell, column: str) -> pd.Series:
        done = self.finished(cell)
        return pd.Series(done[column].to_numpy(dtype=float), index=done.seed.to_numpy()).sort_index()

    def gain(self, cell: Cell, metric: str = "auc") -> tuple[pd.Series, list[int], list[str]]:
        """Arm minus BASE of the same task, condition, learner and budget, paired by seed."""
        base = base_of(cell)
        diff, missing = paired(self.scores(cell, metric), self.scores(base, metric))
        return diff, missing, self.excluded(cell) + self.excluded(base)

    def has(self, cell: Cell) -> bool:
        return len(self.finished(cell)) > 0


def paired(a: pd.Series, b: pd.Series) -> tuple[pd.Series, list[int]]:
    """a minus b on the seeds both have, and the seeds only one side has."""
    common = a.index.intersection(b.index)
    missing = sorted(set(a.index).symmetric_difference(b.index))
    return (a.loc[common] - b.loc[common]).sort_index(), [int(s) for s in missing]


# ---------------------------------------------------------------- contrasts


@dataclass
class ContrastRow:
    id: str
    question: str
    label: str
    task: str
    learner: str
    condition: str
    budget: int
    kind: str = "paired"
    n: int = 0
    estimate: float = np.nan
    low: float = np.nan
    high: float = np.nan
    win_rate: float = np.nan
    sd: float = np.nan
    clear: bool = False
    seeds: str = ""
    missing_seeds: str = ""
    excluded_runs: str = ""
    cut_from_longer_runs: bool = False
    note: str = ""
    extra: dict = field(default_factory=dict)


def paired_row(base: ContrastRow, diffs: pd.Series, missing: list[int], excluded: list[str], cut: bool = False,
               seed: int = 0) -> ContrastRow:
    base.missing_seeds = " ".join(map(str, missing))
    base.excluded_runs = "; ".join(sorted(set(excluded)))
    base.cut_from_longer_runs = cut
    if len(diffs) == 0:
        base.note = "no seed has both sides"
        return base
    result = paired_bootstrap(diffs.to_numpy(), seed=seed)
    base.n, base.estimate, base.low, base.high = result.n, result.mean, result.low, result.high
    base.win_rate, base.sd, base.clear = result.win_rate, result.sd, result.clear
    base.seeds = " ".join(str(int(s)) for s in diffs.index)
    return base


def spearman_bootstrap(x: pd.DataFrame, y: pd.DataFrame, seed: int = 0, resamples: int = N_RESAMPLES) -> dict:
    """Spearman correlation across cells between per-cell means, with a bootstrap CI over seeds.

    x and y have one row per cell and one column per seed, NaN where a cell lacks that seed. Each resample
    draws seeds with replacement and recomputes every cell's mean over the drawn seeds it has.
    """
    x, y = x.astype(float), y.astype(float)
    seeds = x.columns.union(y.columns)
    x, y = x.reindex(columns=seeds), y.reindex(columns=seeds)
    X, Y = x.to_numpy(), y.to_numpy()

    def rho(columns) -> float:
        with warnings.catch_warnings():  # a cell may lack every drawn seed, which gives NaN on purpose
            warnings.simplefilter("ignore", RuntimeWarning)
            mx = np.nanmean(X[:, columns], axis=1) if X.shape[1] else np.full(len(X), np.nan)
            my = np.nanmean(Y[:, columns], axis=1) if Y.shape[1] else np.full(len(Y), np.nan)
        ok = np.isfinite(mx) & np.isfinite(my)
        if ok.sum() < 3 or np.ptp(mx[ok]) == 0 or np.ptp(my[ok]) == 0:
            return np.nan
        return float(spearmanr(mx[ok], my[ok]).statistic)

    estimate = rho(np.arange(len(seeds)))
    rng = np.random.default_rng(seed)
    draws = np.array([rho(rng.integers(0, len(seeds), len(seeds))) for _ in range(resamples)])
    finite = draws[np.isfinite(draws)]
    low, high = (np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan))
    return {"cells": int(len(X)), "estimate": estimate, "low": float(low), "high": float(high),
            "clear": bool(np.isfinite(low) and (low > 0 or high < 0)),
            "resamples_used": int(len(finite)), "seeds": int(len(seeds))}


def contrasts(store: Store) -> pd.DataFrame:
    rows: list[ContrastRow] = []
    runs = store.runs
    tasks = tasks_in(runs) if len(runs) else []

    def add(row: ContrastRow, a: Cell, b: Cell):
        if store.has(a) and store.has(b):
            diffs, missing = paired(store.scores(a), store.scores(b))
            rows.append(paired_row(row, diffs, missing, store.excluded(a) + store.excluded(b),
                                   store.cut(a) or store.cut(b)))

    def add_gain_difference(row: ContrastRow, a: Cell, b: Cell):
        """(gain of a over its BASE) minus (gain of b over its BASE), paired by seed."""
        if not all(store.has(c) for c in (a, b, base_of(a), base_of(b))):
            return
        gain_a, miss_a, ex_a = store.gain(a)
        gain_b, miss_b, ex_b = store.gain(b)
        diffs, missing = paired(gain_a, gain_b)
        cut = any(store.cut(c) for c in (a, b, base_of(a), base_of(b)))
        rows.append(paired_row(row, diffs, sorted(set(missing + miss_a + miss_b)), ex_a + ex_b, cut))

    for task in tasks:
        # P1: IER minus BASE, SAC, default, 1M.
        add(ContrastRow("P1", "Q1", "IER minus BASE", task, "sac", "default", LONG),
            Cell(task, "default", "sac", "IER", LONG), Cell(task, "default", "sac", "BASE", LONG))
        # P2: IER minus IER_REL, SAC, default, 1M.
        add(ContrastRow("P2", "Q1", "IER minus IER_REL", task, "sac", "default", LONG),
            Cell(task, "default", "sac", "IER", LONG), Cell(task, "default", "sac", "IER_REL", LONG))
        # P3: IER3 minus DUP, SAC, 300k, default and noise5.
        for condition in ("default", "noise5"):
            add(ContrastRow("P3", "Q2", "IER3 minus DUP", task, "sac", condition, SHORT),
                Cell(task, condition, "sac", "IER3", SHORT), Cell(task, condition, "sac", "DUP", SHORT))
        # P4: IER3 gain under a stress condition minus IER3 gain under default, 300k, every learner with data.
        for learner in ("sac", "crossq"):
            for condition in ("noise5", "noise10", "dyn20"):
                add_gain_difference(
                    ContrastRow("P4", "Q4", f"IER3 gain under {condition} minus IER3 gain under default", task, learner,
                                condition, SHORT),
                    Cell(task, condition, learner, "IER3", SHORT), Cell(task, "default", learner, "IER3", SHORT))
        # P6: IER gain for CrossQ, and CrossQ gain minus SAC gain, at every budget CrossQ IER ran at.
        crossq_budgets = sorted(runs[(runs.task == task) & (runs.learner == "crossq") & (runs.arm == "IER")
                                     & (runs.condition == "default")].budget.unique())
        for budget in crossq_budgets:
            budget = int(budget)
            sac_from = LONG if budget < LONG else None  # A1 cut at the CrossQ budget
            add(ContrastRow("P6", "Q5", "IER minus BASE, CrossQ", task, "crossq", "default", budget),
                Cell(task, "default", "crossq", "IER", budget), Cell(task, "default", "crossq", "BASE", budget))
            add_gain_difference(ContrastRow("P6", "Q5", "IER gain for CrossQ minus IER gain for SAC", task, "both",
                                            "default", budget),
                                Cell(task, "default", "crossq", "IER", budget),
                                Cell(task, "default", "sac", "IER", budget, sac_from))
            # P11: is the 1024-wide CrossQ the stronger learner here? Descriptive, not part of any verdict.
            add(ContrastRow("P11", "Q5", "CrossQ BASE minus SAC BASE", task, "both", "default", budget),
                Cell(task, "default", "crossq", "BASE", budget), Cell(task, "default", "sac", "BASE", budget, sac_from))
        # P7: IER3 minus IER, SAC, default, 1M.
        add(ContrastRow("P7", "Q6", "IER3 minus IER", task, "sac", "default", LONG),
            Cell(task, "default", "sac", "IER3", LONG), Cell(task, "default", "sac", "IER", LONG))
        # P9 and P10: retention of every IER-type run, per task, learner, condition, arm and budget.
        for cell in repeat_cells(store, task):
            rows.extend(retention_rows(store, cell))

    rows.extend(p5_rows(store))
    rows.extend(p8_rows(store))
    return pd.DataFrame([{k: v for k, v in vars(r).items() if k != "extra"} | r.extra for r in rows])


def base_of(cell: Cell) -> Cell:
    return Cell(cell.task, cell.condition, cell.learner, "BASE", cell.budget, cell.from_budget)


def repeat_cells(store: Store, task: str | None = None) -> list[Cell]:
    runs = store.runs
    chosen = runs[runs.arm.isin(REPEAT_ARMS)] if task is None else runs[(runs.task == task) & runs.arm.isin(REPEAT_ARMS)]
    keys = chosen[["task", "condition", "learner", "arm", "budget"]].drop_duplicates()
    cells = [Cell(r.task, r.condition, r.learner, r.arm, int(r.budget)) for r in keys.itertuples()]
    return sorted(cells, key=lambda c: (task_sort_key(c.task), c.learner, condition_key(c.condition), c.arm, c.budget))


def condition_key(condition: str) -> int:
    return CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER)


def retention_rows(store: Store, cell: Cell) -> list[ContrastRow]:
    """P9: per-run median retention ratio minus 1. P10: per-run Spearman of start distance against ratio."""
    reps = store.repeats
    done = store.finished(cell)
    mine = reps[reps.run_id.isin(done.run_id)]
    median, rank = {}, {}
    for seed, group in mine.groupby("seed"):
        median[seed] = float(np.median(group["ratio"])) - 1.0
        if len(group) >= MIN_REPEATS_FOR_RANK and np.ptp(group["ratio"]) > 0 and np.ptp(group["start_distance"]) > 0:
            rank[seed] = float(spearmanr(group["start_distance"], group["ratio"]).statistic)
    no_repeats = sorted(set(done.seed) - set(median))
    out = []
    common = dict(task=cell.task, learner=cell.learner, condition=cell.condition, budget=cell.budget)
    for pid, label, values, lacking in (
        ("P9", f"{cell.arm}: median retention ratio minus 1", median, no_repeats),
        ("P10", f"{cell.arm}: Spearman of start distance against retention, within runs", rank,
         sorted(set(done.seed) - set(rank))),
    ):
        row = ContrastRow(pid, "Q3", label, **common)
        series = pd.Series(values, dtype=float).sort_index()
        row = paired_row(row, series, [], store.excluded(cell))
        if lacking:
            row.note = f"seeds without enough repeats: {' '.join(map(str, lacking))}"
        out.append(row)
    return out


def p5_rows(store: Store) -> list[ContrastRow]:
    """Q4 axis: mean start distance of a (task, condition) cell against its mean IER3 gain, 300k."""
    out = []
    for learner in ("sac", "crossq"):
        for family, keep in (("all tasks", None), ("Gymnasium", True), ("DeepMind Control", False)):
            gains, distances = {}, {}
            for cell in repeat_cells(store):
                if cell.arm != "IER3" or cell.learner != learner or cell.budget != SHORT:
                    continue
                if keep is not None and (cell.task in GYM_TASKS) != keep:
                    continue
                if not store.has(base_of(cell)):
                    continue
                gains[(cell.task, cell.condition)] = store.gain(cell)[0]
                distances[(cell.task, cell.condition)] = store.run_values(cell, "start_distance_mean")
            if len(gains) < 3:
                continue
            result = spearman_bootstrap(pd.DataFrame(distances).T, pd.DataFrame(gains).T)
            out.append(correlation_row("P5", "Q4", f"Spearman of mean start distance against mean IER3 gain, {family}",
                                       learner, SHORT, result))
    return out


def p8_rows(store: Store) -> list[ContrastRow]:
    """Q7: R_ret of every IER-type cell against its gain over BASE, one point per cell."""
    gains, r_ret = {}, {}
    for cell in repeat_cells(store):
        if not store.has(base_of(cell)):
            continue
        key = (cell.task, cell.condition, cell.learner, cell.arm, cell.budget)
        gains[key] = store.gain(cell)[0]
        r_ret[key] = store.run_values(cell, "r_ret")
    if len(gains) < 3:
        return []
    result = spearman_bootstrap(pd.DataFrame(r_ret).T, pd.DataFrame(gains).T)
    return [correlation_row("P8", "Q7", "Spearman of R_ret against IER gain, one point per cell", "all", 0, result)]


def correlation_row(pid: str, question: str, label: str, learner: str, budget: int, result: dict) -> ContrastRow:
    row = ContrastRow(pid, question, label, "all", learner, "all", budget, kind="spearman", n=result["cells"],
                      estimate=result["estimate"], low=result["low"], high=result["high"], clear=result["clear"])
    row.note = f"{result['cells']} cells, bootstrap over {result['seeds']} seeds, {result['resamples_used']} usable resamples"
    return row


def gain_table(store: Store) -> pd.DataFrame:
    """Gain over BASE of every non-BASE cell that has a BASE partner, with its CI, start distance and R_ret."""
    rows = []
    runs = store.runs
    keys = runs[runs.arm != "BASE"][["task", "condition", "learner", "arm", "budget"]].drop_duplicates()
    for key in keys.itertuples():
        cell = Cell(key.task, key.condition, key.learner, key.arm, int(key.budget))
        if not (store.has(cell) and store.has(base_of(cell))):
            continue
        diffs, missing, excluded = store.gain(cell)
        row = paired_row(ContrastRow("gain", "", f"{cell.arm} minus BASE", cell.task, cell.learner, cell.condition,
                                     cell.budget), diffs, missing, excluded, store.cut(cell) or store.cut(base_of(cell)))
        out = {k: v for k, v in vars(row).items() if k not in ("extra", "id", "question", "kind")}
        out["arm"] = cell.arm
        for column in ("start_distance_mean", "r_ret", "repetition_share", "trigger_count"):
            values = store.run_values(cell, column)
            out[column] = float(values.mean()) if len(values) else np.nan
        rows.append(out)
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["learner", "budget", "task", "condition", "arm"],
                                  key=lambda c: c.map(task_sort_key) if c.name == "task"
                                  else c.map(condition_key) if c.name == "condition" else c)
    return table.reset_index(drop=True)


def reference_curves(raw_root: Path, experiments: tuple[str, ...] = ("g1", "g1_reference", "g1_debug")) -> pd.DataFrame:
    """Evaluation curves of the implementation check: ours against the stable-baselines3 references."""
    parts = []
    for experiment in experiments:
        for record_path in sorted((Path(raw_root) / experiment).glob("*/record.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") != "finished":
                continue
            steps, returns = eval_curve(record_path.parent)
            spec = record.get("spec") or {}
            implementation = f"sb3_{record['algo']}" if "algo" in record else f"ours_{spec.get('learner')}"
            if record.get("actor") == "upstream":
                implementation += "_upstream_logstd"
            parts.append(pd.DataFrame({"experiment": experiment, "implementation": implementation,
                                       "task": record.get("task", spec.get("task")),
                                       "seed": record.get("seed", spec.get("seed")),
                                       "step": steps.astype(np.int64), "return": returns}))
    columns = ["experiment", "implementation", "task", "seed", "step", "return"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------- rliable aggregates


def score_matrix(store: Store, learner: str, arms: list[str], budget: int, condition: str = "default",
                 metric: str = "auc") -> tuple[dict[str, np.ndarray], list[str], list[int], list[str]]:
    """Scores of shape (seeds, tasks) per arm, on the seeds every arm has on every task.

    Returns the matrices, the tasks, the seeds used, and every run left out with the reason.
    """
    tasks = [t for t in TASK_ORDER if all(store.has(Cell(t, condition, learner, a, budget)) for a in arms)]
    series = {(a, t): store.scores(Cell(t, condition, learner, a, budget), metric) for a in arms for t in tasks}
    if not tasks:
        return {}, [], [], []
    seeds = sorted(set.intersection(*(set(s.index) for s in series.values())))
    left_out = []
    for (a, t), s in series.items():
        left_out += [f"{learner} {a} {t} seed {seed}: missing its partner on another task or arm"
                     for seed in s.index if seed not in seeds]
        left_out += store.excluded(Cell(t, condition, learner, a, budget))
    matrices = {a: np.column_stack([series[(a, t)].loc[seeds].to_numpy() for t in tasks]) for a in arms}
    return matrices, tasks, seeds, left_out


def rliable_summary(store: Store, learner: str, budget: int, reps: int = N_RESAMPLES) -> dict | None:
    """IQM of normalized AUC with stratified bootstrap CIs, and the probability that IER improves on BASE."""
    from rliable import library as rly
    from rliable import metrics

    matrices, tasks, seeds, left_out = score_matrix(store, learner, ["BASE", "IER"], budget)
    if not tasks or len(seeds) < 2:
        return None
    iqm, iqm_ci = rly.get_interval_estimates(matrices, lambda x: np.array([metrics.aggregate_iqm(x)]), reps=reps)
    pair = {"IER,BASE": (matrices["IER"], matrices["BASE"])}
    poi, poi_ci = rly.get_interval_estimates(pair, metrics.probability_of_improvement, reps=reps)
    return {
        "learner": learner, "budget": budget, "tasks": tasks, "seeds": seeds, "left_out": left_out,
        "iqm": {arm: {"estimate": float(iqm[arm][0]), "low": float(iqm_ci[arm][0][0]), "high": float(iqm_ci[arm][1][0])}
                for arm in matrices},
        "probability_of_improvement": {"estimate": float(np.ravel(poi["IER,BASE"])[0]),
                                       "low": float(np.ravel(poi_ci["IER,BASE"][0])[0]),
                                       "high": float(np.ravel(poi_ci["IER,BASE"][1])[0])},
    }


def performance_profile(store: Store, learner: str, budget: int, reps: int = N_RESAMPLES) -> dict | None:
    """Fraction of runs above each normalized final return threshold, with bootstrap CIs."""
    from rliable import library as rly

    matrices, tasks, seeds, _ = score_matrix(store, learner, ["BASE", "IER"], budget, metric="final")
    if not tasks or len(seeds) < 2:
        return None
    upper = max(1.5, float(max(m.max() for m in matrices.values())))
    taus = np.linspace(0.0, upper, 61)
    profiles, cis = rly.create_performance_profile(matrices, taus, reps=reps)
    return {"taus": taus.tolist(), "tasks": tasks, "seeds": seeds,
            "profiles": {a: np.asarray(profiles[a]).tolist() for a in matrices},
            "low": {a: np.asarray(cis[a][0]).tolist() for a in matrices},
            "high": {a: np.asarray(cis[a][1]).tolist() for a in matrices}}
