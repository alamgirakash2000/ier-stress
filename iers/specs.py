"""What the supervisor, the grid and the gate scripts share with the runner, without importing torch.

On Windows, importing the CUDA build of torch commits about 1.4 GB per process, whether or not it is used.
The processes that only start and watch other processes must stay light, or they use up the commit charge
that the training processes need. A test checks that these modules stay free of torch.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import ROOT
from .repetition import arm_rn, arm_schedule

RESULTS_RAW = ROOT / "results" / "raw"
RESULTS_CKPT = ROOT / "results" / "checkpoints"
TASKS_FILE = ROOT / "configs" / "tasks" / "tasks.yaml"
SPREADS_FILE = ROOT / "configs" / "tasks" / "reset_spreads.yaml"


def load_tasks() -> dict:
    with open(TASKS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_spreads() -> dict:
    if not SPREADS_FILE.exists():
        return {}
    with open(SPREADS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# ------------------------------------------------------------------ configuration


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    task: str
    condition: str
    learner: str
    arm: str
    seed: int
    budget: int
    rn: int
    schedule: str = "algorithm1"

    @property
    def run_id(self) -> str:
        suffix = "" if self.schedule == arm_schedule(self.arm) else f"__{self.schedule}"
        return f"{self.task}__{self.condition}__{self.learner}__{self.arm}__seed{self.seed}{suffix}"


@dataclass
class TrainConfig:
    # Table 5 of the IER paper.
    exploration_steps: int = 1000
    eval_every: int = 10_000
    eval_episodes: int = 10
    buffer_capacity: int = 1_000_000
    checkpoint_every: int = 50_000
    updates_per_step: int = 1


@dataclass
class Hardware:
    device: str = "cuda"
    replay_device: str = "cuda"
    float64: bool = False
    label: str = "unset"

    @property
    def dtype(self):
        import torch

        return torch.float64 if self.float64 else torch.float32


def make_spec(experiment: str, task: str, condition: str, learner: str, arm: str, seed: int, budget: int,
              task_rn: int, schedule: str | None = None, rn: int | None = None) -> RunSpec:
    """rn overrides the arm's RN. Only DUP uses this, to match the arm it is compared with."""
    if rn is not None and arm != "DUP":
        raise ValueError("Only DUP may override its RN.")
    return RunSpec(experiment, task, condition, learner, arm, seed, budget,
                   arm_rn(arm, task_rn) if rn is None else rn, schedule or arm_schedule(arm))


def is_finished(spec: RunSpec, results_root: Path = RESULTS_RAW) -> bool:
    path = Path(results_root) / spec.experiment / spec.run_id / "record.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") in ("finished", "failed")
    except json.JSONDecodeError:
        return False


# ------------------------------------------------------------------ atomic writes


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
    replace_file(tmp, path)


def replace_file(tmp: Path, path: Path) -> None:
    # Windows can refuse a rename for a moment while a scanner holds the file open.
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_csv(path: Path, header: list[str], rows: list) -> None:
    lines = [",".join(header)]
    lines += [",".join(str(v) for v in row) for row in rows]
    atomic_write_text(path, "\n".join(lines) + "\n")


def atomic_savez(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    replace_file(tmp, path)
