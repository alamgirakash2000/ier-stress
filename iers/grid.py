"""Runs a list of RunSpec cells in GPU groups, with resume, retries and resource checks.

This process only starts and watches workers, so it must not import torch. See iers/specs.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from . import ROOT
from .guard import (
    EXIT_RETRY,
    STOP_FILE,
    GpuProbe,
    GuardLimits,
    estimate_group_vram_mb,
    largest_group_that_fits,
    wait_for_room,
)
from .logging import get_logger
from .repetition import ARM_TO_MODE, Mode
from .resources import ResourceMonitor
from .specs import RESULTS_CKPT, RESULTS_RAW, Hardware, TrainConfig, atomic_write_text, is_finished, load_tasks

HARDWARE_FILE = ROOT / "configs" / "hardware.yaml"
PROCESS_RAM_MB = 1300  # private RAM of one CUDA process, plus margin, from logs/benchmark_results.json
VRAM_FLOOR_MB = 500.0  # the GPU memory wait_for_room keeps free


def process_cap_mb(hw_file: dict) -> float:
    """The most GPU memory one training process may take. Cloud machines with more memory set IERS_GPU_CAP_MB, so
    that no hashed file changes between machines."""
    return float(os.environ.get("IERS_GPU_CAP_MB") or hw_file.get("gpu_process_cap_mb", GuardLimits.gpu_process_cap_mb))


def group_memory_cap_mb(processes: int, hw_file: dict, probe: GpuProbe | None = None) -> float:
    """GPU memory one group may plan for: the per-process cap, but never more than an equal share of the GPU
    memory free now, less the floor that wait_for_room keeps. Before this, the cap alone allowed a group that
    then waited forever for memory that never freed."""
    cap = process_cap_mb(hw_file)
    memory = (probe or GpuProbe()).memory_mb()
    if memory:
        cap = min(cap, (memory[0] - VRAM_FLOOR_MB) / max(1, processes))
    return cap


def hardware_defaults() -> dict:
    if HARDWARE_FILE.exists():
        with open(HARDWARE_FILE, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {"device": "cuda", "replay_device": "cuda", "group_size": 1, "processes": 1, "label": "unset"}


def task_dims(task: str) -> tuple[int, int]:
    """Observation and action sizes from configs/tasks/tasks.yaml. A test checks them against the environments."""
    spec = load_tasks()[task]
    return spec["obs_dim"], spec["act_dim"]


def group_capacity(group: list, train: TrainConfig) -> int:
    """The replay rows allocated, the same rule as GroupRunner."""
    needs_full = any(ARM_TO_MODE[s.arm] == Mode.DUPLICATE for s in group)
    return train.buffer_capacity if needs_full else min(train.buffer_capacity, group[0].budget)


def make_groups(specs: list, size: int, train: TrainConfig, cap_mb: float | None = None) -> list[list]:
    """Group by what must match inside one group: task, learner and budget, and by item, so that a grid over
    several items never mixes them in one group.

    DUP runs group together, because a group holding one DUP run allocates the full replay capacity for
    all its seeds. A group shrinks if its estimated GPU memory would go past the cap.
    """
    buckets: dict = {}
    for spec in specs:
        buckets.setdefault((spec.experiment, spec.task, spec.learner, spec.budget), []).append(spec)
    groups = []
    for (_experiment, task, learner, _budget), bucket in buckets.items():
        obs_dim, act_dim = task_dims(task)
        bucket = sorted(bucket, key=lambda s: ARM_TO_MODE[s.arm] == Mode.DUPLICATE)
        position = 0
        while position < len(bucket):
            members = bucket[position:position + size]
            capacity = group_capacity(members, train)
            kwargs = {} if cap_mb is None else {"cap_mb": cap_mb}
            fit = largest_group_that_fits(learner, len(members), obs_dim, act_dim, capacity, **kwargs)
            groups.append(members[:fit])
            position += fit
    return groups


def job_key(group: list) -> str:
    """The job file's name. Run folder names leave out the item and the budget, so A1 BASE and B1 BASE on one task
    share them. The key includes both, so two items in one grid never share a job file."""
    return hashlib.sha256("|".join(f"{s.experiment}/{s.run_id}/{s.budget}" for s in group).encode()).hexdigest()[:12]


def record_crash(group: list, reason: str, results_root: Path) -> None:
    for spec in group:
        out = results_root / spec.experiment / spec.run_id
        record = {"run_id": spec.run_id, "status": "crashed", "reason": reason, "spec": asdict(spec),
                  "group_seeds": [s.seed for s in group]}
        atomic_write_text(out / "record.json", json.dumps(record, indent=1))


def run_grid(experiment: str, specs: list, train: TrainConfig | None = None, group_size: int | None = None,
             processes: int | None = None, device: str | None = None, replay_device: str | None = None,
             results_root: Path = RESULTS_RAW, ckpt_root: Path = RESULTS_CKPT, force: bool = False,
             retries: int = 2, monitor_interval: float = 60.0) -> int:
    """Run every unfinished cell. Returns 0 if all finished, 1 if any group failed, EXIT_RETRY if asked to stop."""
    train = train or TrainConfig()
    hw_file = hardware_defaults()
    device = device or hw_file["device"]
    replay_device = replay_device or hw_file.get("replay_device", device)
    group_size = group_size or hw_file.get("group_size", 1)
    processes = processes or hw_file.get("processes", 1)
    hardware = Hardware(device=device, replay_device=replay_device,
                        label=f"{device}, S={group_size}, P={processes}, replay={replay_device}")
    results_root, ckpt_root = Path(results_root), Path(ckpt_root)
    logger = get_logger(experiment)
    logger.info("experiment %s hardware: %s", experiment, hardware.label)

    todo = specs if force else [s for s in specs if not is_finished(s, results_root)]
    logger.info("%d runs in the grid, %d to run", len(specs), len(todo))
    cap_mb = group_memory_cap_mb(processes, hw_file) if device == "cuda" else None
    if cap_mb is not None:
        logger.info("planning GPU memory per group: at most %.0f MB", cap_mb)
    groups = make_groups(todo, group_size, train, cap_mb=cap_mb)

    jobs_dir = ckpt_root / experiment / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONWARNINGS="ignore::DeprecationWarning",
               IERS_PROCESSES=str(processes))

    queue = []
    for group in groups:
        key = job_key(group)
        path = jobs_dir / f"{key}.json"
        job = {
            "specs": [asdict(s) for s in group],
            "hardware": asdict(hardware),
            "train": asdict(train),
            "results_root": str(results_root),
            "ckpt_root": str(ckpt_root),
            "console": True,
            "gpu_cap_mb": process_cap_mb(hw_file),
        }
        atomic_write_text(path, json.dumps(job, indent=1))
        estimate = 0.0
        if device == "cuda":
            obs_dim, act_dim = task_dims(group[0].task)
            estimate = estimate_group_vram_mb(group[0].learner, len(group), obs_dim, act_dim,
                                              group_capacity(group, train))
        logger.info("group %s: %d seeds, %s, %s, estimated GPU memory %.0f MB", key, len(group), group[0].task,
                    group[0].learner, estimate)
        queue.append({"group": group, "path": path, "tries": 0, "vram": estimate})

    # A caller that runs several grids at once keeps its own monitor, and passes 0 here.
    monitor = ResourceMonitor(experiment, interval=monitor_interval).start() if monitor_interval else None
    running: list[dict] = []
    failed = 0
    stopped = False
    start = time.time()
    script = str(ROOT / "scripts" / "run_group.py")
    while queue or running:
        if STOP_FILE.exists() and not running:
            stopped = True
            logger.info("%s exists, stopping with %d groups left", STOP_FILE.name, len(queue))
            break
        while queue and len(running) < processes and not STOP_FILE.exists():
            item = queue[0]
            log = (lambda _message: None) if item.get("announced") else logger.info
            if not wait_for_room(PROCESS_RAM_MB, item["vram"], log=log, timeout_s=5.0):
                item["announced"] = True
                break
            queue.pop(0)
            item["announced"] = False
            item["process"] = subprocess.Popen([sys.executable, script, str(item["path"])], env=env)
            item["tries"] += 1
            running.append(item)
        time.sleep(1.0)
        for item in list(running):
            code = item["process"].poll()
            if code is None:
                continue
            running.remove(item)
            if code == 0:
                continue
            if code == EXIT_RETRY:
                # A clean stop by the guard. This does not use up a retry.
                item["tries"] -= 1
                logger.warning("group %s stopped cleanly, it will resume from its checkpoint", item["path"].stem)
                queue.append(item)
                time.sleep(30.0)
            elif item["tries"] <= retries:
                logger.warning("group %s exited with %s, retrying from its checkpoint", item["path"].stem, code)
                queue.append(item)
                time.sleep(30.0)
            else:
                failed += 1
                crash = item["path"].with_suffix(".crash.txt")
                reason = crash.read_text(encoding="utf-8") if crash.exists() else f"exit code {code}"
                record_crash(item["group"], reason, results_root)
                logger.error("group %s failed for good: %s", item["path"].stem, reason[-400:])
    summary = monitor.stop() if monitor else {}
    logger.info("experiment %s done in %.1f s, resources: %s", experiment, time.time() - start, summary)
    if stopped:
        return EXIT_RETRY
    return 1 if failed else 0
