"""Implementation check (Gate G1): our SAC and CrossQ against independent references, seeds 100 to 102.

    python scripts/gate_g1.py run       trains whatever is unfinished, then reports
    python scripts/gate_g1.py report    only reports

Pass rule: for each pair, our 3-seed mean final return is within 20 percent of the reference
mean, or the reference mean lies inside the range of our 3 seeds.

Exit codes: 0 gate passed, 3 gate failed, 75 stopped early and should run again, 1 something crashed.
The batched GPU runs here also serve as a distribution-level check of batched training.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

import numpy as np  # noqa: E402

from iers import ROOT  # noqa: E402
from iers.guard import EXIT_RETRY, STOP_FILE, wait_for_room  # noqa: E402
from iers.logging import get_logger  # noqa: E402
from iers.metrics import eval_curve, final_return, read_record  # noqa: E402
from iers.resources import ResourceMonitor  # noqa: E402
from iers.specs import RESULTS_RAW, atomic_write_text, is_finished, make_spec  # noqa: E402

TASK = "HalfCheetah-v4"
LEARNERS = ("sac", "crossq")
SEEDS = [100, 101, 102]
BUDGET = 300_000
OURS = "g1"
REFERENCE = "g1_reference"
DEBUG = "g1_debug"
AGGREGATED = ROOT / "results" / "aggregated"
SB3_JOBS = {"sac": ("cpu", 700.0, 0.0), "crossq": ("cuda", 1300.0, 1400.0)}


def our_specs(learner: str) -> list:
    return [make_spec(OURS, TASK, "default", learner, "BASE", seed, BUDGET, 0) for seed in SEEDS]


def reference_dir(algo: str, seed: int) -> Path:
    return RESULTS_RAW / REFERENCE / f"{TASK}__default__sb3_{algo}__BASE__seed{seed}"


def debug_dir(seed: int) -> Path:
    return RESULTS_RAW / DEBUG / f"{TASK}__default__sb3_sac_upstream_logstd__BASE__seed{seed}"


def run_debug(log) -> int:
    """SAC follow-up check: SB3 SAC with the log std of the upstream IER actor, on the same seeds."""
    script = str(ROOT / "scripts" / "sb3_reference.py")
    jobs = []
    for seed in SEEDS:
        if (debug_dir(seed) / "record.json").exists():
            continue
        command = [sys.executable, script, "--algo", "sac", "--actor", "upstream", "--seed", str(seed),
                   "--budget", str(BUDGET), "--task", TASK, "--device", "cpu", "--experiment", DEBUG]
        jobs.append({"name": f"sb3_sac_upstream_logstd_{seed}", "command": command, "ram": 700.0, "vram": 0.0,
                     "tries": 0})
    return launch(jobs, log, "g1_debug")


def wait_for_other_debug(log, poll_s: float = 30.0) -> None:
    """Wait while another `gate_g1.py debug-sac` process is alive, so the runs are never started twice."""
    import psutil

    me = os.getpid()
    announced = False
    while True:
        others = []
        for process in psutil.process_iter(["pid", "cmdline"]):
            cmdline = process.info["cmdline"] or []
            if process.info["pid"] != me and any(p.endswith("gate_g1.py") for p in cmdline) and "debug-sac" in cmdline:
                others.append(process.info["pid"])
        if not others:
            return
        if not announced:
            log(f"waiting for the debug-sac process {others} started earlier")
            announced = True
        time.sleep(poll_s)


def report_debug() -> dict | None:
    ours, reference = [], []
    for spec in our_specs("sac"):
        ours.append(final_return(*eval_curve(RESULTS_RAW / OURS / spec.run_id)))
    for seed in SEEDS:
        run = debug_dir(seed)
        if not (run / "record.json").exists():
            return None
        reference.append(final_return(*eval_curve(run)))
    result = {"task": TASK, "seeds": SEEDS, "budget": BUDGET,
              "question": "Does SB3 SAC with the upstream log std of the IER actor match our SAC?",
              "rule": "the Gate G1 rule, fixed before this run: within 20 percent, or reference mean inside our range",
              "pair": pair_verdict(ours, reference)}
    atomic_write_text(AGGREGATED / "gate_g1_sac_debug.json", json.dumps(result, indent=1))
    print("sac debug", json.dumps(result["pair"]))
    print("GATE G1 SAC DEBUG:", "PASS" if result["pair"]["passed"] else "FAIL")
    return result


def run_all(log, learners=LEARNERS) -> int:
    """Each of our groups and each reference run is its own process, started as memory allows."""
    script = str(ROOT / "scripts" / "sb3_reference.py")
    ours_script = str(ROOT / "scripts" / "run_experiment.py")
    jobs = []
    for learner in learners:
        if not all(is_finished(s) for s in our_specs(learner)):
            command = [sys.executable, ours_script, "--experiment", OURS, "--tasks", TASK, "--learners", learner,
                       "--arms", "BASE", "--seeds", *[str(s) for s in SEEDS], "--budget", str(BUDGET),
                       "--group-size", "3", "--processes", "1", "--monitor-interval", "0"]
            need = (1300.0, 900.0) if learner == "sac" else (1300.0, 1700.0)
            jobs.append({"name": f"ours_{learner}", "command": command, "ram": need[0], "vram": need[1], "tries": 0})
        device, ram, vram = SB3_JOBS[learner]
        for seed in SEEDS:
            if (reference_dir(learner, seed) / "record.json").exists():
                continue
            command = [sys.executable, script, "--algo", learner, "--seed", str(seed), "--budget", str(BUDGET),
                       "--task", TASK, "--device", device]
            jobs.append({"name": f"sb3_{learner}_{seed}", "command": command, "ram": ram, "vram": vram, "tries": 0})
    return launch(jobs, log, "g1")


def launch(jobs: list[dict], log, monitor_name: str) -> int:
    """Start each job as memory allows, retry crashes up to 3 times, and resume clean stops."""
    monitor = ResourceMonitor(monitor_name, interval=60.0).start()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONWARNINGS="ignore")
    log(f"{monitor_name} jobs: {[j['name'] for j in jobs]}")
    running = []
    failed = []
    stopped = False
    while jobs or running:
        if STOP_FILE.exists() and not running:
            stopped = True
            break
        if jobs and not STOP_FILE.exists():
            job = jobs[0]
            quiet = (lambda _message: None) if job.get("announced") else log
            if not wait_for_room(job["ram"], job["vram"], log=quiet, timeout_s=5.0):
                job["announced"] = True
            else:
                jobs.pop(0)
                job["announced"] = False
                job["tries"] += 1
                log(f"launching {job['name']}, try {job['tries']}")
                job["process"] = subprocess.Popen(job["command"], env=env)
                running.append(job)
                time.sleep(45.0)  # Let CUDA finish its startup surge before judging the free memory.
        time.sleep(5.0)
        for job in list(running):
            code = job["process"].poll()
            if code is None:
                continue
            running.remove(job)
            if code == 0:
                log(f"{job['name']} finished")
            elif code == EXIT_RETRY:
                job["tries"] -= 1
                jobs.append(job)
                log(f"{job['name']} stopped cleanly and will resume")
            elif job["tries"] < 3:
                jobs.append(job)
                log(f"{job['name']} exited with {code}, retrying")
            else:
                failed.append(job["name"])
                log(f"{job['name']} failed for good with exit code {code}")
    summary = monitor.stop()
    log(f"{monitor_name} resources: {summary}")
    atomic_write_text(ROOT / "logs" / f"{monitor_name}_resources_summary.json", json.dumps(summary, indent=1))
    if stopped:
        return EXIT_RETRY
    return 1 if failed else 0


def pair_verdict(ours: list[float], reference: list[float]) -> dict:
    ours_mean, ref_mean = float(np.mean(ours)), float(np.mean(reference))
    relative = abs(ours_mean - ref_mean) / abs(ref_mean) if ref_mean else float("inf")
    within = relative <= 0.20
    inside = min(ours) <= ref_mean <= max(ours)
    return {
        "ours_per_seed": ours,
        "reference_per_seed": reference,
        "ours_mean": ours_mean,
        "reference_mean": ref_mean,
        "relative_difference": relative,
        "within_20_percent": within,
        "reference_mean_inside_our_range": inside,
        "passed": bool(within or inside),
    }


def report(learners=LEARNERS) -> dict | None:
    """The verdict on the pairs asked for. None if one of them has not finished yet."""
    verdicts = {}
    wall = {}
    for learner in learners:
        ours, reference = [], []
        wall[learner] = {"ours": None, "reference": []}
        for spec in our_specs(learner):
            run = RESULTS_RAW / OURS / spec.run_id
            if not (run / "record.json").exists():
                return None
            record = read_record(run)
            if record["status"] != "finished":
                raise RuntimeError(f"{spec.run_id}: {record['status']}: {str(record.get('reason'))[-400:]}")
            ours.append(final_return(*eval_curve(run)))
            wall[learner]["ours"] = record["wall_time_s"]
        for seed in SEEDS:
            run = reference_dir(learner, seed)
            if not (run / "record.json").exists():
                return None
            reference.append(final_return(*eval_curve(run)))
            wall[learner]["reference"].append(read_record(run)["wall_time_s"])
        verdicts[learner] = pair_verdict(ours, reference)
    complete = set(verdicts) == set(LEARNERS)
    result = {
        "task": TASK,
        "seeds": SEEDS,
        "budget": BUDGET,
        "rule": "within 20 percent of the reference mean, or reference mean inside the range of our 3 seeds",
        "pairs": verdicts,
        "wall_time_s": wall,
        "complete": complete,  # the gate as a whole is decided only when both pairs are in
        "passed": all(v["passed"] for v in verdicts.values()),
    }
    name = "gate_g1.json" if complete else f"gate_g1_{'_'.join(sorted(verdicts))}.json"
    atomic_write_text(AGGREGATED / name, json.dumps(result, indent=1))
    for learner, verdict in verdicts.items():
        print(learner, json.dumps(verdict))
    label = "GATE G1" if complete else f"GATE G1, pair {', '.join(verdicts)} only"
    print(f"{label}:", "PASS" if result["passed"] else "FAIL")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "report", "debug-sac"])
    parser.add_argument("--learners", nargs="+", default=list(LEARNERS), choices=list(LEARNERS))
    parser.add_argument("--join", action="store_true", help="debug-sac: first wait for a debug-sac already running")
    args = parser.parse_args()
    logger = get_logger("g1_gate")
    if args.action == "debug-sac":
        if args.join:
            wait_for_other_debug(logger.info)
        code = run_debug(logger.info)
        if code:
            return code
        result = report_debug()
        if result is None:
            print("The Gate G1 SAC debug runs are not complete yet.")
            return 1
        return 0 if result["pair"]["passed"] else 3
    if args.action == "run":
        code = run_all(logger.info, args.learners)
        if code:
            return code
    result = report(args.learners)
    if result is None:
        print("Gate G1 is not complete yet.")
        return 1
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
