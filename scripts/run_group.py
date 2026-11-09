"""Runs one group of seeds in this process. Launched by iers/grid.py."""

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import psutil  # noqa: E402
import torch  # noqa: E402

from iers.guard import (  # noqa: E402
    EXIT_RETRY,
    GuardLimits,
    ResourceGuard,
    cap_gpu_memory,
    hold_run_lock,
    keep_awake,
    lower_priority,
)
from iers.logging import get_logger  # noqa: E402
from iers.runner import GroupRunner, Hardware, RunSpec, TrainConfig, is_finished  # noqa: E402


def keep_one_core_free() -> None:
    """Keep logical CPU 0 free, so the machine stays responsive."""
    process = psutil.Process()
    cpus = process.cpu_affinity()
    if len(cpus) > 1:
        process.cpu_affinity([c for c in cpus if c != 0])


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    keep_one_core_free()
    lower_priority()
    keep_awake()
    specs = [RunSpec(**spec) for spec in job["specs"]]
    logger = get_logger(specs[0].experiment, console=job.get("console", True))
    try:
        hold_run_lock(Path(sys.argv[1]).with_suffix(".lock"), log=logger.info)
        if all(is_finished(spec, Path(job["results_root"])) for spec in specs):
            logger.info("all runs of this job are already finished")
            return 0
        hw = Hardware(**job["hardware"])
        if hw.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
            cap_gpu_memory(job.get("gpu_cap_mb", GuardLimits.gpu_process_cap_mb))
        runner = GroupRunner(specs, TrainConfig(**job["train"]), hw, logger=logger,
                             results_root=Path(job["results_root"]), ckpt_root=Path(job["ckpt_root"]))
        guard = ResourceGuard(log=logger.info, watch_gpu="cuda" in (hw.device, hw.replay_device))

        def tick(_t: int) -> bool:
            stop = guard.check()
            runner.paused_s = runner.paused_s_before + guard.paused_s
            if stop:
                logger.info("guard: stopping group %s: %s", runner.group_id, guard.exit_reason)
                runner.checkpoint_on_stop = True
            return stop

        runner.run(tick=tick)
        if guard.must_exit:
            return EXIT_RETRY
    except Exception:
        logger.error("group crashed:\n%s", traceback.format_exc())
        Path(sys.argv[1]).with_suffix(".crash.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
