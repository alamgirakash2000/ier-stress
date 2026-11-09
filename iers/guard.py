"""Keeps long runs from hanging the machine. It never changes what a run computes.

What it does:
- caps the GPU memory of each process, so a run gets a clean out-of-memory error instead of pushing
  Windows into paging video memory, which is what makes the whole machine stall
- pauses training while the GPU is too hot, RAM is nearly gone, or the commit charge is close to its
  limit, and resumes when things recover
- after a long pause, or when logs/STOP_REQUESTED exists, saves a checkpoint and exits with EXIT_RETRY
- lowers the process priority, so the desktop stays responsive

Pausing only sleeps between environment steps, so results do not depend on the guard.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from . import ROOT

EXIT_RETRY = 75  # the run stopped cleanly and should be started again later
# IERS_STOP_FILE moves the file, so that the test suite has its own and a real pause cannot fail a test.
STOP_FILE = Path(os.environ.get("IERS_STOP_FILE") or ROOT / "logs" / "STOP_REQUESTED")
MB = 2**20


@dataclass
class GuardLimits:
    gpu_process_cap_mb: int = 7000  # set IERS_GPU_CAP_MB to change it, for example on GPUs with more memory
    # Pause at least 3 C below the GPU's hardware slowdown point (ResourceGuard clamps the pause point to that).
    gpu_temp_pause_c: int = 88
    # Resume 4 C below the pause point. A wider gap can keep a paused process waiting while other processes keep
    # the GPU warm. The peak temperature is set by the pause point alone.
    gpu_temp_resume_c: int = 84
    ram_pause_mb: int = 700
    ram_resume_mb: int = 1100
    commit_pause_mb: int = 1000
    commit_resume_mb: int = 1500
    check_every_s: float = 5.0
    pause_poll_s: float = 3.0
    patience_s: float = 1800.0  # pause this long, then checkpoint and exit


def commit_available_mb() -> float | None:
    """Commit charge still available on Windows. A CUDA process commits far more than it touches."""
    if os.name != "nt":
        return None

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPageFile / MB


class GpuProbe:
    """Reads GPU memory and temperature through NVML. Every value is None if NVML is unavailable."""

    def __init__(self):
        self.nv = None
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # A machine state without NVML must not stop a run.
            self.nv = None

    def temperature_c(self) -> int | None:
        if self.nv is None:
            return None
        try:
            return self.nv.nvmlDeviceGetTemperature(self.handle, self.nv.NVML_TEMPERATURE_GPU)
        except Exception:
            return None

    def slowdown_c(self) -> int | None:
        if self.nv is None:
            return None
        try:
            return self.nv.nvmlDeviceGetTemperatureThreshold(self.handle, self.nv.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
        except Exception:
            return None

    def memory_mb(self) -> tuple[float, float] | None:
        """(free, total) GPU memory."""
        if self.nv is None:
            return None
        try:
            info = self.nv.nvmlDeviceGetMemoryInfo(self.handle)
            return info.free / MB, info.total / MB
        except Exception:
            return None


def lower_priority() -> None:
    try:
        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 5)
    except (psutil.Error, AttributeError):
        pass


def keep_awake() -> bool:
    """Ask Windows not to sleep on idle while this process runs.

    A sleeping machine freezes training. ES_SYSTEM_REQUIRED with ES_CONTINUOUS holds until the process exits,
    so normal sleep comes back by itself when no run is active. The screen may still turn off.
    """
    if os.name != "nt":
        return False
    es_continuous, es_system_required = 0x80000000, 0x00000001
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required))
    except (AttributeError, OSError):
        return False


def cap_gpu_memory(cap_mb: int) -> None:
    """Make this process raise an out-of-memory error instead of going past cap_mb."""
    import torch

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / MB
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_mb / total), 0)


class ResourceGuard:
    def __init__(self, limits: GuardLimits | None = None, log=print, stop_file: Path = STOP_FILE,
                 probe: GpuProbe | None = None, watch_gpu: bool = True):
        self.limits = limits or GuardLimits()
        self.log = log
        # A process that does not use the GPU adds nothing to its heat, so pausing it would only waste time.
        # IERS_GUARD_WATCH_GPU=0 turns the heat check off for test subprocesses, which check other things and
        # must not time out while the pipeline keeps the GPU hot. Real runs never set it.
        self.watch_gpu = watch_gpu and os.environ.get("IERS_GUARD_WATCH_GPU", "1") != "0"
        self.stop_file = Path(stop_file)
        self.probe = probe or GpuProbe()
        self.paused_s = 0.0
        self.must_exit = False
        self.exit_reason = ""
        self._next_check = 0.0
        slowdown = self.probe.slowdown_c()
        if slowdown:  # Stay clear of the temperature at which the GPU slows itself down.
            self.limits.gpu_temp_pause_c = min(self.limits.gpu_temp_pause_c, slowdown - 3)
            self.limits.gpu_temp_resume_c = min(self.limits.gpu_temp_resume_c, self.limits.gpu_temp_pause_c - 4)
        self.pauses = 0

    def problems(self, recovering: bool = False) -> list[str]:
        """What is wrong right now. While recovering, the higher resume limits apply."""
        lim = self.limits
        found = []
        temp = self.probe.temperature_c() if self.watch_gpu else None
        temp_limit = lim.gpu_temp_resume_c if recovering else lim.gpu_temp_pause_c
        if temp is not None and temp >= temp_limit:
            found.append(f"GPU at {temp} C, limit {temp_limit} C")
        ram = psutil.virtual_memory().available / MB
        ram_limit = lim.ram_resume_mb if recovering else lim.ram_pause_mb
        if ram < ram_limit:
            found.append(f"free RAM {ram:.0f} MB, limit {ram_limit} MB")
        commit = commit_available_mb()
        commit_limit = lim.commit_resume_mb if recovering else lim.commit_pause_mb
        if commit is not None and commit < commit_limit:
            found.append(f"available commit {commit:.0f} MB, limit {commit_limit} MB")
        return found

    def check(self) -> bool:
        """Call between environment steps. Returns True when the run should checkpoint and exit."""
        now = time.monotonic()
        if now < self._next_check:
            return False
        self._next_check = now + self.limits.check_every_s
        if self.stop_file.exists():
            self.must_exit = True
            self.exit_reason = f"{self.stop_file.name} exists"
            return True
        found = self.problems()
        if not found:
            return False
        # Heat pauses are routine under full compute, so only every 50th is logged. The total goes on the record.
        self.pauses += 1
        quiet = self.pauses % 50 != 1
        if not quiet:
            self.log(f"guard: pausing training, pause {self.pauses}, {self.paused_s:.0f} s so far: {'; '.join(found)}")
        start = time.monotonic()
        while True:
            time.sleep(self.limits.pause_poll_s)
            waited = time.monotonic() - start
            if self.stop_file.exists():
                self.must_exit = True
                self.exit_reason = f"{self.stop_file.name} exists"
                break
            found = self.problems(recovering=True)
            if not found:
                if not quiet:
                    self.log(f"guard: resuming after {waited:.0f} s")
                break
            if waited >= self.limits.patience_s:
                self.must_exit = True
                self.exit_reason = f"still unsafe after {waited:.0f} s: {'; '.join(found)}"
                self.log(f"guard: {self.exit_reason}")
                break
        self.paused_s += time.monotonic() - start
        self._next_check = time.monotonic() + self.limits.check_every_s
        return self.must_exit


def hold_run_lock(path: Path, log=print, poll_s: float = 15.0) -> None:
    """Make sure only one process works on a run, so two can never write the same checkpoint.

    If a supervisor dies, its workers live on. A new supervisor starts the same job again, and that new
    worker lands here and waits until the old one has exited. Then it carries on from what is on disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    announced = False
    while path.exists():
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            holder = psutil.Process(info["pid"])
            if holder.pid == os.getpid() or abs(holder.create_time() - info["created"]) >= 2.0:
                break
        except (json.JSONDecodeError, KeyError, psutil.Error, OSError):
            break
        if not announced:
            log(f"waiting for process {info['pid']}, which is still working on this run")
            announced = True
        time.sleep(poll_s)
    me = psutil.Process()
    path.write_text(json.dumps({"pid": me.pid, "created": me.create_time()}), encoding="utf-8")


PROCESS_COMMIT_MB = 2500.0  # measured for one training process that imports the CUDA build of torch


def wait_for_room(ram_mb: float, vram_mb: float = 0.0, log=print, poll_s: float = 15.0,
                  probe: GpuProbe | None = None, timeout_s: float | None = None,
                  ram_floor_mb: float = 800.0, vram_floor_mb: float = 500.0, commit_mb: float = PROCESS_COMMIT_MB,
                  commit_floor_mb: float = 1000.0) -> bool:
    """Block until a new process would fit. Returns False on timeout."""
    probe = probe or GpuProbe()
    start = time.monotonic()
    announced = False
    while True:
        free_ram = psutil.virtual_memory().available / MB
        memory = probe.memory_mb() if vram_mb else None
        free_vram = memory[0] if memory else None
        free_commit = commit_available_mb()
        ram_ok = free_ram - ram_mb >= ram_floor_mb
        vram_ok = free_vram is None or free_vram - vram_mb >= vram_floor_mb
        commit_ok = free_commit is None or free_commit - commit_mb >= commit_floor_mb
        if ram_ok and vram_ok and commit_ok and not STOP_FILE.exists():
            return True
        if not announced:
            log(f"waiting for room: need {ram_mb:.0f} MB RAM, {vram_mb:.0f} MB GPU memory, {commit_mb:.0f} MB commit. "
                f"Free: {free_ram:.0f} MB, {free_vram if free_vram is None else round(free_vram)} MB, "
                f"{free_commit if free_commit is None else round(free_commit)} MB")
            announced = True
        if timeout_s is not None and time.monotonic() - start > timeout_s:
            return False
        time.sleep(poll_s)


CROSSQ_CRITIC_WIDTH = 1024  # must match CrossQConfig.critic_hidden, which this torch-free module cannot import


def estimate_group_vram_mb(learner: str, S: int, obs_dim: int, act_dim: int, capacity: int, batch: int = 256,
                           crossq_width: int = CROSSQ_CRITIC_WIDTH) -> float:
    """GPU memory a group needs, checked against the measured S=5, 10, 20 SAC runs in logs/benchmark_results.json."""
    row = 2 * obs_dim + act_dim + 2
    replay = S * capacity * row * 4 / MB
    base = 700.0  # CUDA context and the desktop, from the benchmark
    if learner == "sac":
        critic = 2 * ((obs_dim + act_dim) * 256 + 256 * 256 + 256)
        actor = obs_dim * 256 + 256 * 256 + 2 * 256 * act_dim
        networks = S * (critic * 4 + actor * 3) * 4 / MB  # critic, its target and Adam's two moments
        activations = S * 2 * batch * 256 * 3 * 4 * 4 / MB
    elif learner == "crossq":
        q = crossq_width
        critic = 2 * ((obs_dim + act_dim) * q + q * q + q)
        actor = obs_dim * 256 + 256 * 256 + 2 * 256 * act_dim
        networks = S * (critic * 3 + actor * 3) * 4 / MB
        activations = S * 2 * 2 * batch * q * 3 * 3 * 4 / MB
    else:
        raise ValueError(f"Unknown learner: {learner}")
    return base + replay + networks + activations


def largest_group_that_fits(learner: str, wanted: int, obs_dim: int, act_dim: int, capacity: int,
                            cap_mb: float = GuardLimits.gpu_process_cap_mb,
                            crossq_width: int = CROSSQ_CRITIC_WIDTH) -> int:
    """The largest group size up to `wanted` whose estimated memory stays within the cap."""
    size = wanted
    while size > 1 and estimate_group_vram_mb(learner, size, obs_dim, act_dim, capacity,
                                              crossq_width=crossq_width) > cap_mb:
        size -= 1
    return size
