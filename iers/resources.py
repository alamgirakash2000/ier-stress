"""Background sampling of GPU and CPU load, written to logs/resources_<experiment>.csv."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import psutil

from . import ROOT

FIELDS = [
    "time",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_power_w",
    "gpu_temp_c",
    "cpu_util_pct",
    "ram_used_mb",
    "ram_available_mb",
]


class ResourceMonitor:
    """Samples the machine every `interval` seconds on a daemon thread."""

    def __init__(self, experiment: str, interval: float = 60.0, path: Path | None = None):
        self.path = Path(path) if path else ROOT / "logs" / f"resources_{experiment}.csv"
        self.interval = interval
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # A failed NVML is reported through empty GPU columns.
            self._nvml = None

    def sample(self) -> dict:
        vm = psutil.virtual_memory()
        row = {
            "time": round(time.time(), 1),
            "gpu_util_pct": "",
            "gpu_mem_used_mb": "",
            "gpu_mem_total_mb": "",
            "gpu_power_w": "",
            "gpu_temp_c": "",
            "cpu_util_pct": psutil.cpu_percent(interval=None),
            "ram_used_mb": round((vm.total - vm.available) / 2**20),
            "ram_available_mb": round(vm.available / 2**20),
        }
        if self._handle is not None:
            nv = self._nvml
            util = nv.nvmlDeviceGetUtilizationRates(self._handle)
            mem = nv.nvmlDeviceGetMemoryInfo(self._handle)
            row["gpu_util_pct"] = util.gpu
            row["gpu_mem_used_mb"] = round(mem.used / 2**20)
            row["gpu_mem_total_mb"] = round(mem.total / 2**20)
            try:
                row["gpu_power_w"] = round(nv.nvmlDeviceGetPowerUsage(self._handle) / 1000, 1)
                row["gpu_temp_c"] = nv.nvmlDeviceGetTemperature(self._handle, nv.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
        return row

    def _loop(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            psutil.cpu_percent(interval=None)
            while not self._stop.wait(self.interval):
                row = self.sample()
                self.rows.append(row)
                writer.writerow(row)
                handle.flush()

    def start(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summary()

    def summary(self) -> dict:
        def mean(key):
            values = [r[key] for r in self.rows if r[key] != ""]
            return round(sum(values) / len(values), 1) if values else None

        def peak(key):
            values = [r[key] for r in self.rows if r[key] != ""]
            return max(values) if values else None

        return {
            "samples": len(self.rows),
            "gpu_util_mean": mean("gpu_util_pct"),
            "gpu_mem_mean_mb": mean("gpu_mem_used_mb"),
            "gpu_mem_peak_mb": peak("gpu_mem_used_mb"),
            "cpu_util_mean": mean("cpu_util_pct"),
            "ram_used_peak_mb": peak("ram_used_mb"),
            "ram_available_min_mb": min(
                (r["ram_available_mb"] for r in self.rows), default=None
            ),
        }
