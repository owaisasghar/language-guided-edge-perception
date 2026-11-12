"""Process CPU and GPU utilisation for benchmarks (no extra dependencies: os.times and nvidia-smi)."""

from __future__ import annotations

import os
import subprocess
import threading
import time

import numpy as np


class CpuMeter:
    """CPU time of this process (all threads) per wall-clock time, in % of one core (400 % = 4 cores busy)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        t = os.times()
        self._cpu0, self._wall0 = t.user + t.system, time.monotonic()

    def percent(self) -> float:
        t = os.times()
        wall = time.monotonic() - self._wall0
        return 100.0 * (t.user + t.system - self._cpu0) / wall if wall > 0 else 0.0


class GpuSampler:
    """Samples GPU utilisation and memory with ``nvidia-smi -lms`` in the background (GPU 0)."""

    def __init__(self, interval_ms: int = 500, gpu: int = 0):
        self.interval_ms, self.gpu = interval_ms, gpu
        self.util: list[float] = []
        self.mem_mib: list[float] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuSampler:
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--id={self.gpu}", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        for line in self._proc.stdout:
            try:
                u, m = (float(v) for v in line.split(","))
            except ValueError:
                continue
            self.util.append(u)
            self.mem_mib.append(m)

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(5)
        if self._thread:
            self._thread.join(2)

    def summary(self) -> dict | None:
        if not self.util:
            return None
        u = np.asarray(self.util)
        return {"samples": len(u), "util_percent_mean": round(float(u.mean()), 1),
                "util_percent_p90": round(float(np.percentile(u, 90)), 1),
                "device_memory_used_mib_max": round(max(self.mem_mib), 0)}
