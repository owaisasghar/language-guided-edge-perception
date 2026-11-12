"""Collect a reproducibility record of the software/hardware environment."""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
from datetime import datetime, timezone

import torch


def _pkg(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    return None


def _nvidia_smi(query: str) -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def collect_environment() -> dict:
    """Facts only, no probing that could crash: safe to call even when CUDA is broken."""
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_logical_cores": __import__("os").cpu_count(),
        "ram_gib": _ram_gib(),
        "packages": {p: _pkg(p) for p in ("torch", "torchvision", "transformers", "huggingface-hub",
                                          "tokenizers", "safetensors", "numpy", "pillow", "lgep", "ultralytics",
                                          "clip", "opencv-python", "pycocotools")},
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "torch_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    # enforced.power.limit: laptops report power.limit as [N/A] but expose the enforced cap
    smi = _nvidia_smi("name,driver_version,memory.total,compute_cap,enforced.power.limit")
    if smi:
        name, driver, mem, cc, power = [s.strip() for s in smi.split(",")]
        info["gpu"] = {"name": name, "driver": driver, "memory_total": mem, "compute_capability": cc,
                       "power_limit": power}
    return info
