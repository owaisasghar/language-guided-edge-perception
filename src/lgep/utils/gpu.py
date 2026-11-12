"""Device selection, synchronised timing and GPU-memory measurement helpers."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)
MIB = 1024**2

_DTYPES = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


class CudaUnavailableError(RuntimeError):
    pass


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name.lower()]
    except KeyError as e:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(_DTYPES)}") from e


def resolve_device(name: str = "auto") -> torch.device:
    """'auto' -> cuda if usable else cpu. An explicit 'cuda' fails loudly instead of silently falling back."""
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    usable = cuda_usable()
    if name == "auto":
        if not usable:
            log.warning("CUDA not usable; falling back to CPU (run scripts/check_env.py for details)")
        return torch.device("cuda" if usable else "cpu")
    if name.startswith("cuda"):
        if not usable:
            raise CudaUnavailableError(
                "device=cuda requested but CUDA is not usable. Run `python scripts/check_env.py` to diagnose "
                "(on laptops a driver Xid after suspend/resume needs a reboot)."
            )
        return torch.device(name)
    raise ValueError(f"Unknown device {name!r}")


def cuda_usable() -> bool:
    """True only if a CUDA context can actually be created (is_available() alone is not enough)."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        return True
    except Exception as e:  # CUDA errors surface as RuntimeError / AcceleratorError
        log.debug("CUDA probe failed: %s", e)
        return False


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(device: torch.device, out: dict, key: str):
    """Store elapsed wall-clock ms in out[key]; synchronises the GPU at both ends."""
    synchronize(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        out[key] = (time.perf_counter() - t0) * 1e3


@dataclass
class MemorySnapshot:
    allocated_mib: float
    reserved_mib: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    device_used_mib: float  # whole device, all processes (from cudaMemGetInfo)
    process_used_mib: float | None  # this process as seen by nvidia-smi (includes CUDA context)

    def to_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(device: torch.device) -> MemorySnapshot | None:
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return MemorySnapshot(
        allocated_mib=torch.cuda.memory_allocated(device) / MIB,
        reserved_mib=torch.cuda.memory_reserved(device) / MIB,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / MIB,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / MIB,
        device_used_mib=(total - free) / MIB,
        process_used_mib=nvidia_smi_process_memory_mib(os.getpid()),
    )


def gpu_compute_processes() -> list[dict]:
    """Compute processes on the GPU according to nvidia-smi (empty list if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            procs.append({"pid": int(parts[0]), "name": parts[1],
                          "used_mib": float(parts[2]) if parts[2].replace(".", "").isdigit() else None})
    return procs


def nvidia_smi_process_memory_mib(pid: int) -> float | None:
    for p in gpu_compute_processes():
        if p["pid"] == pid:
            return p["used_mib"]
    return None
