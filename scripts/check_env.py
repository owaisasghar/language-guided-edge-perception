#!/usr/bin/env python
"""Verify the Python / PyTorch / CUDA environment before running any model.

Checks performed (each reported PASS / WARN / FAIL):
  1. package versions and the CUDA version PyTorch was built for
  2. driver visible through nvidia-smi
  3. a real CUDA context: allocation + fp32 and fp16 matmul (is_available() alone can lie)
  4. other compute processes on the GPU (they distort benchmarks)

Usage:
    python scripts/check_env.py [--json outputs/env_report.json]
Exit code is non-zero if CUDA cannot run kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from lgep.utils.env import collect_environment
from lgep.utils.gpu import gpu_compute_processes


def _status(level: str, msg: str) -> None:
    print(f"[{level:4s}] {msg}")


def cuda_kernel_test() -> dict:
    res: dict = {"ok": False}
    try:
        x = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        for dtype in (torch.float32, torch.float16):
            a = x.to(dtype)
            a @ a  # warm-up (cuBLAS handle, kernel selection)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                a @ a
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10
            res[f"matmul_2048_{str(dtype).split('.')[-1]}_tflops"] = round(2 * 2048**3 / dt / 1e12, 2)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001 - report any CUDA failure verbatim
        res["error"] = f"{type(e).__name__}: {e}".splitlines()[0]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, help="write the full report to this file")
    args = ap.parse_args()

    env = collect_environment()
    print(f"Python {env['python']} | {env['os']}")
    print(f"CPU    {env['cpu']} ({env['cpu_logical_cores']} threads), RAM {env['ram_gib']} GiB")
    for k, v in env["packages"].items():
        print(f"  {k:16s} {v}")
    print(f"PyTorch CUDA build {env['torch_cuda_build']}, cuDNN {env['cudnn']}")

    failed = False
    gpu = env.get("gpu")
    if gpu:
        _status("PASS", f"nvidia-smi: {gpu['name']} | driver {gpu['driver']} | {gpu['memory_total']} | "
                        f"cc {gpu['compute_capability']} | power limit {gpu['power_limit']}")
    else:
        _status("FAIL", "nvidia-smi not found or no GPU visible")
        failed = True

    # Always attempt a real kernel launch: when is_available() is False the launch attempt
    # surfaces the underlying driver error (e.g. "CUDA unknown error" = cuInit 999).
    test = cuda_kernel_test()
    env["cuda_kernel_test"] = test
    if test["ok"]:
        _status("PASS", f"CUDA kernels run: fp32 {test['matmul_2048_float32_tflops']} TFLOPS, "
                        f"fp16 {test['matmul_2048_float16_tflops']} TFLOPS (2048^2 matmul, rough)")
    else:
        failed = True
        _status("FAIL", f"CUDA cannot run kernels: {test.get('error')}")
        if "unknown error" in str(test.get("error", "")).lower() or "busy or unavailable" in str(test.get("error", "")):
            print("       Hint: check `grep -E 'NVRM: Xid' /var/log/kern.log`. On laptops an Xid 31/154 after "
                  "suspend/resume\n       ('Node Reboot Required') can only be cleared by a reboot. "
                  "See docs/milestone1.md#known-issues.")

    others = [p for p in gpu_compute_processes() if p["pid"] != os.getpid()]
    env["other_gpu_processes"] = others
    for p in others:
        _status("WARN", f"other GPU compute process: pid {p['pid']} {p['name']} ({p['used_mib']} MiB) "
                        "- stop it for clean benchmarks")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(env, indent=2))
        print(f"report written to {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
