#!/usr/bin/env python
"""Long-run stability of the server: does throughput hold as tracks and identities accumulate? (Milestone 8)

Starts scripts/serve.py (4 simulated cameras, Milestone 7 configuration, Re-ID in the worker) as its own
process, once per setting, and samples every ``--every`` seconds for ``--minutes``: processed frames per second
(all cameras), median per-camera latency, process CPU (from /proc, over each interval), identities so far.

Settings compared: OpenBLAS's default thread pool (one thread per logical core) and one BLAS thread. The
identity manager's appearance step is a matrix-vector product that grows with every identity; OpenBLAS splits it
over its threads once it is large enough, and the threads spin.

Usage:
    python scripts/benchmark_long_run.py [--minutes 10]
"""

from __future__ import annotations

import os

import argparse
import json
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
TICK = os.sysconf("SC_CLK_TCK")
SETTINGS = {"openblas_default": str(os.cpu_count()), "openblas_1_thread": "1"}


def cpu_seconds(pid: int) -> float:
    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / TICK


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(label: str, blas_threads: str, minutes: float, every: float, out: Path) -> list[dict]:
    port = free_port()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "OPENBLAS_NUM_THREADS": blas_threads}
    env.pop("PYTHONPATH", None)
    db = out / f"{label}.sqlite"
    proc = subprocess.Popen([sys.executable, "scripts/serve.py", "--simulate", "--override", f"server.port={port}",
                             f"store.path={db.relative_to(REPO_ROOT)}", f"output_dir={out.relative_to(REPO_ROOT)}"],
                            cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    samples: list[dict] = []
    try:
        for _ in range(180):
            try:
                if httpx.get(base + "/api/health", timeout=2).json().get("live"):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        t0 = time.monotonic()
        c_prev, t_prev = cpu_seconds(proc.pid), time.monotonic()
        while time.monotonic() - t0 < minutes * 60:
            time.sleep(every)
            st = httpx.get(base + "/api/status", timeout=10).json()
            c, t = cpu_seconds(proc.pid), time.monotonic()
            cams = st["cameras"].values()
            lat = sorted(c["latency_ms_p50"] for c in cams if c.get("latency_ms_p50") is not None)
            samples.append({"t_s": round(t - t0, 1), "uptime_s": st["uptime_s"],
                            "fps_total": round(sum(c.get("processed_fps") or 0 for c in cams), 2),
                            "latency_ms_p50_median_camera": lat[len(lat) // 2] if lat else None,
                            "cpu_percent": round(100 * (c - c_prev) / (t - t_prev), 1),
                            "global_identities": (st.get("reid") or {}).get("global_identities"),
                            "threads": len(os.listdir(f"/proc/{proc.pid}/task"))})
            c_prev, t_prev = c, t
            s = samples[-1]
            print(f"{label:18s} {s['t_s']:6.0f} s  {s['fps_total']:5.1f} fps  CPU {s['cpu_percent']:6.1f} %  "
                  f"latency {s['latency_ms_p50_median_camera']} ms  identities {s['global_identities']}", flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--every", type=float, default=15)
    args = ap.parse_args()
    out = REPO_ROOT / "outputs" / "milestone8" / f"long_run_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    res = {"measured": True, "date": datetime.now().isoformat(timespec="seconds"), "minutes": args.minutes,
           "setup": "scripts/serve.py --simulate (configs/milestone7_server.yaml: 4 MOT17 cameras, YOLO-World-M FP32, "
                    "Re-ID in the perception worker, 10 fps per camera target)", "runs": {}}
    for label, threads in SETTINGS.items():
        res["runs"][label] = {"OPENBLAS_NUM_THREADS": threads, "samples": run(label, threads, args.minutes, args.every, out)}
    target = REPO_ROOT / "benchmarks" / "milestone8" / "m8_long_run.json"
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
