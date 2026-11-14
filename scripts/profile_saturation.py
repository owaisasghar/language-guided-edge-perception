#!/usr/bin/env python
"""Milestone 9 performance analysis: where does one process spend its time at saturation? (RQ5 follow-up)

Milestone 8 found that one process tops out at about 112-118 frames/s with the GPU only ~57 % busy. This starts
the same application (``configs/milestone8_scaling.yaml``, all optimisations, ``--streams`` simulated cameras,
every frame) and, over a measurement window, records:

* CPU time of every thread (from /proc, by native thread ID and the thread's name): which thread is saturated?
* the perception worker's wall time split: waiting for frames, the detector call (its pre-processing,
  GPU inference and post-processing as the detector reports them), tracking, and the result callback;
* GPU utilisation.

Run under py-spy (launched as its parent, no root needed) for a sampled profile of the Python code. With
``--first K`` the process takes cameras K+1 .. K+streams, so two processes can split 8 cameras (each starts
its own simulators, on their own ports) to test whether a second process lifts the ceiling.

Usage:
    python scripts/profile_saturation.py [--streams 8] [--window 40]
    python scripts/profile_saturation.py --streams 4 --first 0 & python scripts/profile_saturation.py --streams 4 --first 4
    py-spy record --rate 100 --format speedscope --threads -o outputs/milestone9/profile.json -- \\
        python scripts/profile_saturation.py --window 30
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from benchmark_scaling import run_config

from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.sysmon import GpuSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
TICK = os.sysconf("SC_CLK_TCK")


def thread_cpu() -> dict[int, float]:
    """CPU seconds per native thread ID of this process."""
    out = {}
    for d in Path("/proc/self/task").iterdir():
        try:
            f = (d / "stat").read_text().rsplit(")", 1)[1].split()
            out[int(d.name)] = (int(f[11]) + int(f[12])) / TICK
        except (FileNotFoundError, ProcessLookupError):
            pass
    return out


def group(name: str) -> str:
    if name.startswith("camera-"):
        return "camera readers (decode)"
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone8_scaling.yaml")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--first", type=int, default=0, help="index of the first camera of the config to use")
    ap.add_argument("--fps", type=float, default=0, help="per camera; 0 = every frame (saturation)")
    ap.add_argument("--warmup", type=float, default=15)
    ap.add_argument("--window", type=float, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    from lgep.api.runtime import LiveSystem

    base = load_config(REPO_ROOT / args.config)
    out_dir = REPO_ROOT / "outputs" / "milestone9" / f"profile_{datetime.now():%Y%m%d-%H%M%S}_cams{args.first + 1}-{args.first + args.streams}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base["cameras"] = base["cameras"][args.first:]  # run_config then takes the first --streams of these
    cfg = run_config(base, base["benchmark"]["variants"][args.variant], args.streams, args.fps, out_dir / "run.sqlite")
    system = LiveSystem(cfg, REPO_ROOT, simulate=True)

    # wrap the detector and the pipeline's stages to split the worker's wall time
    acc = defaultdict(float)
    det = system.detector
    orig_batch, orig_finish = det.detect_batch, system.pipeline._finish

    def detect_batch(images, phrases):
        t0 = time.perf_counter()
        res = orig_batch(images, phrases)
        acc["detector_call"] += time.perf_counter() - t0
        t = res[0].timings if res else None
        if t is not None:  # one batch: the timings are the batch's (the same object for every image)
            acc["detector_pre"] += t.preprocess_ms / 1e3
            acc["detector_infer"] += t.inference_ms / 1e3
            acc["detector_post"] += t.postprocess_ms / 1e3
        acc["batches"] += 1
        acc["frames"] += len(images)
        return res

    def finish(rt, frame, d, trk, t0, batch):
        t1 = time.perf_counter()
        orig_finish(rt, frame, d, trk, t0, batch)
        acc["finish_incl_callback"] += time.perf_counter() - t1
        acc["tracker"] += trk.tracker_ms / 1e3

    det.detect_batch = detect_batch
    system.pipeline._finish = finish
    wait = system.pipeline.manager.wait_for_frame

    def wait_for_frame(timeout):
        t0 = time.perf_counter()
        r = wait(timeout)
        acc["waiting_for_frames"] += time.perf_counter() - t0
        return r

    system.pipeline.manager.wait_for_frame = wait_for_frame
    try:
        system.start()
        end = time.monotonic() + 30
        while time.monotonic() < end and not all(c["state"] == "streaming" for c in system.manager.status().values()):
            time.sleep(0.2)
        time.sleep(args.warmup)
        names = {t.native_id: t.name for t in threading.enumerate()}
        acc.clear()
        processed0 = sum(rt.processed for rt in system.pipeline._cams.values())
        c0, w0 = thread_cpu(), time.perf_counter()
        p0 = os.times()
        with GpuSampler(interval_ms=250) as gpu:
            time.sleep(args.window)
        c1, dt = thread_cpu(), time.perf_counter() - w0
        p1 = os.times()
        processed = sum(rt.processed for rt in system.pipeline._cams.values()) - processed0
        snap = dict(acc)
        names.update({t.native_id: t.name for t in threading.enumerate()})
    finally:
        system.stop()
        torch.cuda.empty_cache()

    threads = defaultdict(float)
    for tid, cpu in c1.items():
        threads[group(names.get(tid, "other (native: PyTorch, OpenCV, CUDA, ...)"))] += cpu - c0.get(tid, 0.0)
    worker = next((n for n in names.values() if "perception" in n or n == "pipeline"), None)
    res = {
        "measured": True, "date": datetime.now().isoformat(timespec="seconds"), "config": args.config,
        "variant": args.variant, "streams": args.streams, "cameras": [c["camera_id"] for c in cfg["cameras"]],
        "process_fps": args.fps, "window_s": round(dt, 1),
        "processed_fps_total": round(processed / dt, 1), "batches_per_s": round(snap.get("batches", 0) / dt, 1),
        "mean_batch": round(snap.get("frames", 0) / max(snap.get("batches", 1), 1), 2),
        "process_cpu_percent": round(100 * ((p1.user + p1.system) - (p0.user + p0.system)) / dt, 1),
        "thread_cpu_percent": {k: round(100 * v / dt, 1) for k, v in sorted(threads.items(), key=lambda x: -x[1])},
        "worker_thread": worker,
        "worker_wall_percent": {k: round(100 * snap.get(k, 0) / dt, 1) for k in
                                ("waiting_for_frames", "detector_call", "detector_pre", "detector_infer", "detector_post",
                                 "tracker", "finish_incl_callback")},
        "gpu": gpu.summary(), "environment": collect_environment(),
    }
    w = res["worker_wall_percent"]
    # the rest of the worker loop: scheduling, frame selection, grouping detections by phrase, bookkeeping
    w["other_worker_python"] = round(100 - w["waiting_for_frames"] - w["detector_call"] - w["tracker"]
                                     - w["finish_incl_callback"], 1)
    path = Path(args.out) if args.out else out_dir / "profile.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("processed_fps_total", "mean_batch", "process_cpu_percent",
                                          "thread_cpu_percent", "worker_wall_percent")}, indent=1))
    print("gpu", {k: v for k, v in (res["gpu"] or {}).items() if "util" in k or "power" in k})
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
