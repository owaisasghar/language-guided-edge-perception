#!/usr/bin/env python
"""Milestone 8 benchmark: scaling from 1 to 8 camera streams, and what each optimisation buys (RQ5, RQ6).

Each run starts the full edge application (``LiveSystem``: simulated RTSP cameras -> YOLO-World-M ->
ByteTrack -> CLIP Re-ID -> SQLite store) with N of the config's cameras and one variant of the settings,
waits ``warmup_s``, then measures for ``window_s``:

* processed frames per second per camera (exact: counted over the window) and in total;
* latency, frame decoded -> tracks ready, pooled over all cameras (p50 / p90 / p99);
* detector call time and batch size; Re-ID lag (capture -> Re-ID done) and dropped frames when asynchronous;
* GPU utilisation, memory, power, and GPU energy per processed frame; CPU of this process (the application),
  of the camera simulators (separate processes) and of the whole machine.

Usage:
    python scripts/benchmark_scaling.py [--config configs/milestone8_scaling.yaml] [--only VARIANT ...]
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from lgep.utils.config import deep_merge, load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging
from lgep.utils.sysmon import GpuSampler

log = logging.getLogger("benchmark_scaling")
REPO_ROOT = Path(__file__).resolve().parents[1]


def system_cpu() -> tuple[float, float]:
    """(busy, total) jiffies of the whole machine."""
    f = [int(v) for v in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
    idle = f[3] + f[4]
    return sum(f) - idle, sum(f)


def pct(v, q):
    return None if not len(v) else round(float(np.percentile(v, q)), 1)


def run_config(base: dict, variant: dict, streams: int, process_fps: float, store_path: Path) -> dict:
    cfg = deep_merge(copy.deepcopy(base), variant)
    cfg.pop("benchmark", None)
    cams = cfg["cameras"][:streams]
    ids = {c["camera_id"] for c in cams}
    cfg["cameras"] = cams
    cfg["simulation"]["cameras"] = [c for c in cfg["simulation"]["cameras"] if c["camera_id"] in ids]
    cfg["pipeline"]["process_fps"] = process_fps
    cfg["store"]["path"] = str(store_path.relative_to(REPO_ROOT))
    return cfg


def measure(cfg: dict, warmup: float, window: float) -> dict:
    import cv2
    import torch

    from lgep.api.runtime import LiveSystem

    cv2.setNumThreads(-1)  # a known state; every variant sets its thread count explicitly (applied after loading)
    system = LiveSystem(cfg, REPO_ROOT, simulate=True)
    res: dict = {"opencv_threads": None}
    try:
        system.start()
        res["opencv_threads"] = cv2.getNumThreads()
        end = time.monotonic() + 30
        while time.monotonic() < end and not all(c["state"] == "streaming" for c in system.manager.status().values()):
            time.sleep(0.2)
        time.sleep(warmup)
        pipe, worker = system.pipeline, system.reid_worker
        pipe.clear_stats()
        if worker is not None:
            worker.lag_ms.clear()
        torch.cuda.reset_peak_memory_stats()
        cams0 = {cid: rt.processed for cid, rt in pipe._cams.items()}
        read0 = {cid: c["frames_read"] for cid, c in system.manager.status().items()}
        reid0 = (system.reid_worker or system.reid).stats() if system.reid else {}
        sims0 = sum(s.cpu_seconds() or 0 for s in system.sims.values())
        t, s0 = os.times(), system_cpu()
        cpu0, w0 = t.user + t.system, time.monotonic()
        with GpuSampler(interval_ms=250) as gpu:
            time.sleep(window)
        t, s1 = os.times(), system_cpu()
        dt = time.monotonic() - w0
        st = system.status()
        lat = np.concatenate([np.fromiter(rt.latency_ms, float) for rt in pipe._cams.values()] or [np.zeros(0)])
        det = np.concatenate([np.fromiter(rt.detector_ms, float) for rt in pipe._cams.values()] or [np.zeros(0)])
        batch = np.concatenate([np.fromiter(rt.batch, float) for rt in pipe._cams.values()] or [np.zeros(0)])
        fps = {cid: (rt.processed - cams0.get(cid, 0)) / dt for cid, rt in pipe._cams.items()}
        read = {cid: (c["frames_read"] - read0.get(cid, 0)) / dt for cid, c in st["cameras"].items()}
        total = sum(fps.values())
        g = gpu.summary() or {}
        reid1 = (system.reid_worker or system.reid).stats() if system.reid else {}
        res.update({
            "window_s": round(dt, 1),
            "processed_fps_per_camera": {k: round(v, 2) for k, v in sorted(fps.items())},
            "processed_fps_mean": round(total / max(len(fps), 1), 2),
            "processed_fps_min": round(min(fps.values()), 2) if fps else None,
            "processed_fps_total": round(total, 1),
            "decoded_fps_mean": round(float(np.mean(list(read.values()))), 1) if read else None,
            "latency_ms_p50": pct(lat, 50), "latency_ms_p90": pct(lat, 90), "latency_ms_p99": pct(lat, 99),
            "detector_ms_p50": pct(det, 50), "batch_mean": round(float(batch.mean()), 2) if len(batch) else None,
            "gpu_util_percent_mean": g.get("util_percent_mean"), "gpu_memory_used_mib_max": g.get("device_memory_used_mib_max"),
            "gpu_power_w_mean": g.get("power_w_mean"),
            "gpu_energy_mj_per_frame": round(1000 * g["power_w_mean"] / total, 1) if g.get("power_w_mean") and total else None,
            "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "app_cpu_percent": round(100 * (t.user + t.system - cpu0) / dt, 1),
            "simulators_cpu_percent": round(100 * (sum(s.cpu_seconds() or 0 for s in system.sims.values()) - sims0) / dt, 1),
            "machine_cpu_busy_percent": round(100 * (s1[0] - s0[0]) / max(s1[1] - s0[1], 1), 1),
            "reid": {
                "crops_per_s": round((reid1.get("crops_embedded", 0) - reid0.get("crops_embedded", 0)) / dt, 1),
                "frames_dropped": reid1.get("dropped", 0) - reid0.get("dropped", 0) if "dropped" in reid1 else None,
                "frames_submitted": reid1.get("submitted", 0) - reid0.get("submitted", 0) if "submitted" in reid1 else None,
                "lag_ms_p50": reid1.get("lag_ms_p50"), "lag_ms_p90": reid1.get("lag_ms_p90"),
                "embed_ms_per_frame": reid1.get("embed_ms_per_frame"),
                "tracks_refound_after_end": reid1.get("tracks_refound_after_end"),
            },
            "store": {k: st["store"][k] for k in ("dropped", "failed")},
            "callback_ms_mean": st["pipeline"]["callback_ms_mean"],
            "camera_errors": sum(c.get("errors") or 0 for c in st["cameras"].values()),
        })
    finally:
        system.stop()
        del system
        gc.collect()
        torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone8_scaling.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="variants to run (default: the whole matrix)")
    ap.add_argument("--window", type=float, default=None)
    args = ap.parse_args()
    base = load_config(REPO_ROOT / args.config)
    b = base["benchmark"]
    warmup, window = b["warmup_s"], args.window or b["window_s"]
    out = REPO_ROOT / base["output_dir"] / f"scaling_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(base.get("log_level", "INFO"), out / "benchmark.log")
    for name in ("lgep.ingestion.reader", "lgep.ingestion.manager", "lgep.api.runtime", "lgep.database.store"):
        logging.getLogger(name).setLevel(logging.WARNING)  # keep the log to the measurements
    plan = [(r["variant"], n, f) for r in b["runs"] if not args.only or r["variant"] in args.only
            for n in r["streams"] for f in r["process_fps"]]
    log.info("%d runs of %.0f + %.0f s", len(plan), warmup, window)
    report = {"measured": True, "date": datetime.now().isoformat(timespec="seconds"), "config": args.config,
              "warmup_s": warmup, "window_s": window, "variants": b["variants"], "environment": collect_environment(),
              "runs": []}
    t_all = time.monotonic()
    for k, (variant, n, f) in enumerate(plan, 1):
        cfg = run_config(base, b["variants"][variant], n, f, out / f"run{k:02d}.sqlite")
        log.info("[%d/%d] %s, %d stream(s), %s fps per camera", k, len(plan), variant, n, f or "all")
        try:
            r = measure(cfg, warmup, window)
        except Exception:
            log.exception("run failed")
            r = {"error": True}
        r.update({"variant": variant, "streams": n, "process_fps": f})
        report["runs"].append(r)
        log.info("   -> %.2f fps/camera (min %.2f), total %.1f | latency p50 %s p90 %s ms | GPU %s %% %s W %s mJ/frame | "
                 "CPU app %s %% sims %s %% machine %s %% | Re-ID dropped %s lag p50 %s ms",
                 r.get("processed_fps_mean", 0), r.get("processed_fps_min") or 0, r.get("processed_fps_total", 0),
                 r.get("latency_ms_p50"), r.get("latency_ms_p90"), r.get("gpu_util_percent_mean"),
                 r.get("gpu_power_w_mean"), r.get("gpu_energy_mj_per_frame"), r.get("app_cpu_percent"),
                 r.get("simulators_cpu_percent"), r.get("machine_cpu_busy_percent"),
                 (r.get("reid") or {}).get("frames_dropped"), (r.get("reid") or {}).get("lag_ms_p50"))
        (out / "partial.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    report["seconds"] = round(time.monotonic() - t_all)
    bdir = REPO_ROOT / base["benchmark_dir"]
    bdir.mkdir(parents=True, exist_ok=True)
    target = bdir / "m8_scaling.json" if not args.only and not args.window else out / "m8_scaling_partial.json"
    target.write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("wrote %s (%.0f min)", target, report["seconds"] / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
