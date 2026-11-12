#!/usr/bin/env python
"""Milestone 4 measurements on simulated RTSP cameras (MOT17 sequences, 1920x1080, 30 fps, H.264).

  A  ingestion only: 1 / 2 / 4 cameras; decode threads 1 vs FFmpeg default -> input fps, process CPU
  B  full pipeline (detector + ByteTrack): 1 / 2 / 4 cameras at 10 fps per camera and as fast as possible
     -> processed fps, throughput, decode-to-tracks latency, CPU, GPU
  C  outages with 4 cameras running: one camera stopped and restarted, one frozen (silent) and resumed
     -> detection and recovery times, effect on the other cameras, tracker reset, ID continuity
  D  decoder delay: frames FFmpeg still delivers after the camera freezes = frames held in the decoder

Writes benchmarks/milestone4/m4_ingestion.json and README.md. Takes about 8 minutes.

Usage:
    python scripts/benchmark_ingestion.py
    python scripts/benchmark_ingestion.py --only A D --seconds 10     # quick partial run (not published)
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from run_cameras import build_pipeline_config

from lgep.ingestion import CameraState, StreamManager, cameras_from_config, open_capture
from lgep.ingestion.simulate import gstreamer_rtsp_available, simulated_cameras
from lgep.perception.factory import build_detector
from lgep.pipeline import PerceptionPipeline
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.gpu import gpu_compute_processes, memory_snapshot, reset_peak
from lgep.utils.logging import setup_logging
from lgep.utils.sysmon import CpuMeter, GpuSampler

log = logging.getLogger("benchmark_ingestion")
REPO_ROOT = Path(__file__).resolve().parents[1]


def wait_until(cond, timeout: float, step: float = 0.01) -> float | None:
    """Seconds until ``cond()`` held, or None on timeout."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return time.monotonic() - t0
        time.sleep(step)
    return None


def server_cpu(sims, before: dict, wall: float) -> float | None:
    vals = [s.cpu_seconds() for s in sims]
    if any(v is None for v in vals) or wall <= 0:
        return None
    return round(100 * sum(v - before[i] for i, v in enumerate(vals)) / wall, 1)


def ingest_only(cams, sims, seconds: float) -> dict:
    mgr = StreamManager(cams)
    with mgr:
        wait_until(lambda: all(s["frames_read"] > 30 for s in mgr.status().values()), 20)
        cpu, before, t0 = CpuMeter(), [s.cpu_seconds() for s in sims], time.monotonic()
        read0 = {c: s["frames_read"] for c, s in mgr.status().items()}
        time.sleep(seconds)
        wall = time.monotonic() - t0
        st = mgr.status()
        res = {"cameras": len(cams), "decode_threads": cams[0].decode_threads, "seconds": round(wall, 1),
               "input_fps_per_camera": {c: round((s["frames_read"] - read0[c]) / wall, 2) for c, s in st.items()},
               "process_cpu_percent": round(cpu.percent(), 1),
               "simulated_cameras_cpu_percent": server_cpu(sims, before, wall)}
    log.info("A %s", res)
    return res


def run_pipeline(cams, detector, pcfg, seconds: float, warmup: float = 5.0) -> dict:
    mgr = StreamManager(cams)
    with mgr:
        wait_until(lambda: all(s["frames_read"] > 10 for s in mgr.status().values()), 20)
        stop = threading.Event()
        warm = PerceptionPipeline(mgr, detector, pcfg)
        t = threading.Thread(target=warm.run, kwargs={"stop": stop, "duration_s": warmup})
        t.start()
        t.join()
        pipe = PerceptionPipeline(mgr, detector, pcfg)  # fresh statistics after the warm-up
        reset_peak(detector.device)
        cpu, t0 = CpuMeter(), time.monotonic()
        read0 = {c: s["frames_read"] for c, s in mgr.status().items()}
        with GpuSampler() as gpu:
            pipe.run(stop=threading.Event(), duration_s=seconds)
        wall = time.monotonic() - t0
        st, ps = mgr.status(), pipe.stats()
        mem = memory_snapshot(detector.device)
    per = ps["cameras"]
    lat = [v["latency_ms_p50"] for v in per.values()]
    res = {"cameras": len(cams), "process_fps_target": pcfg.process_fps, "seconds": round(wall, 1),
           "processed_fps_per_camera": {c: round(v["processed"] / wall, 2) for c, v in per.items()},
           "throughput_fps": round(sum(v["processed"] for v in per.values()) / wall, 2),
           "input_fps_per_camera": {c: round((s["frames_read"] - read0[c]) / wall, 2) for c, s in st.items()},
           "latency_ms_p50_per_camera": {c: v["latency_ms_p50"] for c, v in per.items()},
           "latency_ms_p90_per_camera": {c: v["latency_ms_p90"] for c, v in per.items()},
           "latency_ms_p50_median": round(statistics.median(lat), 1),
           "wait_ms_p50_per_camera": {c: v["wait_ms_p50"] for c, v in per.items()},
           "detector_ms_p50_per_camera": {c: v["detector_ms_p50"] for c, v in per.items()},
           "tracker_ms_p50_per_camera": {c: v["tracker_ms_p50"] for c, v in per.items()},
           "process_cpu_percent": round(cpu.percent(), 1), "gpu": gpu.summary(),
           "detector_peak_allocated_mib": None if mem is None else round(mem.peak_allocated_mib, 1)}
    log.info("B %d cams @ %s fps: throughput %.1f fps, latency p50 %.1f ms, CPU %.0f %%, GPU %s", len(cams),
             pcfg.process_fps, res["throughput_fps"], res["latency_ms_p50_median"], res["process_cpu_percent"],
             res["gpu"])
    return res


def outages(cams, sims_by_id, detector, pcfg) -> dict:
    """cam02 is stopped at 10 s and restarted at 25 s; cam03 is frozen at 35 s and resumed at 45 s."""
    mgr = StreamManager(cams)
    events, ids_by_epoch = {}, {}

    def on_result(frame, det, trk):
        if frame.camera_id == "cam02":
            ids_by_epoch.setdefault(frame.epoch, set()).update(t.local_id for t in trk.tracks)

    pipe = PerceptionPipeline(mgr, detector, pcfg, on_result=on_result)
    stop = threading.Event()
    samples = []  # (t, processed per camera) once per second
    with mgr:
        wait_until(lambda: all(s["frames_read"] > 10 for s in mgr.status().values()), 20)
        worker = threading.Thread(target=pipe.run, kwargs={"stop": stop})
        worker.start()
        t0 = time.monotonic()

        def sampler():
            while not stop.is_set():
                samples.append((time.monotonic() - t0, {c: v["processed"] for c, v in pipe.stats()["cameras"].items()}))
                stop.wait(1.0)

        threading.Thread(target=sampler, daemon=True).start()
        r2, r3 = mgr.reader("cam02"), mgr.reader("cam03")
        time.sleep(10)
        sims_by_id["cam02"].stop()
        t_stop = time.monotonic()
        events["cam02_stop_detected_s"] = wait_until(lambda: r2.state != CameraState.STREAMING, 15)
        time.sleep(max(0.0, 25 - (time.monotonic() - t0)))
        sims_by_id["cam02"].start()
        t_back = time.monotonic()
        events["cam02_outage_s"] = round(t_back - t_stop, 1)
        events["cam02_streaming_again_s"] = wait_until(lambda: r2.epoch >= 2 and r2.state == CameraState.STREAMING, 40)
        events["cam02_processed_again_s"] = wait_until(lambda: pipe.stats()["cameras"]["cam02"]["epoch"] == 2, 40)
        time.sleep(max(0.0, 35 - (time.monotonic() - t0)))
        sims_by_id["cam03"].freeze()
        t_freeze = time.monotonic()
        events["cam03_stalled_flag_s"] = wait_until(lambda: r3.stats()["stalled"], 15)
        events["cam03_freeze_detected_s"] = wait_until(lambda: r3.state != CameraState.STREAMING, 20)
        time.sleep(max(0.0, 45 - (time.monotonic() - t0)))
        sims_by_id["cam03"].resume()
        events["cam03_streaming_again_s"] = wait_until(lambda: r3.epoch >= 2 and r3.state == CameraState.STREAMING, 40)
        time.sleep(max(0.0, 60 - (time.monotonic() - t0)))
        stop.set()
        worker.join(10)
        st, ps = mgr.status(), pipe.stats()
    for k in events:
        if isinstance(events[k], float):
            events[k] = round(events[k], 2)
    t_rel_stop = t_stop - t0

    def rate(cam, a, b):  # processed fps of `cam` between seconds a and b
        pts = [(t, d.get(cam, 0)) for t, d in samples if a <= t <= b]
        return round((pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]), 2) if len(pts) > 1 else None

    others = [c for c in ("cam01", "cam04")]
    res = {"events": events,
           "processed_fps_before_outage": {c: rate(c, 2, t_rel_stop) for c in others},
           "processed_fps_during_cam02_outage": {c: rate(c, t_rel_stop + 1, 24) for c in others},
           "processed_fps_during_cam03_freeze": {c: rate(c, 36, 44) for c in others},
           "cam02": {"tracker_resets": ps["cameras"]["cam02"]["tracker_resets"], "connects": st["cam02"]["connects"],
                     "max_local_id_epoch1": max(ids_by_epoch.get(1, {0})),
                     "min_local_id_epoch2": min(ids_by_epoch.get(2, {0})),
                     "retry_waits_s": list(mgr.reader("cam02").retry_waits_s)},
           "cam03": {"tracker_resets": ps["cameras"]["cam03"]["tracker_resets"], "connects": st["cam03"]["connects"]}}
    log.info("C %s", res)
    return res


def decoder_delay(sim, base_cam, threads: int, settle_frames: int = 60) -> dict:
    cam = replace(base_cam, decode_threads=threads, read_timeout_s=2.0)
    cap = open_capture(cam)
    for _ in range(settle_frames):
        cap.read()
    sim.freeze()
    t0, held = time.monotonic(), 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        held += 1
    failed_after = time.monotonic() - t0
    cap.release()
    sim.resume()
    res = {"decode_threads": threads or "ffmpeg default", "frames_held": held,
           "added_latency_ms_at_30fps": round(1000 * held / 30, 1), "read_failed_after_s": round(failed_after, 2)}
    log.info("D %s", res)
    return res


def write_report(res: dict, path: Path) -> None:
    env = res["environment"]
    L = ["# Milestone 4 benchmark results — multi-camera RTSP ingestion", "",
         f"**Actual measurements**, {env['timestamp_utc'][:10]}, generated by `scripts/benchmark_ingestion.py`.", "",
         f"Hardware: {env['gpu']['name']} (power limit {env['gpu']['power_limit']}), {env['cpu']} "
         f"({env['cpu_logical_cores']} logical cores). Simulated cameras: local GStreamer RTSP servers looping "
         "MOT17-02/-04/-09/-11, 1920×1080, 30 fps, H.264 (x264 ultrafast, 4 Mbit/s), RTSP over TCP on 127.0.0.1. "
         f"Detector: `{res['model']['model_id']}` (FP32, batch 1), prompt `person`; ByteTrack on the CPU.", ""]
    if "A" in res:
        L += ["## A. Ingestion only (decode + RGB conversion, no detector)", "",
              "| cameras | decode threads | input fps per camera (min–max) | process CPU (% of one core) | simulated cameras' CPU |",
              "|---|---|---|---|---|"]
        for r in res["A"]:
            f = list(r["input_fps_per_camera"].values())
            L.append(f"| {r['cameras']} | {r['decode_threads'] or 'FFmpeg default'} | {min(f):.1f}–{max(f):.1f} | "
                     f"{r['process_cpu_percent']:.0f} % | {r['simulated_cameras_cpu_percent']} % |")
        L.append("")
    if "B" in res:
        L += ["## B. Full pipeline: RTSP → detector → ByteTrack", "",
              "| cameras | target fps per camera | processed fps per camera (min–max) | throughput fps | "
              "latency p50 / p90 ms (median over cameras) | detector ms p50 | process CPU | GPU util mean | detector peak MiB |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in res["B"]:
            f = list(r["processed_fps_per_camera"].values())
            p90 = statistics.median(v for v in r["latency_ms_p90_per_camera"].values() if v is not None)
            det = statistics.median(v for v in r["detector_ms_p50_per_camera"].values() if v is not None)
            tgt = r["process_fps_target"] or "max"
            L.append(f"| {r['cameras']} | {tgt} | {min(f):.1f}–{max(f):.1f} | {r['throughput_fps']:.1f} | "
                     f"{r['latency_ms_p50_median']:.1f} / {p90:.1f} | {det:.1f} | {r['process_cpu_percent']:.0f} % | "
                     f"{(r['gpu'] or {}).get('util_percent_mean', '–')} % | {r['detector_peak_allocated_mib']} |")
        L += ["", "Latency = from the moment a decoded frame is available to the moment its tracks are ready "
              "(waiting + detector + ByteTrack). It excludes network, encoding and the decoder's own buffering "
              "(see D).", ""]
    if "C" in res:
        c, e = res["C"], res["C"]["events"]
        L += ["## C. Outages with 4 cameras running (pipeline at 10 fps per camera)", "",
              "| event | measured |", "|---|---|",
              f"| cam02 server stopped → reader notices | {e['cam02_stop_detected_s']} s |",
              f"| cam02 down for | {e['cam02_outage_s']} s |",
              f"| cam02 server back → streaming again | {e['cam02_streaming_again_s']} s |",
              f"| cam02 streaming again → first frame of the new epoch processed | {e['cam02_processed_again_s']} s |",
              f"| cam02 retry waits (backoff) | {', '.join(f'{w:.2f}' for w in c['cam02']['retry_waits_s'])} s |",
              f"| cam02 tracker resets / highest local ID before / lowest after | {c['cam02']['tracker_resets']} / "
              f"{c['cam02']['max_local_id_epoch1']} / {c['cam02']['min_local_id_epoch2']} |",
              f"| cam03 frozen (silent) → `stalled` flag | {e['cam03_stalled_flag_s']} s |",
              f"| cam03 frozen → reader drops the connection | {e['cam03_freeze_detected_s']} s |",
              f"| cam03 resumed → streaming again | {e['cam03_streaming_again_s']} s |",
              f"| cam01 / cam04 processed fps before / during cam02 outage / during cam03 freeze | "
              + " ; ".join(f"{k}: {c['processed_fps_before_outage'][k]} / {c['processed_fps_during_cam02_outage'][k]} / "
                           f"{c['processed_fps_during_cam03_freeze'][k]}" for k in ("cam01", "cam04")) + " |", ""]
    if "D" in res:
        L += ["## D. Frames held in the FFmpeg decoder (camera frozen after 60 frames)", "",
              "| decode threads | frames still delivered after the freeze | hidden latency at 30 fps | read failed after |",
              "|---|---|---|---|"]
        for r in res["D"]:
            L.append(f"| {r['decode_threads']} | {r['frames_held']} | {r['added_latency_ms_at_30fps']} ms | "
                     f"{r['read_failed_after_s']} s (read timeout 2 s) |")
        L.append("")
    L.append("Reproduce: `python scripts/benchmark_ingestion.py` (needs the system GStreamer RTSP server; "
             "see docs/installation.md).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone4_cameras.yaml")
    ap.add_argument("--only", nargs="*", default=["A", "B", "C", "D"])
    ap.add_argument("--seconds", type=float, default=None, help="measurement length per A/B run (default 20 / 30)")
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    out = REPO_ROOT / cfg["benchmark_dir"]
    out.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO")
    if not gstreamer_rtsp_available():
        log.error("Needs the system GStreamer RTSP server (docs/installation.md)")
        return 1
    sims = simulated_cameras(cfg)
    cams = cameras_from_config(cfg)
    res = {"measured": True, "environment": collect_environment(), "other_gpu_processes": gpu_compute_processes(),
           "config": cfg}
    try:
        for s in sims.values():
            s.start()
        pcfg = build_pipeline_config(cfg)
        detector = build_detector(cfg["models"][cfg["pipeline"]["model"]], cfg.get("device", "auto"), base_dir=REPO_ROOT)
        for _ in range(3):
            detector.detect(np.zeros((1080, 1920, 3), np.uint8), pcfg.prompt)
        res["model"] = detector.describe()
        sim_list = list(sims.values())
        if "A" in args.only:
            sec = args.seconds or 20
            res["A"] = [ingest_only(cams[:n], sim_list[:n], sec) for n in (1, 2, 4)]
            res["A"].append(ingest_only([replace(c, decode_threads=0) for c in cams], sim_list, sec))
        if "B" in args.only:
            sec = args.seconds or 30
            res["B"] = [run_pipeline(cams[:n], detector, replace(pcfg, process_fps=fps), sec)
                        for fps in (10, 0) for n in (1, 2, 4)]
        if "C" in args.only:
            res["C"] = outages(cams, sims, detector, pcfg)
        if "D" in args.only:
            res["D"] = [decoder_delay(sims["cam03"], cams[2], t) for t in (1, 2, 4, 0)]
    except Exception:
        log.exception("benchmark failed")
        return 1
    finally:
        for s in sims.values():
            s.stop()
    full = set(args.only) == {"A", "B", "C", "D"} and args.seconds is None
    name = "m4_ingestion" if full else "m4_ingestion_partial"
    (out / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(res, out / ("README.md" if full else f"{name}.md"))
    log.info("Wrote %s", (out / f"{name}.json").relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
