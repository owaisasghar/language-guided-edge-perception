#!/usr/bin/env python
"""Tables for benchmarks/milestone8/README.md from the Milestone 8 result files (no measurement here).

Inputs: m8_scaling.json (scripts/benchmark_scaling.py), m8_clip_fp16.json (scripts/check_clip_fp16.py),
m8_coco_fp16_lb_minimal_yoloworld_m.json (scripts/eval_coco.py, FP16) and Milestone 2's FP32 counterpart.

Usage:
    python scripts/summarize_milestone8.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
B8 = REPO_ROOT / "benchmarks" / "milestone8"
NAMES = {
    "m7": "Milestone 7 settings",
    "m7_async": "+ Re-ID in its own thread",
    "m7_async_batch": "+ batching across cameras",
    "full": "+ FP16 (all optimisations)",
    "full_no_async": "all but the Re-ID thread",
    "full_no_batch": "all but batching",
    "m7_cv16": "Milestone 7 settings, OpenCV default threads",
    "full_cv16": "all optimisations, OpenCV default threads",
}


def f(v, nd=1, unit=""):
    return "–" if v is None else f"{v:.{nd}f}{unit}"


def row(r: dict) -> list[str]:
    reid = r.get("reid") or {}
    drop = reid.get("frames_dropped")
    sub = reid.get("frames_submitted")
    drop_s = "–" if drop is None else (f"{100 * drop / sub:.0f} %" if sub else "0 %")
    return [f(r.get("processed_fps_mean"), 2), f(r.get("processed_fps_min"), 2), f(r.get("processed_fps_total"), 1),
            f"{f(r.get('latency_ms_p50'))} / {f(r.get('latency_ms_p90'))} / {f(r.get('latency_ms_p99'))}",
            f(r.get("batch_mean"), 2), f(r.get("gpu_util_percent_mean"), 0, " %"), f(r.get("gpu_power_w_mean"), 1, " W"),
            f(r.get("gpu_energy_mj_per_frame"), 0), f(r.get("app_cpu_percent"), 0, " %"),
            f(r.get("machine_cpu_busy_percent"), 0, " %"), drop_s, f(reid.get("lag_ms_p50"), 0)]


HEAD = ("| {first} | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | "
        "GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |")
SEP = "|---" * 13 + "|"


def main() -> int:
    s = json.loads((B8 / "m8_scaling.json").read_text())
    runs = [r for r in s["runs"] if not r.get("error")]
    get = {(r["variant"], r["streams"], r["process_fps"]): r for r in runs}
    out = ["# Milestone 8 benchmark results — scaling and optimisation", "",
           f"**Actual measurements**, {s['date'][:10]} (`scripts/benchmark_scaling.py`, `{s['config']}`): the full edge "
           "application (simulated RTSP cameras, 1080p30 H.264 -> YOLO-World-M -> ByteTrack -> CLIP Re-ID -> SQLite), "
           f"each run a fresh start, {s['warmup_s']:.0f} s warm-up, {s['window_s']:.0f} s measured. RTX 4060 Laptop GPU, "
           "Ryzen 7 8845HS (8 cores / 16 threads). Latency = frame decoded -> tracks ready. GPU energy = mean GPU board "
           "power / processed frames per second. Machine CPU includes the camera simulators.", ""]
    out += ["## Scaling (RQ5): 1, 2, 4, 8 streams, 10 fps per camera target", "",
            HEAD.format(first="settings, streams"), SEP]
    for v in ("m7", "full"):
        for n in (1, 2, 4, 8):
            r = get.get((v, n, 10))
            if r:
                out.append(f"| {NAMES[v] if v == 'm7' else 'all optimisations'}, {n} | " + " | ".join(row(r)) + " |")
    out += ["", "## What each optimisation contributes (RQ6), 8 streams, 10 fps per camera target", "",
            HEAD.format(first="settings"), SEP]
    for v in ("m7", "m7_async", "m7_async_batch", "full", "full_no_async", "full_no_batch", "m7_cv16", "full_cv16"):
        r = get.get((v, 8, 10))
        if r:
            out.append(f"| {NAMES[v]} | " + " | ".join(row(r)) + " |")
    out += ["", "## Frame sampling (RQ6), all optimisations, 8 streams", "", HEAD.format(first="frames per second per camera"), SEP]
    for fps in (2, 5, 10, 15, 0):
        r = get.get(("full", 8, fps))
        if r:
            out.append(f"| {fps or 'every frame (30)'} | " + " | ".join(row(r)) + " |")
    out += ["", "## Saturation: every decoded frame (30 fps per camera), all optimisations", "",
            HEAD.format(first="streams"), SEP]
    for n in (1, 2, 4, 8):
        r = get.get(("full", n, 0))
        if r:
            out.append(f"| {n} | " + " | ".join(row(r)) + " |")

    coco32 = json.loads((REPO_ROOT / "benchmarks/milestone2/m2_coco_lb_minimal_yoloworld_m.json").read_text())
    coco16 = json.loads((B8 / "m8_coco_fp16_lb_minimal_yoloworld_m.json").read_text())
    out += ["", "## FP16 accuracy and speed", "", "YOLO-World-M on COCO val2017 (5,000 images, Milestone 2 protocol, "
            "deployment letterbox `minimal`, batch 1):", "",
            "| precision | AP | AP50 | median latency (total / inference) | peak allocated |", "|---|---|---|---|---|"]
    for name, r in (("FP32 (Milestone 2)", coco32), ("FP16", coco16)):
        lat = r["latency_ms"]
        out.append(f"| {name} | {100 * r['coco']['AP']:.2f} | {100 * r['coco']['AP50']:.2f} | "
                   f"{lat['total']['median']:.1f} / {lat['inference']['median']:.1f} ms | {r['memory_peak_mib']['allocated']:.0f} MiB |")
    c = json.loads((B8 / "m8_clip_fp16.json").read_text())
    out += ["", f"CLIP ViT-B/32 (Re-ID and search embedder) on {c['crops']} real MOT17 person crops ({c['people']} people): "
            f"FP32 vs FP16 embedding cosine mean {c['agreement_cosine']['mean']}, min {c['agreement_cosine']['min']}.", "",
            "| precision | same-vs-different AUROC | ms / crop at batch 1 / 8 / 32 / 64 | peak allocated (batch 64) |",
            "|---|---|---|---|"]
    for p in ("float32", "float16"):
        x = c["precisions"][p]
        sp = x["speed"]
        out.append(f"| {p} | {x['auroc_same_vs_different']['auroc']:.4f} | "
                   + " / ".join(f"{r['ms_per_crop']:.2f}" for r in sp) + f" | {sp[-1]['peak_allocated_mib']} MiB |")
    out += ["", "Interpretation: docs/milestone8.md.", ""]
    (B8 / "README.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
