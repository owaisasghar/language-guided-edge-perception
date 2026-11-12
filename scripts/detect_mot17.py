#!/usr/bin/env python
"""Run one detector over every MOT17 training frame and cache the boxes for tracking evaluation.

Tracking is evaluated many times (threshold tuning, two splits); the detector runs only once here.
Output, per model:
  <cache_dir>/<model>/<sequence>.npz   frame (int32), boxes (float32 x1 y1 x2 y2, image px), scores (float32)
  <cache_dir>/<model>/meta.json        detector description, latency per stage, image decoding time, VRAM

The ``oracle_gt`` preset writes the ground-truth pedestrian boxes (score 1.0) instead; it needs no GPU.

Usage:
    python scripts/detect_mot17.py --model yoloworld_l
    python scripts/detect_mot17.py --model gdino_t --override data.sequences=[MOT17-09]   # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from run_detector import select_model, stats

from lgep.evaluation.mot import load_sequence
from lgep.perception.factory import build_detector
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.gpu import gpu_compute_processes, memory_snapshot, reset_peak
from lgep.utils.logging import setup_logging

log = logging.getLogger("detect_mot17")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sequences(cfg: dict):
    root = REPO_ROOT / cfg["data"]["root"]
    seqs = [load_sequence(root / name) for name in cfg["data"]["sequences"]]
    missing = [s.name for s in seqs if not s.image(s.length).is_file()]
    if missing:
        raise FileNotFoundError(f"Images missing for {missing}. Run: python scripts/download_mot17.py")
    return seqs


def oracle(seq) -> dict[str, np.ndarray]:
    """Ground-truth pedestrians (class 1, considered for evaluation) as detections with score 1.0."""
    gt = np.loadtxt(seq.root / "gt/gt.txt", delimiter=",", ndmin=2)
    gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    boxes = np.column_stack([gt[:, 2], gt[:, 3], gt[:, 2] + gt[:, 4], gt[:, 3] + gt[:, 5]])
    order = np.argsort(gt[:, 0], kind="stable")
    return {"frame": gt[order, 0].astype(np.int32), "boxes": boxes[order].astype(np.float32),
            "scores": np.ones(len(gt), np.float32)}


def run_detector(cfg: dict, seqs, out: Path) -> dict:
    env = collect_environment()
    others = gpu_compute_processes()
    if others:
        log.warning("Other GPU compute processes present (benchmarks may be affected): %s",
                    ", ".join(f"{p['name']}({p['pid']}, {p['used_mib']} MiB)" for p in others))
    detector = build_detector(cfg["model"], cfg.get("device", "auto"), base_dir=REPO_ROOT)
    dev = detector.device
    mem_after_load = memory_snapshot(dev)
    phrases = [cfg["prompt"]]
    first = Image.open(seqs[0].image(1)).convert("RGB")
    for _ in range(int(cfg["benchmark"]["warmup_runs"])):
        warm = detector.detect(first, phrases)
    reset_peak(dev)

    timings = {k: [] for k in ("total", "inference", "preprocess", "postprocess", "decode")}
    per_seq, n_frames, t0 = {}, sum(s.length for s in seqs), time.perf_counter()
    done = 0
    for seq in seqs:
        frames, boxes, scores = [], [], []
        for f in range(1, seq.length + 1):
            td = time.perf_counter()
            image = Image.open(seq.image(f)).convert("RGB")
            timings["decode"].append((time.perf_counter() - td) * 1e3)
            try:
                r = detector.detect(image, phrases)
            except torch.cuda.OutOfMemoryError as e:
                raise RuntimeError(f"CUDA out of memory on {seq.name} frame {f}") from e
            tm = r.timings
            for k, v in (("total", tm.total_ms), ("inference", tm.inference_ms),
                         ("preprocess", tm.preprocess_ms), ("postprocess", tm.postprocess_ms)):
                timings[k].append(v)
            for d in r.detections:
                frames.append(f)
                boxes.append(d.box_xyxy)
                scores.append(d.score)
            done += 1
            if done % 500 == 0 or done == n_frames:
                el = time.perf_counter() - t0
                log.info("%5d/%d frames | median detector %.1f ms | elapsed %.0f s | ETA %.0f s", done, n_frames,
                         stats(timings["total"])["median"], el, el / done * (n_frames - done))
        np.savez_compressed(out / f"{seq.name}.npz", frame=np.asarray(frames, np.int32),
                            boxes=np.asarray(boxes, np.float32).reshape(-1, 4), scores=np.asarray(scores, np.float32))
        per_seq[seq.name] = {"frames": seq.length, "detections": len(scores)}
    wall = time.perf_counter() - t0
    mem_end = memory_snapshot(dev)
    return {
        "measured": True,
        "environment": env,
        "other_gpu_processes": others,
        "model": detector.describe(),
        "prompt": cfg["prompt"],
        "input_size": list(warm.input_size),
        "frames": n_frames,
        "sequences": per_seq,
        "latency_ms": {k: stats(v) for k, v in timings.items()},  # decode = JPEG -> RGB, not in "total"
        "frames_per_second_wall": round(n_frames / wall, 2),
        "memory_after_load": mem_after_load.to_dict() if mem_after_load else None,
        "memory_peak_mib": {"allocated": round(mem_end.peak_allocated_mib, 1),
                            "reserved": round(mem_end.peak_reserved_mib, 1)} if mem_end else None,
        "memory_at_end": mem_end.to_dict() if mem_end else None,
        "wall_seconds": round(wall, 1),
    }


def cached(out: Path, cfg: dict) -> bool:
    """True if ``out`` holds a finished run with the same model settings, prompt and sequences."""
    try:
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    same = (meta.get("sequences_requested") == cfg["data"]["sequences"]
            and meta["config"]["model"] == cfg["model"] and meta["config"]["prompt"] == cfg["prompt"])
    return same and all((out / f"{s}.npz").is_file() for s in cfg["data"]["sequences"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone3_mot17.yaml")
    ap.add_argument("--model", required=True, help="name of a preset in the config's `models:` block")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    ap.add_argument("--force", action="store_true", help="re-run even if a matching cache exists")
    args = ap.parse_args()

    cfg = select_model(load_config(REPO_ROOT / args.config, args.override), args.model)
    out = REPO_ROOT / cfg["detections"]["cache_dir"] / args.model
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), out / "detect.log")
    if not args.force and cached(out, cfg):
        log.info("%s: cache matches this config and all sequences (use --force to re-run)", args.model)
        return 0
    try:
        seqs = sequences(cfg)
        if cfg["model"].get("backend") == "ground_truth":
            for seq in seqs:
                np.savez_compressed(out / f"{seq.name}.npz", **oracle(seq))
            meta = {"measured": True, "model": {"backend": "ground_truth", "model_id": "oracle_gt"},
                    "frames": sum(s.length for s in seqs)}
        else:
            meta = run_detector(cfg, seqs, out)
    except Exception:
        log.exception("Detection failed")
        return 1
    meta.update({"config": cfg, "sequences_requested": cfg["data"]["sequences"]})
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if "latency_ms" in meta:
        log.info("%s: median detector %.1f ms (+ decode %.1f ms), peak %.0f MiB allocated, %d frames in %.0f s",
                 args.model, meta["latency_ms"]["total"]["median"], meta["latency_ms"]["decode"]["median"],
                 meta["memory_peak_mib"]["allocated"] if meta["memory_peak_mib"] else float("nan"),
                 meta["frames"], meta["wall_seconds"])
    log.info("Wrote %s", out.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
