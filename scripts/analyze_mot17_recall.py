#!/usr/bin/env python
"""Which MOT17 pedestrians do the detectors miss? Recall by ground-truth visibility and box height.

Uses the cached detections (scripts/detect_mot17.py) on the evaluation split. A ground-truth pedestrian
(class 1, scored) counts as found if a detection with score > ``--min-score`` (default 0.1, the lowest
score ByteTrack ever uses) overlaps it with IoU >= 0.5 under a one-to-one (Hungarian) matching per frame.
Writes benchmarks/milestone3/recall_by_visibility.{json,md}.

Usage:
    python scripts/analyze_mot17_recall.py
    python scripts/analyze_mot17_recall.py --models gdino_t yoloworld_x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from eval_mot17 import load_detections
from scipy.optimize import linear_sum_assignment

from lgep.evaluation.mot import load_sequence, split_frames
from lgep.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
HEIGHT_BINS = [(0, 50), (50, 100), (100, 200), (200, 1e9)]
DEFAULT_MODELS = ["gdino_t", "gdino_b", "yoloworld_s", "yoloworld_m", "yoloworld_l", "yoloworld_x",
                  "yoloworld_l_1280", "yoloworld_x_1280"]


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def matched_gt(gt: np.ndarray, det: np.ndarray, thr: float = 0.5) -> np.ndarray:
    found = np.zeros(len(gt), bool)
    if len(gt) and len(det):
        m = iou(gt, det)
        r, c = linear_sum_assignment(-m)
        found[r[m[r, c] >= thr]] = True
    return found


def analyse(seqs, cache: Path, split: str, min_score: float) -> dict:
    vis, height, found = [], [], []
    for seq in seqs:
        gt = np.loadtxt(seq.root / "gt/gt.txt", delimiter=",", ndmin=2)
        gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
        dets = load_detections(cache / f"{seq.name}.npz")
        for f in split_frames(seq.length, split):
            g = gt[gt[:, 0] == f]
            boxes = np.column_stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]])
            b, s = dets.get(f, (np.zeros((0, 4)), np.zeros(0)))
            found.append(matched_gt(boxes, np.asarray(b, float)[np.asarray(s) > min_score]))
            vis.append(g[:, 8])
            height.append(g[:, 5])
    vis, height, found = np.concatenate(vis), np.concatenate(height), np.concatenate(found)

    def by(values, bins):
        return [{"range": [lo, min(hi, 1.0) if bins is VIS_BINS else hi], "gt": int(m.sum()),
                 "recall": round(float(found[m].mean()), 4) if m.any() else None}
                for lo, hi in bins for m in [(values >= lo) & (values < hi)]]

    return {"gt": int(len(found)), "recall": round(float(found.mean()), 4), "by_visibility": by(vis, VIS_BINS),
            "by_height_px": by(height, HEIGHT_BINS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone3_mot17.yaml")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--min-score", type=float, default=0.1)
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    seqs = [load_sequence(REPO_ROOT / cfg["data"]["root"] / n) for n in cfg["data"]["sequences"]]
    split = cfg["evaluation"]["split"]
    cache_root = REPO_ROOT / cfg["detections"]["cache_dir"]
    res = {"measured": True, "split": split, "min_score": args.min_score, "iou": 0.5, "models": {}}
    for m in args.models:
        if not (cache_root / m / "meta.json").is_file():
            print(f"skip {m}: no cached detections", file=sys.stderr)
            continue
        res["models"][m] = analyse(seqs, cache_root / m, split, args.min_score)
        print(m, "recall", res["models"][m]["recall"])

    first = next(iter(res["models"].values()))
    vis_hdr = [f"{b['range'][0]:.2f}–{b['range'][1]:.2f}" for b in first["by_visibility"]]
    h_hdr = ["< 50", "50–100", "100–200", "≥ 200"]
    lines = ["# Which pedestrians are missed? Detection recall on MOT17 " + split, "",
             f"**Actual measurement**, `scripts/analyze_mot17_recall.py`: ground-truth pedestrians matched one-to-one "
             f"to detections with score > {args.min_score} at IoU ≥ 0.5, all {first['gt']} ground-truth boxes of {split}.", "",
             "## By visibility (fraction of the person not occluded, from the MOT17 annotation)", "",
             "| detector | all | " + " | ".join(vis_hdr) + " |", "|---|---|" + "---|" * len(vis_hdr),
             "| share of ground truth | 100 % | " + " | ".join(f"{100 * b['gt'] / first['gt']:.0f} %"
                                                          for b in first["by_visibility"]) + " |"]
    for m, r in res["models"].items():
        lines.append(f"| {m} | {100 * r['recall']:.1f} | " + " | ".join(f"{100 * b['recall']:.1f}"
                                                                     for b in r["by_visibility"]) + " |")
    lines += ["", "## By box height (pixels in the original frame)", "",
              "| detector | " + " | ".join(h_hdr) + " |", "|---|" + "---|" * len(h_hdr),
              "| share of ground truth | " + " | ".join(f"{100 * b['gt'] / first['gt']:.0f} %"
                                                     for b in first["by_height_px"]) + " |"]
    for m, r in res["models"].items():
        lines.append(f"| {m} | " + " | ".join(f"{100 * b['recall']:.1f}" for b in r["by_height_px"]) + " |")
    out = REPO_ROOT / cfg["benchmark_dir"]
    (out / "recall_by_visibility.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "recall_by_visibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
