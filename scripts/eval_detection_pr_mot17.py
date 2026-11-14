#!/usr/bin/env python
"""Milestone 9 detection: person precision, recall and AP50 on MOT17 val_half from the Milestone 3 caches (CPU).

Milestone 2 measured COCO AP (80 classes); this adds the detection metrics the project plan lists (precision,
recall) on the pedestrian data the trackers are evaluated on, for every cached detector.

Matching (VOC/COCO style, IoU >= 0.5): detections of all frames sorted by score; each takes the unmatched
pedestrian box (class 1, considered) it overlaps most. As in MOTChallenge's evaluation, a detection that matches
no pedestrian but overlaps a distractor (person on a vehicle, static person, distractor, reflection: classes 2,
7, 8, 12) by IoU >= 0.5 is ignored, not counted as a false positive. AP50 = area under the all-point
interpolated precision-recall curve. Precision / recall / F1 at fixed score thresholds and at the best F1.

Usage:
    python scripts/eval_detection_pr_mot17.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from lgep.evaluation.mot import load_sequence, split_frames
from lgep.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DISTRACTORS = (2, 7, 8, 12)
THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
DETECTORS = ("gdino_t", "gdino_b", "yoloworld_s", "yoloworld_m", "yoloworld_l", "yoloworld_x", "yoloworld_l_1280",
             "yoloworld_x_1280")


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of boxes (N, 4) and (M, 4), x1 y1 x2 y2."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def ground_truth(seq, frames: range) -> dict:
    g = np.loadtxt(seq.root / "gt" / "gt.txt", delimiter=",")
    out = {}
    for f in frames:
        r = g[g[:, 0] == f]
        xyxy = np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]]
        out[f] = (xyxy[(r[:, 7] == 1) & (r[:, 6] == 1)], xyxy[np.isin(r[:, 7], DISTRACTORS)])
    return out


def match(dets: dict, gt: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Scores and TP flags of all non-ignored detections, and the number of pedestrian boxes."""
    order = []
    for f, (b, s) in dets.items():
        if f in gt:
            order += [(float(sc), f, i) for i, sc in enumerate(s)]
    order.sort(key=lambda x: -x[0])
    used = {f: np.zeros(len(p), bool) for f, (p, _) in gt.items()}
    ious = {f: (iou(dets[f][0], gt[f][0]), iou(dets[f][0], gt[f][1])) for f in dets if f in gt}
    scores, tp = [], []
    for sc, f, i in order:
        ped, dis = ious[f]
        cand = np.where(~used[f])[0]
        if len(cand) and ped[i, cand].max() >= 0.5:
            used[f][cand[ped[i, cand].argmax()]] = True
            scores.append(sc)
            tp.append(True)
        elif dis.shape[1] and dis[i].max() >= 0.5:
            continue  # on a distractor: ignored
        else:
            scores.append(sc)
            tp.append(False)
    return np.array(scores), np.array(tp, bool), int(sum(len(p) for p, _ in gt.values()))


def summarise(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> dict:
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1)
    # all-point interpolated AP
    mrec, mpre = np.r_[0, rec, 1], np.r_[1, prec, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap50 = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    out = {"ap50": ap50, "pedestrian_boxes": n_gt, "max_recall": float(rec[-1]) if len(rec) else 0.0, "at": {}}
    for t in THRESHOLDS:
        k = int(np.searchsorted(-scores, -t, side="right"))  # detections with score >= t
        p = float(ctp[k - 1] / k) if k else 0.0
        r = float(ctp[k - 1] / n_gt) if k else 0.0
        out["at"][str(t)] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(f1.argmax())
    out["best_f1"] = {"threshold": float(scores[i]), "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
    return out


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs/milestone3_mot17.yaml")
    root, cache = REPO_ROOT / cfg["data"]["root"], REPO_ROOT / cfg["detections"]["cache_dir"]
    seqs = [load_sequence(root / n) for n in cfg["data"]["sequences"]]
    gt = {}
    for s in seqs:
        gt.update({(s.name, f): v for f, v in ground_truth(s, split_frames(s.length, "val_half")).items()})
    res = {"measured": True, "date": datetime.now().isoformat(timespec="seconds"), "split": "val_half",
           "sequences": [s.name for s in seqs], "frames": len(gt), "iou": 0.5, "detectors": {}}
    for d in DETECTORS:
        dets = {}
        for s in seqs:
            z = np.load(cache / d / f"{s.name}.npz")
            for f in split_frames(s.length, "val_half"):
                m = z["frame"] == f
                dets[(s.name, f)] = (z["boxes"][m].astype(np.float64), z["scores"][m].astype(np.float64))
        m3 = json.loads((REPO_ROOT / f"benchmarks/milestone3/m3_mot17_{d}.json").read_text())
        r = summarise(*match(dets, gt))
        r["milestone3"] = {"hota_tuned": m3["evaluation"]["tuned"]["metrics"]["HOTA"],
                           "detector_ms_median": (m3.get("latency_per_frame_ms") or {}).get("detector_median")}
        res["detectors"][d] = r
        print(f"{d:17s} AP50 {r['ap50']:.3f}  max recall {r['max_recall']:.3f}  "
              f"@0.3 P {r['at']['0.3']['precision']:.3f} R {r['at']['0.3']['recall']:.3f}  best F1 {r['best_f1']['f1']:.3f} "
              f"at {r['best_f1']['threshold']:.2f}", flush=True)
    out = REPO_ROOT / "benchmarks" / "milestone9"
    out.mkdir(parents=True, exist_ok=True)
    (out / "m9_detection_pr.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    n_ped = next(iter(res["detectors"].values()))["pedestrian_boxes"]
    lines = ["# Person detection on MOT17 val_half: precision, recall, AP50 (zero-shot detectors)", "",
             f"**Actual measurements**, {res['date'][:10]}, `scripts/eval_detection_pr_mot17.py`, from the Milestone 3 "
             f"detection caches (prompt \"person\"). {res['frames']} frames, {n_ped} pedestrian boxes; IoU >= 0.5; "
             "detections on distractors (static people, reflections, people on vehicles) ignored, as in MOTChallenge.",
             "", "| detector | AP50 | max recall | P / R at 0.1 | P / R at 0.3 | P / R at 0.5 | best F1 (threshold) | "
             "tracking HOTA (Milestone 3) |", "|---|---|---|---|---|---|---|---|"]
    for d, r in res["detectors"].items():
        pr = lambda t: f"{r['at'][t]['precision']:.2f} / {r['at'][t]['recall']:.2f}"  # noqa: E731
        lines.append(f"| {d} | {r['ap50']:.3f} | {r['max_recall']:.3f} | {pr('0.1')} | {pr('0.3')} | {pr('0.5')} | "
                     f"{r['best_f1']['f1']:.3f} ({r['best_f1']['threshold']:.2f}) | {100 * r['milestone3']['hota_tuned']:.1f} |")
    (out / "m9_detection_pr.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out / "m9_detection_pr.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
