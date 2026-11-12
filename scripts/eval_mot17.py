#!/usr/bin/env python
"""Track cached MOT17 detections with ByteTrack and score them with TrackEval (HOTA, IDF1, MOTA, IDSW).

Needs the detections from scripts/detect_mot17.py. Protocol (configs/milestone3_mot17.yaml):
  1. tune ByteTrack's ``track_thresh`` on ``train_half`` (best HOTA over the configured grid);
  2. report ``val_half`` with ByteTrack's default threshold and with the tuned one.
Each split is tracked from scratch with frames renumbered from 1, as in ByteTrack's evaluation.

Outputs:
  outputs/milestone3/<run_name>_<model>_<timestamp>/   result files, results.json, summary.md
  benchmarks/milestone3/<run_name>_<model>.{json,md}   when every configured sequence was evaluated

Usage:
    python scripts/eval_mot17.py --model yoloworld_l
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from run_detector import select_model, stats

from lgep.evaluation.mot import TRACKEVAL_COMMIT, evaluate_mot, load_sequence, split_frames, write_gt_split, write_mot_results
from lgep.tracking import CameraTracker, TrackerConfig
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_mot17")
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_NAME = "bytetrack"
EMPTY = (np.zeros((0, 4)), np.zeros(0))
TABLE_FIELDS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN")


def load_detections(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    z = np.load(path)
    frames, boxes, scores = z["frame"], z["boxes"], z["scores"]
    out = {}
    for f in np.unique(frames):
        lo, hi = np.searchsorted(frames, f, "left"), np.searchsorted(frames, f, "right")
        out[int(f)] = (boxes[lo:hi], scores[lo:hi])
    return out


def track(seq, dets: dict, split: str, tcfg: TrackerConfig, phrase: str) -> tuple[list[tuple], list[float]]:
    """Run ByteTrack on one split of one sequence; returns MOT rows (frames renumbered) and tracker ms/frame."""
    frames = split_frames(seq.length, split)
    tracker = CameraTracker(seq.name, TrackerConfig(**{**asdict(tcfg), "frame_rate": seq.fps}))
    rows, ms = [], []
    for f in frames:
        r = tracker.update({phrase: dets.get(f, EMPTY)})
        ms.append(r.tracker_ms)
        for t in r.tracks:
            x1, y1, x2, y2 = t.box_xyxy
            rows.append((f - frames.start + 1, t.local_id, x1, y1, x2 - x1, y2 - y1, t.score))
    return rows, ms


def run_split(seqs, dets, split, tcfg, phrase, gt_dir: Path, out_dir: Path) -> dict:
    ms_all, n_rows, ids = [], 0, 0
    for seq in seqs:
        rows, ms = track(seq, dets[seq.name], split, tcfg, phrase)
        n_rows += write_mot_results(out_dir / TRACKER_NAME / "data" / f"{seq.name}.txt", rows)
        ids += len({r[1] for r in rows})
        ms_all += ms
    lengths = {s.name: len(split_frames(s.length, split)) for s in seqs}
    m = evaluate_mot(gt_dir, out_dir, TRACKER_NAME, lengths)
    return {"track_thresh": tcfg.track_thresh, "metrics": m["combined"], "per_sequence": m["per_sequence"],
            "output_boxes": n_rows, "track_ids": ids, "tracker_ms": stats(ms_all)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone3_mot17.yaml")
    ap.add_argument("--model", required=True, help="preset whose cached detections to track")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    ap.add_argument("--keep-files", action="store_true", help="keep per-threshold tracking result files")
    args = ap.parse_args()

    cfg = select_model(load_config(REPO_ROOT / args.config, args.override), args.model)
    run_dir = REPO_ROOT / cfg["output_dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), run_dir / "run.log")
    cache = REPO_ROOT / cfg["detections"]["cache_dir"] / args.model
    try:
        meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
        root = REPO_ROOT / cfg["data"]["root"]
        seqs = [load_sequence(root / n) for n in cfg["data"]["sequences"]]
        missing = [s.name for s in seqs if not (cache / f"{s.name}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"No cached detections for {missing} in {cache}. "
                                    f"Run: python scripts/detect_mot17.py --model {args.model}")
        dets = {s.name: load_detections(cache / f"{s.name}.npz") for s in seqs}
        base = TrackerConfig.from_dict(cfg["tracker"])
        tune_split, eval_split = cfg["tuning"]["split"], cfg["evaluation"]["split"]
        gt = {sp: run_dir / "gt" / sp for sp in (tune_split, eval_split)}
        for sp, d in gt.items():
            for s in seqs:
                write_gt_split(s, sp, d)

        # 1. tune track_thresh on the tuning split (pointless for the oracle: every score is 1.0)
        grid = [base.track_thresh] if meta["model"]["backend"] == "ground_truth" else cfg["tuning"]["track_thresh"]
        key = cfg["tuning"]["metric"]
        tuning = []
        for thr in grid:
            r = run_split(seqs, dets, tune_split, TrackerConfig(**{**asdict(base), "track_thresh": thr}),
                          cfg["prompt"], gt[tune_split], run_dir / tune_split / f"thr{thr}")
            log.info("%s track_thresh %.2f: HOTA %.1f IDF1 %.1f MOTA %.1f IDSW %d", tune_split, thr,
                     100 * r["metrics"]["HOTA"], 100 * r["metrics"]["IDF1"], 100 * r["metrics"]["MOTA"],
                     r["metrics"]["IDSW"])
            tuning.append({"track_thresh": thr, **{k: r["metrics"][k] for k in TABLE_FIELDS}})
        best = max(tuning, key=lambda t: t[key])["track_thresh"]

        # 2. evaluate on the evaluation split with ByteTrack's default and the tuned threshold
        evaluation = {}
        for label, thr in (("default", base.track_thresh), ("tuned", best)):
            r = run_split(seqs, dets, eval_split, TrackerConfig(**{**asdict(base), "track_thresh": thr}),
                          cfg["prompt"], gt[eval_split], run_dir / eval_split / label)
            evaluation[label] = r
            log.info("%s %s (track_thresh %.2f): HOTA %.1f IDF1 %.1f MOTA %.1f IDSW %d | tracker %.2f ms/frame",
                     eval_split, label, thr, 100 * r["metrics"]["HOTA"], 100 * r["metrics"]["IDF1"],
                     100 * r["metrics"]["MOTA"], r["metrics"]["IDSW"], r["tracker_ms"]["median"])
    except Exception:
        log.exception("MOT17 evaluation failed (log: %s)", run_dir / "run.log")
        return 1
    if not args.keep_files:
        shutil.rmtree(run_dir / tune_split, ignore_errors=True)

    det_lat = meta.get("latency_ms")
    res = {
        "run_name": cfg["run_name"],
        "measured": True,
        "environment": collect_environment(),
        "config": cfg,
        "model": meta["model"],
        "detector": {k: meta.get(k) for k in ("input_size", "latency_ms", "frames_per_second_wall",
                                               "memory_peak_mib", "memory_at_end", "other_gpu_processes",
                                               "wall_seconds", "environment")},
        "protocol": {"tuning_split": tune_split, "evaluation_split": eval_split, "tuning_metric": key,
                     "sequences": [s.name for s in seqs], "frames": {sp: sum(len(split_frames(s.length, sp))
                                                                          for s in seqs) for sp in gt},
                     "trackeval_commit": TRACKEVAL_COMMIT},
        "tracker": asdict(base),
        "tuning": tuning,
        "best_track_thresh": best,
        "evaluation": evaluation,
        "latency_per_frame_ms": None if det_lat is None else {
            "detector_median": det_lat["total"]["median"], "decode_median": det_lat["decode"]["median"],
            "tracker_median": evaluation["tuned"]["tracker_ms"]["median"],
            "detector_plus_tracker_median": round(det_lat["total"]["median"]
                                                  + evaluation["tuned"]["tracker_ms"]["median"], 2)},
    }
    (run_dir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_summary(res, run_dir / "summary.md")
    if cfg["data"]["sequences"] == load_config(REPO_ROOT / args.config)["data"]["sequences"]:
        bdir = REPO_ROOT / cfg["benchmark_dir"]
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"{cfg['run_name']}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        write_summary(res, bdir / f"{cfg['run_name']}.md")
        log.info("Published summary to %s", bdir.relative_to(REPO_ROOT))
    log.info("Wrote %s", run_dir.relative_to(REPO_ROOT))
    return 0


def write_summary(res: dict, path: Path) -> None:
    m, ev = res["model"], res["evaluation"]
    lines = [f"# {res['run_name']} — ByteTrack on MOT17 {res['protocol']['evaluation_split']}", "",
             f"**Actual measurement**, generated by `scripts/eval_mot17.py`. Detector: `{m.get('model_id')}` "
             f"({m.get('backend')}). Tuned on {res['protocol']['tuning_split']} by {res['protocol']['tuning_metric']}; "
             f"TrackEval {res['protocol']['trackeval_commit'][:7]}.", "",
             "| track_thresh | split | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN |", "|---|---|---|---|---|---|---|---|---|---|"]
    def row(thr, split, x, bold=False):
        b = "**" if bold else ""
        return (f"| {thr} | {split} | {b}{100 * x['HOTA']:.1f}{b} | {100 * x['DetA']:.1f} | {100 * x['AssA']:.1f} | "
                f"{100 * x['IDF1']:.1f} | {100 * x['MOTA']:.1f} | {x['IDSW']} | {x['CLR_FP']} | {x['CLR_FN']} |")

    for t in res["tuning"]:
        lines.append(row(t["track_thresh"], res["protocol"]["tuning_split"], t))
    for label, r in ev.items():
        lines.append(row(f"**{r['track_thresh']}** ({label})", res["protocol"]["evaluation_split"], r["metrics"], True))
    lat = res.get("latency_per_frame_ms")
    if lat:
        lines += ["", f"Per frame (median): detector {lat['detector_median']:.1f} ms + ByteTrack "
                      f"{lat['tracker_median']:.2f} ms (CPU); JPEG decoding {lat['decode_median']:.1f} ms not included."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
