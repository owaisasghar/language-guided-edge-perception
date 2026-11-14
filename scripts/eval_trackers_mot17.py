#!/usr/bin/env python
"""Milestone 9: compare ByteTrack, OC-SORT and BoT-SORT on MOT17 with the same detections (CPU only).

For every detector in the config (Milestone 3's detection caches) and every tracker:
  1. tune the high-score threshold on ``train_half`` (best HOTA over Milestone 3's grid);
  2. evaluate ``val_half`` with the trackers' default (0.6) and the tuned threshold (official TrackEval).
Then a timing pass, alone on the machine: CPU time per frame of each tracker on one detector's val_half, and
of BoT-SORT's camera-motion estimate (sparse optical flow on the frames).

Outputs:
  outputs/milestone9/m9_trackers_<timestamp>/     per-job result files, results.json, run.log
  benchmarks/milestone9/m9_trackers.{json,md}

Usage:
    python scripts/eval_trackers_mot17.py [--config configs/milestone9_trackers.yaml] [--detectors ...]
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # parallel jobs; NumPy's BLAS pool would oversubscribe

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from eval_mot17 import TABLE_FIELDS, load_detections
from run_detector import stats

from lgep.evaluation.mot import TRACKEVAL_COMMIT, evaluate_mot, load_sequence, split_frames, write_gt_split, write_mot_results
from lgep.tracking import CameraTracker, TrackerConfig
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_trackers_mot17")
REPO_ROOT = Path(__file__).resolve().parents[1]
EMPTY = (np.zeros((0, 4)), np.zeros(0))
OFFICIAL = {"bytetrack": "ifzhang/ByteTrack@d1bf019", "ocsort": "noahcao/OC_SORT@8462e7e",
            "botsort": "NirAharon/BoT-SORT@2519854"}


# --------------------------------------------------------------------------------------------- camera motion
def camera_motion(seq, split: str, method: str, downscale: int, cache_dir: Path) -> tuple[np.ndarray, list[float]]:
    """2 x 3 warps (previous frame -> this frame) for every frame of a split, from BoT-SORT's GMC on the
    MOT17 images. The split starts afresh (its first frame has no predecessor), as its tracking does."""
    path = cache_dir / f"{seq.name}_{split}_{method}_ds{downscale}.npz"
    if path.is_file():
        z = np.load(path)
        return z["warps"], z["ms"].tolist()
    import cv2

    from lgep.tracking.botsort import GMC

    gmc, warps, ms = GMC(method, downscale=downscale), [], []
    for f in split_frames(seq.length, split):
        img = cv2.imread(str(seq.image(f)))  # BGR, as the official code reads the frames
        t0 = time.perf_counter()
        warps.append(gmc.apply(img))
        ms.append((time.perf_counter() - t0) * 1e3)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, warps=np.asarray(warps), ms=np.asarray(ms))
    return np.asarray(warps), ms


def uses_camera_motion(cfg: dict, tracker_name: str) -> bool:
    t = cfg["trackers"][tracker_name]
    return t.get("algorithm") == "botsort" and (t.get("botsort") or {}).get("cmc", "sparseOptFlow") != "none"


# ------------------------------------------------------------------------------------------------- tracking
def track(seq, dets: dict, split: str, tcfg: TrackerConfig, phrase: str, warps=None):
    """Run one tracker over one split of one sequence; MOT rows (frames renumbered from 1) and ms per frame."""
    frames = split_frames(seq.length, split)
    tracker = CameraTracker(seq.name, replace(tcfg, frame_rate=seq.fps))
    rows, ms = [], []
    for k, f in enumerate(frames):
        r = tracker.update({phrase: dets.get(f, EMPTY)}, warp=None if warps is None else warps[k])
        ms.append(r.tracker_ms)
        for t in r.tracks:
            x1, y1, x2, y2 = t.box_xyxy
            rows.append((f - frames.start + 1, t.local_id, x1, y1, x2 - x1, y2 - y1, t.score))
    return rows, ms


def run_split(seqs, dets, split, tcfg, phrase, gt_dir: Path, out_dir: Path, warps: dict | None) -> dict:
    n_rows, ids = 0, 0
    for seq in seqs:
        rows, _ = track(seq, dets[seq.name], split, tcfg, phrase, None if warps is None else warps[(seq.name, split)])
        n_rows += write_mot_results(out_dir / "tracker" / "data" / f"{seq.name}.txt", rows)
        ids += len({r[1] for r in rows})
    lengths = {s.name: len(split_frames(s.length, split)) for s in seqs}
    m = evaluate_mot(gt_dir, out_dir, "tracker", lengths)
    return {"track_thresh": tcfg.track_thresh, "metrics": m["combined"], "per_sequence": m["per_sequence"],
            "output_boxes": n_rows, "track_ids": ids}


def job(cfg: dict, detector: str, tracker_name: str, run_dir: str, warps_paths: dict | None) -> dict:
    """One detector x tracker: tune on the tuning split, evaluate on the evaluation split (own process)."""
    run_dir = Path(run_dir)
    root = REPO_ROOT / cfg["data"]["root"]
    seqs = [load_sequence(root / n) for n in cfg["data"]["sequences"]]
    cache = REPO_ROOT / cfg["detections"]["cache_dir"] / detector
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    dets = {s.name: load_detections(cache / f"{s.name}.npz") for s in seqs}
    warps = None if not warps_paths or not uses_camera_motion(cfg, tracker_name) else {
        tuple(k.split("|")): np.load(v)["warps"] for k, v in warps_paths.items()}
    base = TrackerConfig.from_dict({**cfg["tracker"], **cfg["trackers"][tracker_name]})
    tune_split, eval_split = cfg["tuning"]["split"], cfg["evaluation"]["split"]
    gt = {sp: run_dir / "gt" / sp for sp in (tune_split, eval_split)}
    work = run_dir / f"{detector}__{tracker_name}"
    grid = [base.track_thresh] if meta["model"]["backend"] == "ground_truth" else cfg["tuning"]["track_thresh"]
    tuning = []
    for thr in grid:
        r = run_split(seqs, dets, tune_split, replace(base, track_thresh=thr), cfg["prompt"], gt[tune_split],
                      work / tune_split / f"thr{thr}", warps)
        tuning.append({"track_thresh": thr, **{k: r["metrics"][k] for k in TABLE_FIELDS}})
    best = max(tuning, key=lambda t: t[cfg["tuning"]["metric"]])["track_thresh"]
    evaluation = {}
    for label, thr in (("default", base.track_thresh), ("tuned", best)):
        evaluation[label] = run_split(seqs, dets, eval_split, replace(base, track_thresh=thr), cfg["prompt"],
                                      gt[eval_split], work / eval_split / label, warps)
    shutil.rmtree(work / tune_split, ignore_errors=True)
    return {"detector": detector, "detector_model": meta["model"], "tracker": tracker_name,
            "tracker_config": {**cfg["tracker"], **cfg["trackers"][tracker_name]}, "tuning": tuning,
            "best_track_thresh": best, "evaluation": evaluation}


# ------------------------------------------------------------------------------------------------- report
def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def write_summary(res: dict, path: Path) -> None:
    trackers, detectors = list(res["trackers"]), res["detectors"]
    by = {(r["detector"], r["tracker"]): r for r in res["results"]}
    lines = ["# Milestone 9 — tracker comparison on MOT17 (val_half)", "",
             f"**Actual measurements**, {res['date'][:10]}, `scripts/eval_trackers_mot17.py` "
             f"(`{res['config']}`). Detections: the Milestone 3 caches (zero-shot detectors, never trained on "
             f"MOT17). Each tracker's high-score threshold tuned per detector on train_half by HOTA; "
             f"TrackEval {TRACKEVAL_COMMIT[:7]}; {res['protocol']['frames']['val_half']} frames, "
             f"{len(res['protocol']['sequences'])} sequences.", ""]
    lines += ["## Tuned threshold", "",
              "| detector | tracker | thr | HOTA | DetA | AssA | IDF1 | MOTA | ID switches | FP | FN |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in detectors:
        best_hota = max(by[(d, t)]["evaluation"]["tuned"]["metrics"]["HOTA"] for t in trackers if (d, t) in by)
        for t in trackers:
            r = by.get((d, t))
            if r is None:
                continue
            m = r["evaluation"]["tuned"]["metrics"]
            b = "**" if m["HOTA"] == best_hota else ""
            lines.append(f"| {d} | {t} | {r['best_track_thresh']} | {b}{pct(m['HOTA'])}{b} | {pct(m['DetA'])} | "
                         f"{pct(m['AssA'])} | {pct(m['IDF1'])} | {pct(m['MOTA'])} | {m['IDSW']} | {m['CLR_FP']} | "
                         f"{m['CLR_FN']} |")
    lines += ["", "## Default threshold (0.6, every tracker's own MOT17 setting)", "",
              "| detector | tracker | HOTA | IDF1 | MOTA | ID switches |", "|---|---|---|---|---|---|"]
    for d in detectors:
        for t in trackers:
            r = by.get((d, t))
            if r is None:
                continue
            m = r["evaluation"]["default"]["metrics"]
            lines.append(f"| {d} | {t} | {pct(m['HOTA'])} | {pct(m['IDF1'])} | {pct(m['MOTA'])} | {m['IDSW']} |")
    tm = res.get("timing") or {}
    if tm:
        lines += ["", f"## CPU time per frame ({tm['detector']} detections, val_half, tuned threshold, one process)", "",
                  "| tracker | association median / p90 ms | camera motion median / p90 ms |", "|---|---|---|"]
        for t, v in tm["trackers"].items():
            cm = v.get("camera_motion_ms")
            lines.append(f"| {t} | {v['tracker_ms']['median']:.2f} / {v['tracker_ms']['p90']:.2f} | "
                         f"{'–' if not cm else f'{cm['median']:.1f} / {cm['p90']:.1f}'} |")
        lines += ["", f"Camera motion: BoT-SORT's sparse optical flow on the full-resolution MOT17 frames "
                      f"(downscaled {res['camera_motion']['downscale']}x inside), JPEG decoding not included."]
    chk = res.get("milestone3_check") or {}
    if chk:
        lines += ["", "Consistency: ByteTrack here vs Milestone 3's published result (same code and data): "
                  + ", ".join(f"{d} {'identical' if ok else 'DIFFERENT'}" for d, ok in chk.items()) + "."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone9_trackers.yaml")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--trackers", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config, args.override)
    detectors = args.detectors or cfg["detectors"]
    trackers = args.trackers or list(cfg["trackers"])
    full = not args.detectors and not args.trackers and not args.override
    run_dir = REPO_ROOT / cfg["output_dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), run_dir / "run.log")
    t_all = time.monotonic()

    root = REPO_ROOT / cfg["data"]["root"]
    seqs = [load_sequence(root / n) for n in cfg["data"]["sequences"]]
    for d in detectors:
        cache = REPO_ROOT / cfg["detections"]["cache_dir"] / d
        missing = [s.name for s in seqs if not (cache / f"{s.name}.npz").is_file()]
        if missing:
            log.error("no cached detections for %s in %s; run: python scripts/detect_mot17.py --model %s", missing, cache, d)
            return 1
    tune_split, eval_split = cfg["tuning"]["split"], cfg["evaluation"]["split"]
    for sp in (tune_split, eval_split):
        for s in seqs:
            write_gt_split(s, sp, run_dir / "gt" / sp)

    # 1. camera motion for BoT-SORT (once; cached across runs)
    cm = cfg["camera_motion"]
    needs_cmc = any(uses_camera_motion(cfg, t) for t in trackers)
    gmc_dir = REPO_ROOT / cfg["output_dir"] / "camera_motion"
    warps_paths, gmc_ms = None, []
    if needs_cmc:
        warps_paths = {}
        for s in seqs:
            for sp in (tune_split, eval_split):
                t0 = time.monotonic()
                _, ms = camera_motion(s, sp, cm["method"], cm["downscale"], gmc_dir)
                warps_paths[f"{s.name}|{sp}"] = str(gmc_dir / f"{s.name}_{sp}_{cm['method']}_ds{cm['downscale']}.npz")
                if sp == eval_split:
                    gmc_ms += ms
                log.info("camera motion %s %s: %d frames, median %.1f ms (%.0f s)", s.name, sp, len(ms),
                         float(np.median(ms)), time.monotonic() - t0)

    # 2. tuning and evaluation, detector x tracker jobs in parallel
    jobs = [(d, t) for d in detectors for t in trackers]
    log.info("%d jobs (%s x %s), %d workers", len(jobs), detectors, trackers, cfg["workers"])
    results = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = {ex.submit(job, cfg, d, t, str(run_dir), warps_paths): (d, t) for d, t in jobs}
        for fut in as_completed(futs):
            d, t = futs[fut]
            try:
                r = fut.result()
            except Exception:
                log.exception("job %s x %s failed", d, t)
                return 1
            m = r["evaluation"]["tuned"]["metrics"]
            log.info("%-17s %-15s thr %.2f: HOTA %.1f IDF1 %.1f MOTA %.1f IDSW %d", d, t, r["best_track_thresh"],
                     100 * m["HOTA"], 100 * m["IDF1"], 100 * m["MOTA"], m["IDSW"])
            results.append(r)
    order = {(d, t): k for k, (d, t) in enumerate(jobs)}
    results.sort(key=lambda r: order[(r["detector"], r["tracker"])])

    # 3. timing pass, one process, nothing else running
    td = cfg["timing_detector"]
    timing = None
    if td in detectors:
        cache = REPO_ROOT / cfg["detections"]["cache_dir"] / td
        dets = {s.name: load_detections(cache / f"{s.name}.npz") for s in seqs}
        timing = {"detector": td, "split": eval_split, "trackers": {}}
        for t in trackers:
            r = next(x for x in results if x["detector"] == td and x["tracker"] == t)
            tcfg = TrackerConfig.from_dict({**cfg["tracker"], **cfg["trackers"][t], "track_thresh": r["best_track_thresh"]})
            ms = []
            for s in seqs:
                w = np.load(warps_paths[f"{s.name}|{eval_split}"])["warps"] if uses_camera_motion(cfg, t) else None
                ms += track(s, dets[s.name], eval_split, tcfg, cfg["prompt"], w)[1]
            cmc = uses_camera_motion(cfg, t)
            timing["trackers"][t] = {"tracker_ms": stats(ms), "camera_motion_ms": stats(gmc_ms) if cmc else None}
            log.info("timing %-15s association median %.2f ms/frame%s", t, timing["trackers"][t]["tracker_ms"]["median"],
                     f", camera motion {np.median(gmc_ms):.1f} ms" if cmc else "")

    # 4. consistency with Milestone 3 (ByteTrack, same code path and data)
    check = {}
    for r in results:
        if r["tracker"] != "bytetrack":
            continue
        m3 = REPO_ROOT / "benchmarks" / "milestone3" / f"m3_mot17_{r['detector']}.json"
        if m3.is_file():
            ref = json.loads(m3.read_text())["evaluation"]["tuned"]["metrics"]
            check[r["detector"]] = all(abs(ref[k] - r["evaluation"]["tuned"]["metrics"][k]) < 1e-9 for k in TABLE_FIELDS)

    res = {"run_name": cfg["run_name"], "measured": True, "date": datetime.now().isoformat(timespec="seconds"),
           "config": args.config, "environment": collect_environment(), "detectors": detectors,
           "trackers": {t: cfg["trackers"][t] for t in trackers}, "official_code": OFFICIAL,
           "camera_motion": cm, "protocol": {
               "tuning_split": tune_split, "evaluation_split": eval_split, "tuning_metric": cfg["tuning"]["metric"],
               "grid": cfg["tuning"]["track_thresh"], "sequences": [s.name for s in seqs],
               "frames": {sp: sum(len(split_frames(s.length, sp)) for s in seqs) for sp in (tune_split, eval_split)},
               "shared_tracker_settings": cfg["tracker"], "trackeval_commit": TRACKEVAL_COMMIT},
           "results": results, "timing": timing, "milestone3_check": check,
           "seconds": round(time.monotonic() - t_all)}
    (run_dir / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_summary(res, run_dir / "summary.md")
    if full:
        bdir = REPO_ROOT / cfg["benchmark_dir"]
        bdir.mkdir(parents=True, exist_ok=True)
        slim = copy.deepcopy(res)
        for r in slim["results"]:
            for ev in r["evaluation"].values():
                ev["per_sequence"] = {s: {k: v[k] for k in ("HOTA", "IDF1", "MOTA", "IDSW")} for s, v in ev["per_sequence"].items()}
        (bdir / "m9_trackers.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
        write_summary(res, bdir / "m9_trackers.md")
        log.info("published %s", bdir.relative_to(REPO_ROOT))
    log.info("wrote %s (%.0f min); Milestone 3 check: %s", run_dir.relative_to(REPO_ROOT), res["seconds"] / 60, check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
