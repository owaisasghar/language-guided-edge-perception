#!/usr/bin/env python
"""Interim Re-ID evaluation: re-link fragmented ByteTrack tracks on MOT17 by appearance (CLIP) and time.

For each detector (``relink.detectors``), using its Milestone 3 detections and tuned ``track_thresh``:
  1. track each sequence half with ByteTrack (as scripts/eval_mot17.py does),
  2. sample crops per track (``reid.sampling``) and embed them (``reid.embedder``),
  3. link finished tracklets within each sequence with ``IdentityManager`` (``reid.association``),
  4. score the re-labelled tracks with TrackEval.
``appearance_threshold`` x ``max_gap_same_camera_s`` is tuned on train-half (the grid includes "never link",
so tuning can choose ByteTrack alone); val-half is reported before (ByteTrack alone) and after re-linking.

Mean-centring is leak-free: each sequence's camera mean comes from its train half ("earlier footage") and is
applied to both halves. Also reported, per crop mode (``relink.ablation_crop_modes``): how well similarity
separates tracklets of the same person from tracklets of different people (AUROC on val-half, raw and
mean-centred; tracklets matched to ground truth by IoU >= 0.5).

Writes outputs/milestone5/<run>/ and benchmarks/milestone5/m5_relink_<detector>.{json,md}.

Usage:
    python scripts/eval_reid_mot17.py
    python scripts/eval_reid_mot17.py --detectors yoloworld_m
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from eval_mot17 import EMPTY, load_detections
from PIL import Image
from run_detector import stats
from scipy.optimize import linear_sum_assignment

from lgep.evaluation.mot import TRACKEVAL_COMMIT, evaluate_mot, load_sequence, split_frames, write_gt_split, write_mot_results
from lgep.reid import AssociationConfig, CameraTopology, IdentityManager, SamplingConfig, TrackFeatureBank
from lgep.reid.embedder import build_embedder
from lgep.reid.features import crop
from lgep.reid.metrics import same_vs_different_auroc
from lgep.tracking import CameraTracker, TrackerConfig
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_reid_mot17")
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER = "tracker"


def track_and_embed(seq, dets, split, tcfg: TrackerConfig, phrase, embedder, sampling: SamplingConfig):
    """ByteTrack over one split, with appearance samples. Returns (per-frame tracks, feature bank, timings)."""
    frames = split_frames(seq.length, split)
    tracker = CameraTracker(seq.name, replace(tcfg, frame_rate=seq.fps))
    bank = TrackFeatureBank(sampling)
    per_frame, embed_ms, n_crops = {}, [], 0
    for f in frames:
        r = tracker.update({phrase: dets.get(f, EMPTY)})
        fr = f - frames.start + 1
        per_frame[fr] = r.tracks
        want = bank.observe(seq.name, fr, fr / seq.fps, r.tracks, (seq.width, seq.height))
        if want:
            image = np.asarray(Image.open(seq.image(f)).convert("RGB"))
            t0 = time.perf_counter()
            feats = embedder.embed_images([crop(image, tr.box_xyxy) for tr in want])
            embed_ms.append((time.perf_counter() - t0) * 1e3 / len(want))
            n_crops += len(want)
            for tr, e in zip(want, feats):
                bank.add(tr.key, fr, e)
    return per_frame, bank, {"crops": n_crops, "embed_ms_per_crop": stats(embed_ms)}


def relink(banks: dict, acfg: AssociationConfig) -> dict[str, int]:
    """Global ID per track key; sequences are separate cameras with no cross-camera links."""
    ids, offset = {}, 0
    for seq_name, bank in banks.items():
        mgr = IdentityManager(acfg, CameraTopology(default=None))
        for rec in sorted(bank.tracks.values(), key=lambda r: r.t_end):
            mgr.assign(rec)
        for key in bank.tracks:
            ids[key] = mgr.global_id_of(key) + offset
        offset += 100000
    return ids


def write_results(out_dir: Path, per_frame_by_seq: dict, id_of: dict | None) -> None:
    for seq_name, per_frame in per_frame_by_seq.items():
        rows = []
        for fr, tracks in per_frame.items():
            for t in tracks:
                x1, y1, x2, y2 = t.box_xyxy
                tid = id_of[t.key] if id_of else t.local_id
                rows.append((fr, tid, x1, y1, x2 - x1, y2 - y1, t.score))
        write_mot_results(out_dir / TRACKER / "data" / f"{seq_name}.txt", rows)


def score(out_dir, gt_dir, lengths, per_frame_by_seq, id_of) -> dict:
    write_results(out_dir, per_frame_by_seq, id_of)
    m = evaluate_mot(gt_dir, out_dir, TRACKER, lengths)["combined"]
    return {k: m[k] for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")}


def gt_identity(seq, split, per_frame) -> dict[str, int]:
    """Majority ground-truth pedestrian ID per track key (IoU >= 0.5, one-to-one per frame)."""
    frames = split_frames(seq.length, split)
    gt = np.loadtxt(seq.root / "gt/gt.txt", delimiter=",", ndmin=2)
    gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    votes: dict[str, dict[int, int]] = {}
    counts: dict[str, int] = {}
    for fr, tracks in per_frame.items():
        g = gt[gt[:, 0] == fr + frames.start - 1]
        for t in tracks:
            counts[t.key] = counts.get(t.key, 0) + 1
        if not len(g) or not tracks:
            continue
        gb = np.column_stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]])
        tb = np.array([t.box_xyxy for t in tracks])
        ix = np.clip(np.minimum(tb[:, None, 2], gb[None, :, 2]) - np.maximum(tb[:, None, 0], gb[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(tb[:, None, 3], gb[None, :, 3]) - np.maximum(tb[:, None, 1], gb[None, :, 1]), 0, None)
        inter = ix * iy
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
        iou = inter / (area(tb)[:, None] + area(gb)[None, :] - inter)
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] >= 0.5:
                v = votes.setdefault(tracks[i].key, {})
                v[int(g[j, 1])] = v.get(int(g[j, 1]), 0) + 1
    out = {}
    for key, v in votes.items():
        gid, n = max(v.items(), key=lambda kv: kv[1])
        if n >= 0.5 * counts[key]:
            out[key] = gid
    return out


def embed_all(seqs, cache, splits, tcfg, phrase, embedder, sampling) -> dict:
    data = {}
    for sp in splits:
        per_frame, banks, timing = {}, {}, []
        for s in seqs:
            pf, bank, tm = track_and_embed(s, load_detections(cache / f"{s.name}.npz"), sp, tcfg, phrase, embedder, sampling)
            per_frame[s.name], banks[s.name] = pf, bank
            timing.append(tm)
        data[sp] = (per_frame, banks, timing)
    return data


def set_centres(data: dict, calib_split: str, on: bool) -> None:
    """Centre every tracklet with its camera's mean from ``calib_split`` (earlier footage), or not at all."""
    calib = {seq: bank.camera_mean(seq) for seq, bank in data[calib_split][1].items()}
    for _, banks, _ in data.values():
        for seq, bank in banks.items():
            for rec in bank.tracks.values():
                rec.center = calib[seq] if on else None


def separability(seqs, split, data) -> dict:
    pf, banks, _ = data[split]
    feats, labels, groups = [], [], []
    for s in seqs:
        truth = gt_identity(s, split, pf[s.name])
        for k, rec in banks[s.name].tracks.items():
            if rec.n_samples and k in truth:
                feats.append(rec.descriptor)
                labels.append(truth[k])
                groups.append(s.name)
    return same_vs_different_auroc(np.array(feats), labels, groups)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone5_reid.yaml")
    ap.add_argument("--detectors", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    run_dir = REPO_ROOT / cfg["output_dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), run_dir / "run.log")
    rcfg = cfg["reid"]
    sampling = SamplingConfig.from_dict(rcfg["sampling"])
    base_acfg = AssociationConfig.from_dict(rcfg["association"])
    seqs = [load_sequence(REPO_ROOT / cfg["data"]["root"] / n) for n in cfg["data"]["sequences"]]
    splits = (cfg["tuning"]["split"], cfg["evaluation"]["split"])
    gt_dirs = {sp: run_dir / "gt" / sp for sp in splits}
    for sp, d in gt_dirs.items():
        for s in seqs:
            write_gt_split(s, sp, d)
    lengths = {sp: {s.name: len(split_frames(s.length, sp)) for s in seqs} for sp in splits}
    try:
        embedder = build_embedder(rcfg["embedder"])
        for det in args.detectors or cfg["relink"]["detectors"]:
            m3 = json.loads((REPO_ROOT / f"benchmarks/milestone3/m3_mot17_{det}.json").read_text())
            tcfg = TrackerConfig.from_dict({**cfg["tracker"], "track_thresh": m3["best_track_thresh"]})
            cache = REPO_ROOT / cfg["detections"]["cache_dir"] / det
            log.info("%s: track_thresh %.2f (Milestone 3), embedding crops ...", det, tcfg.track_thresh)
            configured = embedder.cfg.crop_mode
            modes = [configured] + [m for m in cfg["relink"].get("ablation_crop_modes", []) if m != configured]
            ablation, data = [], None
            for mode in modes:
                embedder.cfg = replace(embedder.cfg, crop_mode=mode)
                d = embed_all(seqs, cache, splits, tcfg, cfg["prompt"], embedder, sampling)
                row = {"crop_mode": mode}
                for name, on in (("raw", False), ("centred", True)):
                    set_centres(d, splits[0], on)
                    row[name] = separability(seqs, splits[1], d)
                ablation.append(row)
                log.info("%s crop %-11s AUROC raw %.3f, centred %.3f (%d same / %d different pairs)", det, mode,
                         row["raw"]["auroc"] or float("nan"), row["centred"]["auroc"] or float("nan"),
                         row["raw"]["pairs_same"], row["raw"]["pairs_different"])
                if mode == configured:
                    data = d
            embedder.cfg = replace(embedder.cfg, crop_mode=configured)
            set_centres(data, splits[0], sampling.center)
            # tune on the first split
            tr_pf, tr_banks, _ = data[splits[0]]
            grid = cfg["relink"]["grid"]
            tuning = []
            for thr, gap in itertools.product(grid["appearance_threshold"], grid["max_gap_same_camera_s"]):
                acfg = replace(base_acfg, appearance_threshold=thr, max_gap_same_camera_s=gap)
                m = score(run_dir / det / "tune" / f"{thr}_{gap}", gt_dirs[splits[0]], lengths[splits[0]], tr_pf,
                          relink(tr_banks, acfg))
                tuning.append({"appearance_threshold": thr, "max_gap_same_camera_s": gap, **m})
                log.info("%s %s thr %.2f gap %4.0f s: HOTA %.1f IDF1 %.1f IDSW %d", det, splits[0], thr, gap,
                         100 * m["HOTA"], 100 * m["IDF1"], m["IDSW"])
            shutil.rmtree(run_dir / det / "tune", ignore_errors=True)  # 28 x 7 result files
            key = cfg["relink"]["tune_metric"]
            best = max(tuning, key=lambda t: t[key])
            best_acfg = replace(base_acfg, appearance_threshold=best["appearance_threshold"],
                                max_gap_same_camera_s=best["max_gap_same_camera_s"])
            # evaluate
            res = {}
            for sp in splits:
                pf, banks, timing = data[sp]
                ids = relink(banks, best_acfg)
                res[sp] = {"bytetrack": score(run_dir / det / sp / "bytetrack", gt_dirs[sp], lengths[sp], pf, None),
                           "relinked": score(run_dir / det / sp / "relinked", gt_dirs[sp], lengths[sp], pf, ids),
                           "local_tracks": sum(len(b.tracks) for b in banks.values()),
                           "global_ids": len(set(ids.values())),
                           "tracklets_with_samples": sum(1 for b in banks.values() for r in b.tracks.values() if r.n_samples),
                           "crops_embedded": sum(t["crops"] for t in timing),
                           "embed_ms_per_crop_median": round(float(np.median(
                               [t["embed_ms_per_crop"]["median"] for t in timing if t["embed_ms_per_crop"]] or [np.nan])), 3)}
                log.info("%s %s: ByteTrack HOTA %.1f IDF1 %.1f IDSW %d -> relinked HOTA %.1f IDF1 %.1f IDSW %d "
                         "(%d local tracks -> %d IDs)", det, sp, 100 * res[sp]["bytetrack"]["HOTA"],
                         100 * res[sp]["bytetrack"]["IDF1"], res[sp]["bytetrack"]["IDSW"], 100 * res[sp]["relinked"]["HOTA"],
                         100 * res[sp]["relinked"]["IDF1"], res[sp]["relinked"]["IDSW"], res[sp]["local_tracks"],
                         res[sp]["global_ids"])
            sep = next(a for a in ablation if a["crop_mode"] == configured)["centred" if sampling.center else "raw"]
            out = {"measured": True, "environment": collect_environment(), "detector": det,
                   "track_thresh": tcfg.track_thresh, "embedder": embedder.describe(), "sampling": asdict(sampling),
                   "association_base": asdict(base_acfg), "best": best, "tuning": tuning, "results": res,
                   "appearance_separability": sep, "crop_mode_ablation": ablation, "protocol": {"tuning_split": splits[0], "evaluation_split": splits[1],
                                                                "metric": key, "trackeval_commit": TRACKEVAL_COMMIT}}
            bdir = REPO_ROOT / cfg["benchmark_dir"]
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / f"{cfg['run_name']}_{det}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
            (run_dir / f"{det}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        log.exception("Re-ID evaluation failed (log: %s)", run_dir / "run.log")
        return 1
    log.info("Wrote %s", run_dir.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
