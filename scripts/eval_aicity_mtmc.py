#!/usr/bin/env python
"""Cross-camera vehicle tracking on CityFlowV2, scored with the official AI City evaluator (RQ4).

Input: the per-camera caches of ``process_aicity.py`` (local tracks + CLIP embeddings). Nothing is trained.
``aicity.appearance`` (Milestone 9) picks the appearance feature: ``clip`` (default) or the name of one of
``reid.extra_embedders`` stored in the cache (e.g. a FastReID vehicle model); colours always come from CLIP.

1. Every local track becomes a tracklet: mean-centred CLIP descriptor (as in Milestone 5), time span on the
   scenario clock, and CLIP zero-shot colour probabilities of its mean embedding.
2. Transit-time windows between cameras are learned from the train ground truth.
3. Tracklets are linked online (``IdentityManager``, in the order they end) with four cue sets:
     appearance       appearance only (no time constraints at all)
     time             + generic time rules: one place at a time per camera, transit 0-300 s between cameras
     topology         + transit windows learned from the train ground truth (pairs never seen: generic rule)
     topology_colour  + colour consistency (the colour distributions must overlap)
   and an **oracle** links each tracklet by the ground-truth vehicle it overlaps most: the best any
   association can do with these detections and local tracks.
4. Each variant's threshold(s) are tuned on train (S01, S03, S04); validation (S02, S05) is evaluated once
   with the tuned values, overall and per scenario.
5. Post hoc (``validation_curves``, never used to choose anything): validation IDF1 over the whole threshold
   grid for every variant, so cues can be compared at equal thresholds. The learned windows come from the
   train ground truth, so tuning them on train is optimistic and shifts their chosen threshold.

Usage:
    python scripts/eval_aicity_mtmc.py [--workers 8]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # many worker processes; NumPy does only small products here

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lgep.evaluation.aicity import (camera_clock, learn_transit_windows, official_mtmc_eval, oracle_track_ids,
                                    read_gt, write_predictions)
from lgep.reid import AssociationConfig, CameraTopology, IdentityManager
from lgep.reid.features import TrackletRecord
from lgep.reid.identity import TransitWindow
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_aicity_mtmc")
REPO_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "appearance": {"use_time": False, "learned": False, "use_attributes": False},
    "time": {"use_time": True, "learned": False, "use_attributes": False},
    "topology": {"use_time": True, "learned": True, "use_attributes": False},
    "topology_colour": {"use_time": True, "learned": True, "use_attributes": True},
}
METRICS = ("idf1", "idp", "idr", "mota", "num_frames", "idtp", "idfp", "idfn")


@dataclass
class Cam:
    camera: str
    scenario: str
    rows: np.ndarray  # frame, local ID, x, y, w, h, score
    size: tuple[int, int]
    recs: list[TrackletRecord]
    clip_means: list = None  # per tracklet: mean raw CLIP embedding (for colours), None without samples


# ------------------------------------------------------------------------------------------ data
def unit_mean(emb: np.ndarray) -> np.ndarray:
    """As TrackletRecord.mean_embedding: L2-normalised mean of the raw embeddings."""
    m = np.mean(list(emb.astype(np.float32)), axis=0)
    return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)


def load_split(cfg: dict, split: str, scenarios: list[str]) -> list[Cam]:
    a = cfg["aicity"]
    root, cache = REPO_ROOT / a["root"], REPO_ROOT / a["cache_dir"]
    feat = a.get("appearance", "clip")
    suffix = "" if feat == "clip" else f"__{feat}"
    cams = []
    for scen in scenarios:
        for cdir in sorted((root / split / scen).glob("c*")):
            name = f"{scen}_{cdir.name}"
            meta = json.loads((cache / f"{name}.json").read_text())
            z = np.load(cache / f"{name}.npz")
            if f"camera_mean{suffix}" not in z.files and "camera_mean" in z.files:
                raise KeyError(f"{name}: no {feat!r} embeddings in {cache} (run process_aicity.py with it)")
            center = z[f"camera_mean{suffix}"] if f"camera_mean{suffix}" in z.files else None
            recs, clip_means = [], []
            for i, t in enumerate(meta["tracklets"]):
                r = TrackletRecord(key=t["key"], camera_id=cdir.name, local_id=t["local_id"], phrase="vehicle",
                                   first_frame=t["first_frame"], last_frame=t["last_frame"], t_start=t["t_start"],
                                   t_end=t["t_end"], frames_seen=t["frames_seen"])
                if f"trk_{i}{suffix}" in z.files:
                    r.embeddings = list(z[f"trk_{i}{suffix}"].astype(np.float32))
                    r.center = center
                clip_means.append(unit_mean(z[f"trk_{i}"]) if f"trk_{i}" in z.files else None)
                recs.append(r)
            h, w = cv2.imread(str(cdir / "roi.jpg"), cv2.IMREAD_GRAYSCALE).shape
            cams.append(Cam(cdir.name, scen, z["rows"], (w, h), recs, clip_means))
    return cams


def add_colours(cams: list[Cam], cfg: dict) -> list[str]:
    """CLIP zero-shot colour distribution of each tracklet's mean (raw) embedding -> ``rec.attributes``."""
    from lgep.query.attributes import QueryScorer
    from lgep.reid.embedder import build_embedder

    scorer = QueryScorer(build_embedder(cfg["reid"]["embedder"]))
    pairs = [(r, m) for c in cams for r, m in zip(c.recs, c.clip_means) if r.embeddings and m is not None]
    recs = [r for r, _ in pairs]
    probs = scorer.colour_probs(np.stack([m for _, m in pairs]), cfg["aicity"]["colour_template"])
    for r, p in zip(recs, probs):
        r.attributes = p.astype(np.float32)
    return list(scorer.colours)


# ------------------------------------------------------------------------------------------ association
def association_config(cfg: dict, variant: str, threshold: float, overlap: float) -> AssociationConfig:
    v, a = VARIANTS[variant], cfg["aicity"]["association"]
    return AssociationConfig(appearance_threshold=threshold, member_reduction=a["member_reduction"],
                             use_time=v["use_time"], use_phrase=False,
                             max_gap_same_camera_s=a["max_gap_same_camera_s"], min_samples=a["min_samples"],
                             use_attributes=v["use_attributes"], attribute_min_overlap=overlap)


def topology(cfg: dict, learned: dict | None) -> CameraTopology:
    pairs = {p: TransitWindow(w["min_s"], w["max_s"], w["overlapping"]) for p, w in (learned or {}).items()}
    return CameraTopology(pairs, TransitWindow(**cfg["aicity"]["default_transit"]))


def associate(cams: list[Cam], cfg: dict, variant: str, threshold: float, overlap: float,
              learned: dict) -> tuple[dict, dict]:
    """(camera, local ID) -> global ID (unique across scenarios), and statistics."""
    acfg = association_config(cfg, variant, threshold, overlap)
    topo = topology(cfg, learned if VARIANTS[variant]["learned"] else None)
    ids, stats = {}, {"tracklets": 0, "linked_tracklets": 0, "assign_ms": []}
    for k, scen in enumerate(sorted({c.scenario for c in cams})):
        recs = sorted((r for c in cams if c.scenario == scen for r in c.recs if r.n_samples >= acfg.min_samples),
                      key=lambda r: (r.t_end, r.key))
        mgr = IdentityManager(acfg, topo)
        t0 = time.perf_counter()
        for r in recs:
            mgr.assign(r)
        stats["assign_ms"].append((time.perf_counter() - t0) * 1e3 / max(len(recs), 1))
        for r in recs:
            ids[(r.camera_id, r.local_id)] = (k + 1) * 1_000_000 + mgr.global_id_of(r.key)
        stats["tracklets"] += len(recs)
        stats["linked_tracklets"] += sum(len(i.members) for i in mgr.identities.values() if len(i.cameras) > 1)
    multi = {}
    for (cam, _), g in ids.items():
        multi.setdefault(g, set()).add(cam)
    stats["multi_camera_ids"] = sum(len(v) > 1 for v in multi.values())
    stats["assign_ms"] = round(float(np.mean(stats["assign_ms"])), 3)
    return ids, stats


def oracle_ids(cams: list[Cam], gt: np.ndarray) -> dict:
    ids, nxt = {}, itertools.count(10_000_000)
    for c in cams:
        lab = oracle_track_ids(c.rows, gt[gt[:, 0] == int(c.camera[1:])])
        for r in c.recs:
            ids[(c.camera, r.local_id)] = lab.get(r.local_id, next(nxt))
    return ids


# ------------------------------------------------------------------------------------------ evaluation
_G: dict = {}  # data shared with forked worker processes


def score(ids: dict, split: str, tag: str, gt_file: Path | None = None) -> dict:
    pred = _G["tmp"] / f"{tag}.txt"
    lines = write_predictions(pred, [(c.camera, c.rows, c.size) for c in _G["cams"][split]], ids)
    if lines == 0:  # nothing linked across cameras; the official script cannot read an empty file
        pred.unlink()
        return {"idf1": 0.0, "idp": None, "idr": 0.0, "mota": None, "idtp": 0, "prediction_lines": 0}
    m = official_mtmc_eval(_G["root"], pred, dstype=split, ground_truth=gt_file)
    if not _G["keep"].get(tag):
        pred.unlink()
    return {**{k: m[k] for k in METRICS if k in m}, "prediction_lines": lines}


def grid_point(point: tuple) -> dict:
    split, variant, thr, overlap = point
    t0 = time.perf_counter()
    ids, st = associate(_G["cams"][split], _G["cfg"], variant, thr, overlap, _G["learned"])
    res = score(ids, split, f"{split}_{variant}_{thr}_{overlap}")
    return {"split": split, "variant": variant, "threshold": thr, "attribute_min_overlap": overlap, **res, **st,
            "seconds": round(time.perf_counter() - t0, 1)}


def scored(job: tuple) -> dict:
    ids, split, tag, gt_file = job
    return score(ids, split, tag, gt_file)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone5_cityflow.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-curves", action="store_true", help="skip the post-hoc validation curves")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config, args.override)
    a = cfg["aicity"]
    root = REPO_ROOT / a["root"]
    out = REPO_ROOT / cfg["output_dir"] / f"aicity_mtmc_{cfg['run_name']}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), out / "eval.log")
    splits = {"train": a["train_scenarios"], "validation": a["validation_scenarios"]}

    t_all = time.monotonic()
    cams = {s: load_split(cfg, s, scen) for s, scen in splits.items()}
    colours = add_colours(cams["train"] + cams["validation"], cfg)
    clock_train = camera_clock(root, "train", splits["train"], a["fps"])
    gt = {s: read_gt(root / "eval" / f"ground_truth_{s}.txt") for s in splits}
    learned = learn_transit_windows(gt["train"], clock_train, a["transit_margin_s"])
    for s, cs in cams.items():
        log.info("%s: %d cameras, %d tracklets (%d with >= %d samples), %d ground-truth vehicles", s, len(cs),
                 sum(len(c.recs) for c in cs), sum(r.n_samples >= a["association"]["min_samples"] for c in cs
                                                   for r in c.recs), a["association"]["min_samples"],
                 len(np.unique(gt[s][:, 1])))
    val_cams = {c.camera for c in cams["validation"]}
    log.info("learned %d transit windows from train; %d of them between validation cameras", len(learned),
             sum(p[0] in val_cams and p[1] in val_cams for p in learned))

    # per-scenario ground truth for the validation breakdown
    gt_files = {}
    for scen in splits["validation"]:
        nums = {int(c.camera[1:]) for c in cams["validation"] if c.scenario == scen}
        p = out / f"ground_truth_validation_{scen}.txt"
        g = np.loadtxt(root / "eval" / "ground_truth_validation.txt", dtype=np.int64).reshape(-1, 9)
        np.savetxt(p, g[np.isin(g[:, 0], list(nums))], fmt="%d")
        gt_files[scen] = p

    _G.update(cams=cams, cfg=cfg, learned=learned, root=root, tmp=out, keep={})
    ctx = mp.get_context("fork")

    # ---- tune on train
    grid = [("train", v, round(t, 2), None) for v in ("appearance", "time", "topology")
            for t in a["tune_thresholds"]]
    grid += [("train", "topology_colour", round(t, 2), o) for t in a["tune_thresholds"]
             for o in a["tune_attribute_overlaps"]]
    log.info("tuning: %d settings on train with %d workers", len(grid), args.workers)
    with ctx.Pool(args.workers) as pool:
        tuning = []
        for r in pool.imap_unordered(grid_point, [(s, v, t, o if o is not None else 0.0) for s, v, t, o in grid]):
            tuning.append(r)
            log.info("  train %-15s thr %.2f ovl %.1f: IDF1 %.4f (IDP %s IDR %.4f) | %d multi-camera IDs",
                     r["variant"], r["threshold"], r["attribute_min_overlap"], r["idf1"],
                     "-" if r["idp"] is None else f"{r['idp']:.4f}", r["idr"], r["multi_camera_ids"])
    best = {v: max((r for r in tuning if r["variant"] == v), key=lambda r: (r["idf1"], -r["threshold"]))
            for v in VARIANTS}

    # ---- validation, once per variant with its tuned setting; association timed in this process
    jobs, val_stats = [], {}
    for v, b in best.items():
        ids, st = associate(cams["validation"], cfg, v, b["threshold"], b["attribute_min_overlap"], learned)
        val_stats[v] = st
        _G["keep"][f"validation_{v}"] = True
        jobs.append((ids, "validation", f"validation_{v}", None))
        jobs += [(ids, "validation", f"validation_{v}_{s}", gt_files[s]) for s in splits["validation"]]
    for s in splits:
        o = oracle_ids(cams[s], gt[s])
        jobs.append((o, s, f"{s}_oracle", None))
        if s == "validation":
            jobs += [(o, s, f"validation_oracle_{x}", gt_files[x]) for x in splits["validation"]]
    with ctx.Pool(args.workers) as pool:
        res = pool.map(scored, jobs)
    final = {job[2]: r for job, r in zip(jobs, res)}
    curves = []
    if not args.no_curves:
        vgrid = [("validation", v, round(t, 2), best[v]["attribute_min_overlap"]) for v in VARIANTS
                 for t in a["tune_thresholds"]]
        log.info("post-hoc validation curves: %d settings", len(vgrid))
        with ctx.Pool(args.workers) as pool:
            curves = sorted(pool.imap_unordered(grid_point, vgrid),
                            key=lambda r: (r["variant"], r["threshold"]))
    for tag, r in final.items():
        log.info("%-34s IDF1 %.4f  IDP %.4f  IDR %.4f", tag, r["idf1"], r["idp"], r["idr"])

    report = {
        "protocol": "CityFlowV2 (AI City Challenge 2022 Track 1); thresholds tuned on train (S01, S03, S04) with "
                    "the official evaluator; validation (S02, S05) evaluated once per variant",
        "evaluator": "data/aicity/cityflowv2/eval/eval.py (official, unchanged; see lgep.evaluation.aicity)",
        "appearance_feature": a.get("appearance", "clip"), "cache_dir": a["cache_dir"],
        "detector": a["detector"], "prompt": a["prompt"], "tracker": cfg["tracker"],
        "sampling": cfg["reid"]["sampling"], "association": a["association"],
        "default_transit": a["default_transit"], "transit_margin_s": a["transit_margin_s"],
        "colours": colours, "colour_template": a["colour_template"],
        "learned_windows": {f"{p[0]}->{p[1]}": w for p, w in sorted(learned.items())},
        "data": {s: {"cameras": len(cams[s]), "tracklets": sum(len(c.recs) for c in cams[s]),
                     "gt_vehicles": int(len(np.unique(gt[s][:, 1])))} for s in splits},
        "tuning": sorted(tuning, key=lambda r: (r["variant"], r["threshold"], r["attribute_min_overlap"])),
        "selected": {v: {"threshold": b["threshold"], "attribute_min_overlap": b["attribute_min_overlap"],
                         "train_idf1": b["idf1"]} for v, b in best.items()},
        "validation": {v: {"all": final[f"validation_{v}"],
                           **{s: final[f"validation_{v}_{s}"] for s in splits["validation"]},
                           "association": val_stats[v]} for v in VARIANTS},
        "validation_curves_post_hoc": curves,
        "oracle": {"train": final["train_oracle"], "validation": {"all": final["validation_oracle"],
                   **{s: final[f"validation_oracle_{s}"] for s in splits["validation"]}}},
        "processing": {f"{c.scenario}_{c.camera}": json.loads((REPO_ROOT / a["cache_dir"] /
                       f"{c.scenario}_{c.camera}.json").read_text())["timing_ms"] for s in splits for c in cams[s]},
        "environment": collect_environment(),
        "seconds": round(time.monotonic() - t_all, 1),
    }
    bdir = REPO_ROOT / cfg["benchmark_dir"]
    bdir.mkdir(parents=True, exist_ok=True)
    name = cfg["run_name"]  # m5_cityflow for Milestone 5's config
    (bdir / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (bdir / f"{name}.md").write_text(markdown(report, splits["validation"], name), encoding="utf-8")
    log.info("wrote %s (%.0f s)", bdir / f"{name}.json", report["seconds"])
    return 0


def markdown(r: dict, val_scen: list[str], name: str) -> str:
    f = lambda x: f"{x:.4f}"  # noqa: E731
    lines = [f"# Cross-camera vehicle tracking on CityFlowV2 (official evaluator): {name}", "",
             f"Protocol: {r['protocol']}. Appearance feature: {r.get('appearance_feature', 'clip')}.", "",
             "| cues | threshold (train) | train IDF1 | validation IDF1 | IDP | IDR | "
             + " | ".join(f"{s} IDF1" for s in val_scen) + " | multi-camera IDs | ms / tracklet |",
             "|---|---|---|---|---|---|" + "---|" * len(val_scen) + "---|---|"]
    for v, sel in r["selected"].items():
        val = r["validation"][v]
        thr = f"{sel['threshold']:.2f}" + (f", overlap {sel['attribute_min_overlap']:.1f}"
                                         if VARIANTS[v]["use_attributes"] else "")
        lines.append(f"| {v} | {thr} | {f(sel['train_idf1'])} | **{f(val['all']['idf1'])}** | {f(val['all']['idp'])} | "
                     f"{f(val['all']['idr'])} | " + " | ".join(f(val[s]["idf1"]) for s in val_scen)
                     + f" | {val['association']['multi_camera_ids']} | {val['association']['assign_ms']} |")
    o = r["oracle"]
    lines.append(f"| oracle association (upper bound) | - | {f(o['train']['idf1'])} | {f(o['validation']['all']['idf1'])} | "
                 f"{f(o['validation']['all']['idp'])} | {f(o['validation']['all']['idr'])} | "
                 + " | ".join(f(o["validation"][s]["idf1"]) for s in val_scen) + " | - | - |")
    if r.get("validation_curves_post_hoc"):
        thrs = sorted({c["threshold"] for c in r["validation_curves_post_hoc"]})
        lines += ["", "Post hoc (not used for any choice): validation IDF1 at every threshold.", "",
                  "| cues | " + " | ".join(f"{t:.2f}" for t in thrs) + " |", "|---|" + "---|" * len(thrs)]
        for v in r["selected"]:
            got = {c["threshold"]: c["idf1"] for c in r["validation_curves_post_hoc"] if c["variant"] == v}
            lines.append(f"| {v} | " + " | ".join(f"{got[t]:.3f}" if t in got else "-" for t in thrs) + " |")
    lines += ["", "Tracklets: " + ", ".join(f"{s} {d['tracklets']} in {d['cameras']} cameras "
                                            f"({d['gt_vehicles']} annotated vehicles)" for s, d in r["data"].items()),
              "", f"Generated by scripts/eval_aicity_mtmc.py; full tuning curves in {name}.json.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
