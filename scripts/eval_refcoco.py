#!/usr/bin/env python
"""Interim evaluation of language-guided retrieval on RefCOCO+ (zero-shot referring-expression comprehension).

For every expression (e.g. "man in red shirt") the system must return one box; it is correct if
IoU >= 0.5 with the described object (= Rank-1 retrieval among the objects of the image). Systems:

* sentence baselines: the whole expression as the detector prompt, top box (Grounding DINO-T, YOLO-World-M);
* ours: parse the expression (lgep.query) -> detect the parsed category (or the description if no category
  was found) -> re-rank the top candidates with CLIP: weighted sum of detection score, description
  similarity and colour-attribute probability (lgep.query.attributes).

The weights are tuned on the first ``tune_limit`` val expressions; testA (people) and testB (objects) are
evaluated once with the tuned weights, plus ablations (each component alone). Also reported: parser category
accuracy (parsed head category == the target's COCO category), candidate recall (the target is among the
candidates: the ceiling for re-ranking) and latency per expression.

Writes benchmarks/milestone6/m6_refcoco.{json,md}.

Usage:
    python scripts/eval_refcoco.py
    python scripts/eval_refcoco.py --limit 200        # quick run, not published
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from run_detector import stats

from lgep.perception.factory import build_detector
from lgep.query import parse_query
from lgep.query.attributes import QueryScorer, ScoreWeights
from lgep.reid.embedder import build_embedder
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_refcoco")
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = {"val": "validation", "testA": "test-", "testB": "testB"}


def coco_names() -> dict[int, str]:
    ann = REPO_ROOT / "data/coco/annotations/instances_val2017.json"
    return {c["id"]: c["name"] for c in json.loads(ann.read_text())["categories"]}


def load_expressions(root: Path, split: str, limit: int | None) -> list[dict]:
    import pyarrow.parquet as pq

    path = next(p for p in sorted((root / "annotations/data").glob("*.parquet")) if p.name.startswith(SPLIT_FILES[split]))
    out = []
    for r in pq.read_table(path).to_pylist():
        for s in r["sentences"]:
            out.append({"image_id": int(r["image_id"]), "ref_id": int(r["ref_id"]), "sentence": s["sent"],
                        "bbox": [float(v) for v in r["bbox"]], "category_id": int(r["category_id"])})
    return out[:limit] if limit else out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Evaluator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = REPO_ROOT / cfg["refcoco"]["root"]
        self.embedder = build_embedder(cfg["reid"]["embedder"])
        self.scorer = QueryScorer(self.embedder, check_types=False)  # Milestone 6 protocol (no vehicle-type check)
        self.detectors = {}
        self._images = {}
        self._cand_cache = {}

    def detector(self, name):
        if name not in self.detectors:
            self.detectors[name] = build_detector(self.cfg["models"][name], self.cfg.get("device", "auto"),
                                                  base_dir=REPO_ROOT)
        return self.detectors[name]

    def image(self, image_id: int) -> np.ndarray:
        if image_id not in self._images:
            if len(self._images) > 64:
                self._images.clear()
            path = self.root / "images" / f"COCO_train2014_{image_id:012d}.jpg"
            self._images[image_id] = np.asarray(Image.open(path).convert("RGB"))
        return self._images[image_id]

    def sentence_baseline(self, det_name: str, e: dict) -> tuple[bool, float]:
        t0 = time.perf_counter()
        r = self.detector(det_name).detect(self.image(e["image_id"]), [e["sentence"]])
        ms = (time.perf_counter() - t0) * 1e3
        top = max(r.detections, key=lambda d: d.score, default=None)
        return bool(top and iou(top.box_xyxy, e["bbox"]) >= 0.5), ms

    def candidates(self, det_name: str, e: dict, q) -> dict:
        """Candidate boxes + cached score components (independent of the weights)."""
        phrases = q.categories or [q.free_text or e["sentence"]]
        key = (det_name, e["image_id"], tuple(phrases))
        t0 = time.perf_counter()
        if key not in self._cand_cache:
            if len(self._cand_cache) > 4096:
                self._cand_cache.clear()
            img = self.image(e["image_id"])
            r = self.detector(det_name).detect(img, phrases)
            dets = sorted(r.detections, key=lambda d: -d.score)[: self.cfg["query"]["candidates"]]
            boxes = [d.box_xyxy for d in dets]
            crops = [img[int(b[1]):max(int(b[3]), int(b[1]) + 1), int(b[0]):max(int(b[2]), int(b[0]) + 1)] for b in boxes]
            self._cand_cache[key] = {"boxes": boxes, "det": np.array([d.score for d in dets]), "crops": crops, "embs": {}}
        c = self._cand_cache[key]
        if c["boxes"]:  # embed only the regions this expression needs, once per candidate set
            need = self.scorer.regions_needed(q)
            missing = need - set(c["embs"])
            if missing:
                c["embs"].update(self.scorer.embed_regions(c["crops"], missing))
        out = {"boxes": c["boxes"], "hits": np.array([iou(b, e["bbox"]) >= 0.5 for b in c["boxes"]], bool)}
        if c["boxes"]:
            s = self.scorer.score(q, c["embs"], c["det"])
            out.update(det=s["detection"], text=s["text"], attr=s["attributes"])
        out["ms"] = (time.perf_counter() - t0) * 1e3
        return out


def pick(c: dict, w: ScoreWeights) -> bool:
    if not c["boxes"]:
        return False
    total = w.detection * c["det"] + w.text * c["text"] + w.attributes * c["attr"]
    return bool(c["hits"][int(np.argmax(total))])


def accuracy(cands: list[dict], w: ScoreWeights) -> float:
    return float(np.mean([pick(c, w) for c in cands])) if cands else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone6_query.yaml")
    ap.add_argument("--limit", type=int, default=None, help="expressions per split (quick run; not published)")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config, args.override)
    setup_logging(cfg.get("log_level", "INFO"))
    rc = cfg["refcoco"]
    names = coco_names()
    ev = Evaluator(cfg)
    res = {"measured": True, "environment": collect_environment(), "embedder": ev.embedder.describe(),
           "protocol": {"metric": "accuracy: top-1 box IoU >= 0.5 with the described object", "tune_split": rc["tune_split"],
                        "tune_limit": rc["tune_limit"], "candidates": cfg["query"]["candidates"]}, "detectors": {}}
    try:
        for det in rc["candidate_detectors"]:
            # 1. tune the weights on val
            tune = load_expressions(ev.root, rc["tune_split"], min(rc["tune_limit"], args.limit or rc["tune_limit"]))
            cands = [ev.candidates(det, e, parse_query(e["sentence"])) for e in tune]
            grid = [ScoreWeights(*w) for w in itertools.product(rc["grid"]["detection"], rc["grid"]["text"],
                                                                 rc["grid"]["attributes"])]
            scored = sorted(((accuracy(cands, w), w) for w in grid), key=lambda t: -t[0])
            best_acc, best = scored[0]
            log.info("%s: tuned on %d %s expressions -> %s (accuracy %.3f)", det, len(tune), rc["tune_split"], best, best_acc)
            out = {"tuned_weights": asdict(best), "tune_accuracy": best_acc,
                   "tune_top5": [{"accuracy": a, **asdict(w)} for a, w in scored[:5]], "splits": {}}
            # 2. evaluate
            variants = {"ours (tuned)": best, "detection only": ScoreWeights(1, 0, 0), "text only": ScoreWeights(0, 1, 0),
                        "attributes only": ScoreWeights(0, 0, 1),
                        "detection + text": ScoreWeights(best.detection or 1, best.text or 1, 0)}
            for split in rc["eval_splits"]:
                exprs = load_expressions(ev.root, split, args.limit)
                qs = [parse_query(e["sentence"]) for e in exprs]
                cands = [ev.candidates(det, e, q) for e, q in zip(exprs, qs)]
                cat_ok = [bool(q.categories) and names[e["category_id"]] in q.categories for e, q in zip(exprs, qs)]
                with_attr = [bool(q.object_colours or q.clothing) for q in qs]
                sp = {"expressions": len(exprs),
                      "accuracy": {k: accuracy(cands, w) for k, w in variants.items()},
                      "candidate_recall": float(np.mean([c["hits"].any() for c in cands])),
                      "parser_category_accuracy": float(np.mean(cat_ok)),
                      "parser_found_category": float(np.mean([bool(q.categories) for q in qs])),
                      "with_colour_attributes": float(np.mean(with_attr)),
                      "accuracy_with_colour_attributes": {k: float(np.mean([pick(c, w) for c, a in zip(cands, with_attr) if a]))
                                                          for k, w in variants.items()},
                      "ms_per_expression": stats([c["ms"] for c in cands])}
                out["splits"][split] = sp
                log.info("%s %s: ours %.3f | det only %.3f | text only %.3f | candidate recall %.3f | parser category %.3f",
                         det, split, sp["accuracy"]["ours (tuned)"], sp["accuracy"]["detection only"],
                         sp["accuracy"]["text only"], sp["candidate_recall"], sp["parser_category_accuracy"])
            res["detectors"][det] = out
        res["sentence_baselines"] = {}
        for det in rc["sentence_baselines"]:
            res["sentence_baselines"][det] = {}
            for split in rc["eval_splits"]:
                exprs = load_expressions(ev.root, split, args.limit)
                hits, ms = zip(*(ev.sentence_baseline(det, e) for e in exprs))
                res["sentence_baselines"][det][split] = {"accuracy": float(np.mean(hits)), "ms_per_expression": stats(list(ms))}
                log.info("sentence baseline %s %s: %.3f", det, split, float(np.mean(hits)))
        res["models"] = {n: d.describe() for n, d in ev.detectors.items()}
    except Exception:
        log.exception("RefCOCO+ evaluation failed")
        return 1
    if args.limit is None and not args.override:
        out_dir = REPO_ROOT / cfg["benchmark_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "m6_refcoco.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        log.info("Wrote %s", (out_dir / "m6_refcoco.json").relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
