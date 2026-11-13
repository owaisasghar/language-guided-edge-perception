#!/usr/bin/env python
"""CLIP (Re-ID / search embedder) in FP16 vs FP32 (Milestone 8, RQ6): agreement, identity separation, speed.

Real MOT17 person crops (ground-truth boxes of MOT17-02, -04, -09; visibility >= 0.5, height >= 64 px, every
10th frame), embedded with the Milestone 5 embedder in both precisions:
  * agreement: cosine between the FP32 and FP16 embedding of each crop;
  * identity separation: same-vs-different-person AUROC of crop pairs within a sequence, per precision;
  * speed: ms per call and per crop at batch 1, 8, 32, 64; peak allocated GPU memory.

Usage:
    python scripts/check_clip_fp16.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lgep.reid.embedder import build_embedder
from lgep.reid.metrics import same_vs_different_auroc
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("check_clip_fp16")
REPO_ROOT = Path(__file__).resolve().parents[1]
SEQS = ("MOT17-02", "MOT17-04", "MOT17-09")


def crops(per_seq: int = 200) -> tuple[list[np.ndarray], list[int], list[str]]:
    out, ids, groups = [], [], []
    for seq in SEQS:
        d = REPO_ROOT / "data/mot17/train" / seq
        gt = np.loadtxt(d / "gt/gt.txt", delimiter=",")
        gt = gt[(gt[:, 7] == 1) & (gt[:, 8] >= 0.5) & (gt[:, 5] >= 64) & (gt[:, 0] % 10 == 1)]
        rng = np.random.default_rng(0)
        gt = gt[rng.permutation(len(gt))[:per_seq]]
        gt = gt[np.argsort(gt[:, 0])]
        img, cur = None, None
        for f, pid, x, y, w, h, *_ in gt:
            if f != cur:
                img, cur = np.asarray(Image.open(d / "img1" / f"{int(f):06d}.jpg").convert("RGB")), f
            x1, y1 = max(0, int(x)), max(0, int(y))
            c = img[y1:int(y + h), x1:int(x + w)]
            if c.size:
                out.append(c)
                ids.append(int(pid))
                groups.append(seq)
    return out, ids, groups


def speed(emb, cs: list[np.ndarray]) -> list[dict]:
    rows = []
    for bs in (1, 8, 32, 64):
        batch = [cs[i % len(cs)] for i in range(bs)]
        for _ in range(3):
            emb.embed_images(batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(20 if bs < 32 else 10):
            t0 = time.perf_counter()
            emb.embed_images(batch)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        med = float(np.median(ts))
        rows.append({"batch": bs, "ms_per_call": round(med, 2), "ms_per_crop": round(med / bs, 3),
                     "crops_per_s": round(1000 * bs / med), "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20)})
    return rows


def main() -> int:
    out = REPO_ROOT / "benchmarks" / "milestone8"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", REPO_ROOT / "outputs" / "milestone8" / "check_clip_fp16.log")
    base = load_config(REPO_ROOT / "configs/milestone5_reid.yaml")["reid"]["embedder"]
    cs, ids, groups = crops()
    log.info("%d crops of %d people from %s", len(cs), len(set(zip(groups, ids))), ", ".join(SEQS))
    res = {"measured": True, "crops": len(cs), "people": len(set(zip(groups, ids))), "sequences": SEQS,
           "environment": collect_environment(), "precisions": {}}
    feats = {}
    for dtype in ("float32", "float16"):
        emb = build_embedder({**base, "dtype": dtype})
        f = np.concatenate([emb.embed_images(cs[i:i + 64]) for i in range(0, len(cs), 64)])
        feats[dtype] = f
        gid = [hash((g, i)) for g, i in zip(groups, ids)]
        res["precisions"][dtype] = {"auroc_same_vs_different": same_vs_different_auroc(f, gid, groups),
                                    "speed": speed(emb, cs), "embedder": emb.describe()}
        log.info("%s: AUROC %s | %s", dtype, res["precisions"][dtype]["auroc_same_vs_different"],
                 res["precisions"][dtype]["speed"])
        del emb
        torch.cuda.empty_cache()
    cos = np.sum(feats["float32"] * feats["float16"], axis=1)
    res["agreement_cosine"] = {"mean": round(float(cos.mean()), 6), "min": round(float(cos.min()), 6),
                               "p01": round(float(np.percentile(cos, 1)), 6)}
    log.info("FP32 vs FP16 cosine: %s", res["agreement_cosine"])
    (out / "m8_clip_fp16.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
