#!/usr/bin/env python
"""Speed and GPU memory of the appearance embedder on real crops (MOT17-09 ground-truth person boxes).

Measures, per batch size, the full ``embed_images`` call (CPU -> GPU transfer, resizing, CLIP forward,
normalisation, copy back), plus text embedding (used by language queries in Milestone 6).
Writes benchmarks/milestone5/m5_embedder.{json,md}.

Usage:
    python scripts/benchmark_embedder.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from run_detector import stats

from lgep.reid.embedder import build_embedder
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.gpu import gpu_compute_processes, memory_snapshot, reset_peak, synchronize

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQ = REPO_ROOT / "data/mot17/train/MOT17-09"


def person_crops(n: int = 256) -> list[np.ndarray]:
    gt = np.loadtxt(SEQ / "gt/gt.txt", delimiter=",", ndmin=2)
    gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1) & (gt[:, 8] > 0.5)]
    crops = []
    for f in np.unique(gt[:, 0])[::25]:
        im = np.asarray(Image.open(SEQ / f"img1/{int(f):06d}.jpg").convert("RGB"))
        for _, _, x, y, w, h, *_ in gt[gt[:, 0] == f]:
            x0, y0 = max(0, int(x)), max(0, int(y))
            c = im[y0:int(y + h), x0:int(x + w)]
            if c.shape[0] >= 32 and c.shape[1] >= 16:
                crops.append(c)
        if len(crops) >= n:
            break
    return crops[:n]


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs/milestone5_reid.yaml")
    emb = build_embedder(cfg["reid"]["embedder"])
    crops = person_crops()
    heights = [c.shape[0] for c in crops]
    res = {"measured": True, "environment": collect_environment(), "other_gpu_processes": gpu_compute_processes(),
           "embedder": emb.describe(), "crops": {"n": len(crops), "height_px_median": float(np.median(heights)),
                                                  "height_px_min": int(min(heights)), "height_px_max": int(max(heights))},
           "images": [], "texts": []}
    for _ in range(3):
        emb.embed_images(crops[:64])
    for bs in (1, 8, 32, 64):
        emb.cfg.batch_size = bs
        reset_peak(emb.device)
        ms = []
        for i in range(30):
            batch = [crops[(i * bs + j) % len(crops)] for j in range(bs)]
            synchronize(emb.device)
            t0 = time.perf_counter()
            emb.embed_images(batch)
            synchronize(emb.device)
            ms.append((time.perf_counter() - t0) * 1e3)
        mem = memory_snapshot(emb.device)
        s = stats(ms)
        res["images"].append({"batch": bs, "ms_per_call": s, "ms_per_crop_median": round(s["median"] / bs, 3),
                              "crops_per_second": round(1000 * bs / s["median"], 1),
                              "peak_allocated_mib": None if mem is None else round(mem.peak_allocated_mib, 1)})
        print(f"batch {bs:3d}: {s['median']:.2f} ms per call, {s['median'] / bs:.3f} ms per crop, "
              f"{1000 * bs / s['median']:.0f} crops/s, peak {mem.peak_allocated_mib if mem else 0:.0f} MiB")
    for n in (1, 16):
        texts = [f"a person wearing a {c} shirt" for c in ("blue", "red", "green", "white")] * 4
        ms = []
        for _ in range(30):
            synchronize(emb.device)
            t0 = time.perf_counter()
            emb.embed_texts(texts[:n])
            synchronize(emb.device)
            ms.append((time.perf_counter() - t0) * 1e3)
        res["texts"].append({"texts": n, "ms_per_call": stats(ms)})
        print(f"{n:2d} text(s): {stats(ms)['median']:.2f} ms per call")
    out = REPO_ROOT / "benchmarks/milestone5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "m5_embedder.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    e = res["embedder"]
    lines = ["# Embedder speed (Milestone 5)", "",
             f"**Actual measurement**, `scripts/benchmark_embedder.py`, {res['environment']['timestamp_utc'][:10]}: "
             f"`{e['model_id']}` (vision {e['vision_parameters_millions']} M parameters, {e['dtype']}, "
             f"{e['input_size']} px, crop mode `{e['crop_mode']}`) on {res['environment']['gpu']['name']}. "
             f"{len(crops)} real MOT17-09 person crops (median height {res['crops']['height_px_median']:.0f} px). "
             "Time = the whole `embed_images` call.", "",
             "| batch | ms per call (median) | ms per crop | crops per second | peak allocated MiB |", "|---|---|---|---|---|"]
    for r in res["images"]:
        lines.append(f"| {r['batch']} | {r['ms_per_call']['median']:.2f} | {r['ms_per_crop_median']:.3f} | "
                     f"{r['crops_per_second']:.0f} | {r['peak_allocated_mib']} |")
    lines += ["", "| texts | ms per call (median) |", "|---|---|"]
    lines += [f"| {r['texts']} | {r['ms_per_call']['median']:.2f} |" for r in res["texts"]]
    (out / "m5_embedder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
