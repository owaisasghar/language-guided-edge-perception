#!/usr/bin/env python
"""Query latency vs index size.

Starts from a real track index (written by ``run_cameras.py --reid``) and enlarges it to 1k / 10k / 100k
tracks by copying entries with small random changes to their embeddings (unit-normalised), cameras and
times. Measures, per query: parsing, and a warm search (text embeddings cached after the first query: the
steady state of a running system). Search runs on the CPU (NumPy); only text encoding uses CLIP.
Writes benchmarks/milestone6/m6_query_latency.{json,md}.

Usage:
    python scripts/benchmark_query.py outputs/milestone4/m4_cameras_<timestamp>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from run_detector import stats

from lgep.query import parse_query
from lgep.query.attributes import QueryScorer, ScoreWeights
from lgep.query.index import TrackIndex
from lgep.reid.embedder import build_embedder
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES = ["person in a white shirt", "man in black jeans on cam01", "woman in red",
           "person with a backpack in the last 30 seconds", "person"]


def enlarge(index: TrackIndex, n: int, rng: np.random.Generator) -> TrackIndex:
    base = index.entries
    out = []
    for i in range(n):
        e = copy.copy(base[i % len(base)])
        e.key = f"{e.camera_id}/{100000 + i}"
        e.camera_id = f"cam{rng.integers(1, 9):02d}"
        shift = float(rng.uniform(-3600, 0))
        e.wall_start, e.wall_end = e.wall_start + shift, e.wall_end + shift
        e.embeddings = {r: (v + rng.normal(0, 0.01, v.shape)).astype(np.float32) for r, v in e.embeddings.items()}
        for r, v in e.embeddings.items():
            e.embeddings[r] = v / np.linalg.norm(v)
        e.thumbnail = None
        out.append(e)
    return TrackIndex(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--sizes", nargs="*", type=int, default=[1000, 10000, 100000])
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / "configs/milestone6_query.yaml")
    base = TrackIndex.load(Path(args.run_dir) / "index")
    scorer = QueryScorer(build_embedder(cfg["reid"]["embedder"]),
                         ScoreWeights.from_dict({**cfg["query"]["weights"], "detection": 0.0}))
    rng = np.random.default_rng(0)
    res = {"measured": True, "environment": collect_environment(), "base_index_tracks": len(base),
           "queries": QUERIES, "sizes": []}
    for q in QUERIES:  # warm the text-embedding cache
        base.search(parse_query(q), scorer, now=time.time())
    for n in args.sizes:
        idx = enlarge(base, n, rng)
        now = max(e.wall_end for e in idx.entries)
        parse_ms, search_ms = [], []
        for _ in range(args.repeats):
            for text in QUERIES:
                t0 = time.perf_counter()
                q = parse_query(text)
                t1 = time.perf_counter()
                idx.search(q, scorer, k=10, now=now)
                t2 = time.perf_counter()
                parse_ms.append((t1 - t0) * 1e3)
                search_ms.append((t2 - t1) * 1e3)
        bytes_per_track = float(np.mean([sum(v.nbytes for v in e.embeddings.values()) for e in idx.entries[:1000]]))
        row = {"tracks": n, "parse_ms": stats(parse_ms), "search_ms": stats(search_ms),
               "embedding_bytes_per_track": bytes_per_track}
        res["sizes"].append(row)
        print(f"{n:7d} tracks: parse {row['parse_ms']['median']:.2f} ms, search {row['search_ms']['median']:.1f} ms "
              f"(p90 {row['search_ms']['p90']:.1f}), {bytes_per_track / 1024:.1f} KiB embeddings per track")
    out = REPO_ROOT / "benchmarks/milestone6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "m6_query_latency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = ["# Query latency vs index size (Milestone 6)", "",
             f"**Actual measurement**, `scripts/benchmark_query.py`: a real index of {len(base)} tracks (4 simulated "
             "cameras, 2 minutes) enlarged with jittered copies; warm search (text embeddings cached), NumPy on the "
             f"CPU ({res['environment']['cpu']}); {len(QUERIES)} queries x {args.repeats} repeats.", "",
             "| tracks | parse ms (median) | search ms (median / p90) | embedding storage per track |", "|---|---|---|---|"]
    lines += [f"| {r['tracks']:,} | {r['parse_ms']['median']:.2f} | {r['search_ms']['median']:.1f} / {r['search_ms']['p90']:.1f} "
              f"| {r['embedding_bytes_per_track'] / 1024:.1f} KiB |" for r in res["sizes"]]
    (out / "m6_query_latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
