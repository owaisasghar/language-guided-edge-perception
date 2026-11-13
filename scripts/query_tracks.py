#!/usr/bin/env python
"""Ask a recorded multi-camera run a question in plain language.

Needs a run directory written by ``run_cameras.py --reid`` (its ``index/`` folder). Prints how the query was
understood (structured query), the ranked tracks (camera, local and global ID, time, cameras where that
identity was seen) and writes a thumbnail sheet to ``RUN_DIR/queries/<query>.jpg``.

Time filters ("last 5 minutes", "after 14:00") are relative to the end of the recording by default.

Usage:
    python scripts/query_tracks.py outputs/milestone4/m4_cameras_<timestamp> "person in a white shirt"
    python scripts/query_tracks.py RUN_DIR "red car on cam02" --k 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lgep.query import parse_query
from lgep.query.attributes import QueryScorer, ScoreWeights
from lgep.query.index import TrackIndex
from lgep.reid.embedder import build_embedder
from lgep.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def contact_sheet(hits, path: Path, height: int = 128) -> None:
    tiles = []
    for h in hits:
        t = h.entry.thumbnail if h.entry.thumbnail is not None else np.zeros((height, height // 2, 3), np.uint8)
        im = Image.fromarray(t)
        im = im.resize((max(1, round(im.width * height / im.height)), height))
        canvas = Image.new("RGB", (max(im.width, 120), height + 34), (250, 250, 250))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((2, height + 2), f"#{h.rank} {h.entry.key}", fill=(0, 0, 0))
        d.text((2, height + 17), f"ID {h.entry.global_id}  {h.score:.2f}", fill=(80, 80, 80))
        tiles.append(canvas)
    if not tiles:
        return
    sheet = Image.new("RGB", (sum(t.width + 6 for t in tiles), tiles[0].height), (255, 255, 255))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 6
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--config", default="configs/milestone6_query.yaml")
    ap.add_argument("--now", choices=("recording-end", "clock"), default="recording-end")
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = load_config(REPO_ROOT / args.config)
    index = TrackIndex.load(run / "index")
    embedder = build_embedder(cfg["reid"]["embedder"])
    scorer = QueryScorer(embedder, ScoreWeights.from_dict({**cfg["query"]["weights"], "detection": 0.0}))
    t0 = time.perf_counter()
    q = parse_query(args.query)
    t1 = time.perf_counter()
    now = max((e.wall_end for e in index.entries), default=time.time()) if args.now == "recording-end" else time.time()
    hits = index.search(q, scorer, k=args.k, now=now)
    t2 = time.perf_counter()
    print(f"query: {args.query!r}")
    print(f"understood as: categories={q.categories or 'any'} colours={q.object_colours} "
          f"clothing={[(c.item, c.colour) for c in q.clothing]} cameras={q.cameras or 'any'} time={q.time} "
          f"related={q.carried}")
    print(f"{len(index)} tracks indexed; parse {1e3 * (t1 - t0):.1f} ms, search {1e3 * (t2 - t1):.1f} ms "
          f"(incl. text encoding on first use)\n")
    for h in hits:
        r = h.row()
        print(f"#{r['rank']:2d} {r['score']:.3f}  {r['track']:12s} ID {str(r['global_id']):5s} {r['start']}  "
              f"{r['duration_s']:5.1f} s  text {r['text']:.3f} attr {r['attributes']:.2f}  seen in {', '.join(r['seen_in_cameras'])}")
    slug = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_")[:60]
    contact_sheet(hits, run / "queries" / f"{slug}.jpg")
    (run / "queries" / f"{slug}.json").write_text(json.dumps({"query": args.query, "hits": [h.row() for h in hits]},
                                                             indent=2), encoding="utf-8")
    print(f"\nthumbnails: {run / 'queries' / (slug + '.jpg')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
