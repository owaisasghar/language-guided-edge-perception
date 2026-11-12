#!/usr/bin/env python
"""Check our TrackEval integration (lgep.evaluation.mot) on a real tracker's results.

TrackEval's example data (data.zip, ~150 MB, linked from TrackEval's README) contains the MOT17-train ground
truth and the results of the MPNTrack tracker. It does **not** contain reference scores (TrackEval's own
tests read them from a separate ``data/tests`` package that is not linked from its README), so this script
checks what it can:

1. ``evaluate_mot`` (official TrackEval at the pinned commit + NumPy-alias shim) runs on all 21 MOT17-train
   sequence variants of a real tracker without error;
2. the numbers our wrapper reports (field mapping, HOTA averaged over its 19 thresholds, percentages) equal
   the summary file TrackEval itself writes in the same run (``pedestrian_summary.txt``, 5 significant digits).

This validates our wrapper code; TrackEval's metric definitions themselves are covered by the hand-computed
cases in tests/test_mot_eval.py.

Usage:
    python scripts/validate_trackeval.py          # downloads data.zip into data/trackeval/ if missing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

from lgep.evaluation.mot import TRACKEVAL_COMMIT, evaluate_mot
from lgep.utils.download import download_file, sha256
from lgep.utils.logging import setup_logging

log = logging.getLogger("validate_trackeval")
REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip"  # from TrackEval's README
SHA256 = "148a4bdf56f8cb32f9bde4b3d70717242cbb2e61900ef0101abc4652e268f2a5"  # our first download, 2026-11-12
GT = "data/gt/mot_challenge/MOT17-train/"
TRACKER = "data/trackers/mot_challenge/MOT17-train/MPNTrack/"
FIELDS = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDR", "IDP", "IDSW", "CLR_FP", "CLR_FN", "MT", "ML")


def extract(zip_path: Path, root: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith((GT, TRACKER)) and not m.endswith("/")]
        if not members:
            raise RuntimeError(f"{zip_path} has no {GT} / {TRACKER} entries")
        zf.extractall(root, members)


def read_summary(path: Path) -> dict[str, float]:
    header, values = path.read_text().splitlines()[:2]
    return dict(zip(header.split(), (float(v) for v in values.split())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/trackeval")
    args = ap.parse_args()
    setup_logging()
    root = REPO_ROOT / args.root
    try:
        z = download_file(URL, root / "data.zip", sha256_hex=SHA256)
        log.info("sha256 %s  %s", sha256(z), z.name)
        if not (root / TRACKER / "data").is_dir():
            extract(z, root)
        seqs = sorted(p.name for p in (root / GT).iterdir() if (p / "gt/gt.txt").is_file())
        with tempfile.TemporaryDirectory() as tmp:
            ours = evaluate_mot(root / GT, root / TRACKER.rsplit("/", 2)[0], "MPNTrack", {s: None for s in seqs},
                                output_dir=tmp)["combined"]
            written = read_summary(Path(tmp) / "MPNTrack" / "pedestrian_summary.txt")
    except Exception:
        log.exception("TrackEval validation failed")
        return 1
    ok = True
    rows = {}
    for k in FIELDS:
        mine = ours[k] if isinstance(ours[k], int) else 100 * ours[k]
        same = mine == written[k] if isinstance(ours[k], int) else float(f"{mine:.5g}") == written[k]
        ok &= same
        rows[k] = {"wrapper": round(mine, 4), "trackeval_summary": written[k], "equal": same}
        log.info("%-6s wrapper %-10s TrackEval summary %-10s %s", k, round(mine, 3), written[k],
                 "ok" if same else "DIFFERENT")
    out = REPO_ROOT / "benchmarks/milestone3/trackeval_validation.json"
    out.write_text(json.dumps({"measured": True, "tracker": "MPNTrack", "split": "MOT17-train", "sequences": len(seqs),
                               "trackeval_commit": TRACKEVAL_COMMIT, "all_equal": ok, "fields": rows}, indent=2))
    log.info("%s on %d sequences; wrote %s", "wrapper == TrackEval's own summary" if ok else "MISMATCH", len(seqs),
             out.relative_to(REPO_ROOT))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
