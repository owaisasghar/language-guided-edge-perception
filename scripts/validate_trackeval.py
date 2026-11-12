#!/usr/bin/env python
"""Check our TrackEval integration (lgep.evaluation.mot) against TrackEval's own reference output.

TrackEval's example data (data.zip, ~150 MB, from the TrackEval README) contains MOT17-train ground truth,
the MPNTrack tracker's results, and the summary TrackEval itself wrote for them
(``pedestrian_summary.txt``). This script scores MPNTrack with ``evaluate_mot`` (official TrackEval at the
pinned commit, NumPy-alias shim) and compares HOTA / MOTA / IDF1 / IDSW with that file.

Usage:
    python scripts/validate_trackeval.py          # downloads data.zip into data/trackeval/ if missing
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

from lgep.evaluation.mot import evaluate_mot
from lgep.utils.download import download_file, sha256
from lgep.utils.logging import setup_logging

log = logging.getLogger("validate_trackeval")
REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip"  # from TrackEval's README
GT = "data/gt/mot_challenge/MOT17-train/"
TRACKER = "data/trackers/mot_challenge/MOT17-train/MPNTrack/"
COMPARE = {"HOTA": "HOTA", "DetA": "DetA", "AssA": "AssA", "MOTA": "MOTA", "IDF1": "IDF1", "IDSW": "IDSW"}


def extract(zip_path: Path, root: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith((GT, TRACKER)) and not m.endswith("/")]
        if not members:
            raise RuntimeError(f"{zip_path} has no {GT} / {TRACKER} entries")
        zf.extractall(root, members)


def reference(summary: Path) -> dict[str, float]:
    header, values = summary.read_text().split("\n")[:2]
    return dict(zip(header.split(), (float(v) for v in values.split())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/trackeval")
    args = ap.parse_args()
    setup_logging()
    root = REPO_ROOT / args.root
    try:
        z = download_file(URL, root / "data.zip")
        log.info("sha256 %s  %s", sha256(z), z.name)
        if not (root / TRACKER / "pedestrian_summary.txt").is_file():
            extract(z, root)
        ref = reference(root / TRACKER / "pedestrian_summary.txt")
        seqs = sorted(p.name for p in (root / GT).iterdir() if (p / "gt/gt.txt").is_file())
        ours = evaluate_mot(root / GT, root / TRACKER.rsplit("/", 2)[0], "MPNTrack", {s: None for s in seqs})
        c = ours["combined"]
        ok = True
        for ours_key, ref_key in COMPARE.items():
            mine = c[ours_key] if ours_key == "IDSW" else round(100 * c[ours_key], 3)
            # The reference file stores 5 significant digits.
            same = abs(mine - ref[ref_key]) <= (0 if ours_key == "IDSW" else 5e-3 * max(1.0, abs(ref[ref_key]) / 100))
            ok &= same
            log.info("%-5s ours %-10s reference %-10s %s", ours_key, mine, ref[ref_key], "ok" if same else "DIFFERENT")
    except Exception:
        log.exception("TrackEval validation failed")
        return 1
    log.info("%s on %d sequences", "MATCHES TrackEval's reference output" if ok else "DOES NOT MATCH", len(seqs))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
