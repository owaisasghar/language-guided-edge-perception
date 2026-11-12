#!/usr/bin/env python
"""Download the MOT17 training sequences (images + tracking ground truth) for tracking evaluation.

The full MOT17.zip (5.9 GB) stores every image sequence three times (one copy per public detector:
DPM, FRCNN, SDP). This script instead combines
  * MOT17Det.zip    (1.95 GB): each sequence's images once, and
  * MOT17Labels.zip (10 MB):   tracking ground truth (gt.txt) and seqinfo.ini,
and extracts only the 7 training sequences (5,316 frames); the test split has no public ground truth.
Peak disk use is about 3.1 GB during extraction; about 1.1 GB remains after the zips are deleted.

motchallenge.net does not publish checksums. The archives are checked against their known sizes, and
every sequence is checked against its seqinfo.ini (frame count, image size) and the known MOT17 lengths.
The SHA-256 values below were recorded from our first download on 2026-11-12 (trust on first use).

MOT17 is licensed CC BY-NC-SA 3.0 (non-commercial). Do not commit the data.

Usage:
    python scripts/download_mot17.py              # into data/mot17/train/MOT17-XX/{img1,gt,seqinfo.ini}
    python scripts/download_mot17.py --keep-zip
"""

from __future__ import annotations

import argparse
import configparser
import logging
import re
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

from lgep.utils.download import download_file, sha256
from lgep.utils.logging import setup_logging

log = logging.getLogger("download_mot17")
REPO_ROOT = Path(__file__).resolve().parents[1]

ARCHIVES = {
    "labels": {"url": "https://motchallenge.net/data/MOT17Labels.zip", "size": 10_107_022,
               "sha256": "0aa79322e91583369f42f17c4d79a0b145380d8732487bba59272048dc82b2b9"},
    "images": {"url": "https://motchallenge.net/data/MOT17Det.zip", "size": 1_952_547_113,
               "sha256": "244f013311745e10d9869111a484cf98d285ebb99a5c63fdd14b5b5e42807fa5"},
}
# MOT17 training sequences and their lengths (frames), from the MOT17 paper / seqinfo.ini.
SEQUENCES = {"MOT17-02": 600, "MOT17-04": 1050, "MOT17-05": 837, "MOT17-09": 525, "MOT17-10": 654,
             "MOT17-11": 900, "MOT17-13": 750}
# The tracking ground truth is identical in the -DPM/-FRCNN/-SDP copies; ByteTrack uses -FRCNN.
LABEL_RE = re.compile(r"(?:^|/)train/(MOT17-\d\d)-FRCNN/(gt/gt\.txt|seqinfo\.ini)$")
IMAGE_RE = re.compile(r"(?:^|/)train/(MOT17-\d\d)/img1/(\d{6}\.jpg)$")


def fetch(name: str, root: Path) -> Path:
    spec = ARCHIVES[name]
    dst = root / Path(spec["url"]).name
    path = download_file(spec["url"], dst, sha256_hex=spec["sha256"])
    if path.stat().st_size != spec["size"]:
        raise RuntimeError(f"{path.name}: {path.stat().st_size} bytes, expected {spec['size']}")
    log.info("sha256 %s  %s", sha256(path), path.name)
    return path


def extract(zip_path: Path, pattern: re.Pattern, out_root: Path, subdir: str = "") -> int:
    """Extract members matching ``pattern`` (groups: sequence, relative path) to out_root/<seq>/<subdir>."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            m = pattern.search(info.filename)
            if not m or m.group(1) not in SEQUENCES:
                continue
            dst = out_root / m.group(1) / subdir / m.group(2)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            n += 1
    return n


def seqinfo(seq_dir: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(seq_dir / "seqinfo.ini")
    s = cp["Sequence"]
    return {"length": int(s["seqLength"]), "width": int(s["imWidth"]), "height": int(s["imHeight"]),
            "fps": float(s["frameRate"]), "ext": s["imExt"]}


def verify(train: Path) -> int:
    total = 0
    for seq, length in SEQUENCES.items():
        d = train / seq
        info = seqinfo(d)
        images = sorted((d / "img1").glob("*.jpg"))
        if info["length"] != length or len(images) != length:
            raise RuntimeError(f"{seq}: seqinfo {info['length']}, {len(images)} images, expected {length}")
        if [p.name for p in images] != [f"{i:06d}.jpg" for i in range(1, length + 1)]:
            raise RuntimeError(f"{seq}: image files are not 000001.jpg ... {length:06d}.jpg")
        with Image.open(images[0]) as im:
            if im.size != (info["width"], info["height"]):
                raise RuntimeError(f"{seq}: image size {im.size} != seqinfo {(info['width'], info['height'])}")
        gt_frames = [int(line.split(",", 1)[0]) for line in (d / "gt/gt.txt").read_text().splitlines() if line]
        if not gt_frames or min(gt_frames) < 1 or max(gt_frames) > length:
            raise RuntimeError(f"{seq}: gt.txt frames outside 1..{length}")
        log.info("ok       %s  %4d frames  %dx%d @ %g fps  %6d gt rows", seq, length, info["width"],
                 info["height"], info["fps"], len(gt_frames))
        total += length
    return total


def complete(train: Path) -> bool:
    try:
        verify(train)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/mot17")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    setup_logging()
    root = REPO_ROOT / args.root
    train = root / "train"

    try:
        if complete(train):
            log.info("exists   %s", train.relative_to(REPO_ROOT))
        else:
            for name, pattern, subdir in (("labels", LABEL_RE, ""), ("images", IMAGE_RE, "img1")):
                z = fetch(name, root)
                n = extract(z, pattern, train, subdir)
                log.info("extracted %d files from %s", n, z.name)
                if not args.keep_zip:
                    z.unlink()
        total = verify(train)
    except Exception:
        log.exception("MOT17 download failed")
        return 1
    log.info("MOT17 train ready: %d sequences, %d frames in %s", len(SEQUENCES), total, train.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
