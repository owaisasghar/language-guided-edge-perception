#!/usr/bin/env python
"""Download COCO 2017 val (5,000 images + instances annotations) for detection evaluation.

Peak disk use is about 1.6 GB while the image zip is extracted; the zip is deleted afterwards
(about 0.8 GB remains). cocodataset.org does not publish checksums: the MD5 values below are the
ones widely used by dataset loaders (e.g. torchvision), and the SHA-256 of each archive is logged.

Usage:
    python scripts/download_coco_val2017.py            # into data/coco/
    python scripts/download_coco_val2017.py --keep-zip
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

from lgep.utils.download import download_file, sha256
from lgep.utils.logging import setup_logging

log = logging.getLogger("download_coco_val2017")
REPO_ROOT = Path(__file__).resolve().parents[1]

IMAGES = {"url": "http://images.cocodataset.org/zips/val2017.zip", "md5": "442b8da7639aecaf257c1dceb8ba8c80"}
ANNOTATIONS = {"url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
               "md5": "f4bbac642086de4f52a3fdda2de5fa2c", "member": "annotations/instances_val2017.json"}
# Known sizes of COCO 2017 val, used as a sanity check.
EXPECTED = {"images": 5000, "annotations": 36781, "categories": 80}


def fetch_annotations(root: Path, keep_zip: bool) -> Path:
    ann = root / ANNOTATIONS["member"]
    if ann.is_file():
        log.info("exists   %s", ann.relative_to(REPO_ROOT))
        return ann
    z = download_file(ANNOTATIONS["url"], root / "annotations_trainval2017.zip", md5_hex=ANNOTATIONS["md5"])
    log.info("sha256 %s  %s", sha256(z), z.name)
    with zipfile.ZipFile(z) as zf:
        zf.extract(ANNOTATIONS["member"], root)
    if not keep_zip:
        z.unlink()
    return ann


def fetch_images(root: Path, expected_files: set[str], keep_zip: bool) -> Path:
    img_dir = root / "val2017"
    if img_dir.is_dir() and expected_files <= {p.name for p in img_dir.glob("*.jpg")}:
        log.info("exists   %s (%d images)", img_dir.relative_to(REPO_ROOT), len(expected_files))
        return img_dir
    z = download_file(IMAGES["url"], root / "val2017.zip", md5_hex=IMAGES["md5"])
    log.info("sha256 %s  %s", sha256(z), z.name)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root)
    if not keep_zip:
        z.unlink()
    return img_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/coco")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    setup_logging()
    root = REPO_ROOT / args.root

    try:
        ann_path = fetch_annotations(root, args.keep_zip)
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        counts = {k: len(ann[k]) for k in EXPECTED}
        if counts != EXPECTED:
            raise RuntimeError(f"Unexpected annotation counts {counts}, expected {EXPECTED}")
        files = {im["file_name"] for im in ann["images"]}
        img_dir = fetch_images(root, files, args.keep_zip)
        missing = files - {p.name for p in img_dir.glob("*.jpg")}
        if missing:
            raise RuntimeError(f"{len(missing)} images listed in the annotations are missing from {img_dir}")
    except Exception:
        log.exception("COCO val2017 download failed")
        return 1
    log.info("COCO val2017 ready: %d images, %d annotations, %d categories", *counts.values())
    return 0


if __name__ == "__main__":
    sys.exit(main())
