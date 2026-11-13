#!/usr/bin/env python
"""Download RefCOCO+ (val, testA, testB) for the interim evaluation of language-guided retrieval.

RefCOCO+ (Yu et al., ECCV 2016) has referring expressions for objects in COCO train2014 images, collected so
that they describe *appearance* ("man in red shirt", "white car"), not absolute location. The original
download server (bvisionweb1.cs.unc.edu) no longer responds, so the annotations come from the Hugging Face
mirror ``jxu124/refcocoplus`` (pinned commit; its counts are checked against the published statistics
below). The images come from COCO's official server, one by one (about 3,000 images, ~0.6 GB), and each is
checked to decode and to have its annotated size.

Licences: COCO images keep their per-image Flickr licences (recorded in the annotations); the RefCOCO+
mirror does not state a licence. Use for research; nothing here is committed.

Usage:
    python scripts/download_refcocoplus.py                 # into data/refcocoplus/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from lgep.utils.download import download_file
from lgep.utils.logging import setup_logging

log = logging.getLogger("download_refcocoplus")
REPO_ROOT = Path(__file__).resolve().parents[1]
MIRROR = "jxu124/refcocoplus"
REVISION = "12e20b0b6039fbf656e89e2f26597e84c1037847"
FILES = {"val": "data/validation-00000-of-00001-8c57d66282bc60c9.parquet",
         "testA": "data/test-00000-of-00001-2b8e5d26906553b9.parquet",
         "testB": "data/testB-00000-of-00001-4f1178d399f1874a.parquet"}
# Published RefCOCO+ statistics (UNC split): referred objects, expressions, images.
EXPECTED = {"val": (3805, 10758, 1500), "testA": (1975, 5726, 750), "testB": (1798, 4889, 750)}
IMAGE_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"


def load_split(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def fetch_image(image_id: int, info: dict, out: Path) -> str | None:
    dst = out / f"COCO_train2014_{image_id:012d}.jpg"
    try:
        download_file(IMAGE_URL.format(image_id), dst)
        with Image.open(dst) as im:
            if im.size != (info["width"], info["height"]):
                return f"{dst.name}: size {im.size} != annotated {(info['width'], info['height'])}"
            im.verify()
    except Exception as e:  # noqa: BLE001 - report per image, keep going
        return f"{dst.name}: {e}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/refcocoplus")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    setup_logging()
    logging.getLogger("lgep.utils.download").setLevel(logging.WARNING)
    root = REPO_ROOT / args.root
    try:
        from huggingface_hub import hf_hub_download

        images: dict[int, dict] = {}
        licences = Counter()
        for split, fname in FILES.items():
            path = Path(hf_hub_download(MIRROR, fname, repo_type="dataset", revision=REVISION, local_dir=root / "annotations"))
            rows = load_split(path)
            counts = (len(rows), sum(len(r["sent_ids"]) for r in rows), len({r["image_id"] for r in rows}))
            if counts != EXPECTED[split]:
                raise RuntimeError(f"{split}: (refs, expressions, images) = {counts}, expected {EXPECTED[split]}")
            log.info("%-5s %d refs, %d expressions, %d images (matches the published statistics)", split, *counts)
            for r in rows:
                info = json.loads(r["raw_image_info"])
                images[int(r["image_id"])] = info
        for info in images.values():
            licences[info.get("license")] += 1
        out = root / "images"
        out.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(args.workers) as pool:
            errors = [e for e in pool.map(lambda kv: fetch_image(kv[0], kv[1], out), images.items()) if e]
        if errors:
            raise RuntimeError(f"{len(errors)} image(s) failed, e.g. {errors[:3]}")
    except Exception:
        log.exception("RefCOCO+ download failed")
        return 1
    log.info("RefCOCO+ ready: %d images in %s; COCO licence ids: %s", len(images), out.relative_to(REPO_ROOT),
             dict(sorted(licences.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
