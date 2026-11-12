#!/usr/bin/env python
"""Download the sample images listed in a manifest and verify their SHA-256.

Usage:
    python scripts/download_samples.py --manifest configs/samples/milestone1_coco.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

from lgep.utils.config import load_config
from lgep.utils.logging import setup_logging

log = logging.getLogger("download_samples")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/samples/milestone1_coco.yaml")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()
    setup_logging()

    man = load_config(REPO_ROOT / args.manifest)
    target = REPO_ROOT / man["target_dir"]
    target.mkdir(parents=True, exist_ok=True)
    failures = 0
    for item in man["images"]:
        dst = target / item["file"]
        if dst.exists() and not args.force:
            log.info("exists   %s", dst.relative_to(REPO_ROOT))
        else:
            log.info("download %s", item["url"])
            tmp = dst.with_suffix(dst.suffix + ".part")
            try:
                urllib.request.urlretrieve(item["url"], tmp)
                tmp.replace(dst)
            except OSError as e:
                log.error("failed   %s: %s", item["url"], e)
                tmp.unlink(missing_ok=True)
                failures += 1
                continue
        expected = item.get("sha256")
        if expected and sha256(dst) != expected:
            log.error("checksum mismatch for %s (delete it and re-run)", dst.name)
            failures += 1
    log.info("%d/%d images ready in %s", len(man["images"]) - failures, len(man["images"]),
             target.relative_to(REPO_ROOT))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
