#!/usr/bin/env python
"""Download the files listed in a manifest and verify their SHA-256.

Manifests list entries under ``images:`` (sample images) or ``files:`` (e.g. model weights).

Usage:
    python scripts/download_samples.py --manifest configs/samples/milestone1_coco.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lgep.utils.config import load_config
from lgep.utils.download import download_file
from lgep.utils.logging import setup_logging

log = logging.getLogger("download_samples")
REPO_ROOT = Path(__file__).resolve().parents[1]


def download_manifest(manifest: str | Path, force: bool = False) -> int:
    """Download every entry of a manifest; returns the number of failures."""
    man = load_config(REPO_ROOT / manifest)
    target = REPO_ROOT / man["target_dir"]
    items = man.get("images") or man.get("files") or []
    failures = 0
    for item in items:
        try:
            download_file(item["url"], target / item["file"], sha256_hex=item.get("sha256"), force=force)
        except Exception as e:  # network errors and checksum mismatches are reported, not fatal
            log.error("failed   %s: %s", item["file"], e)
            failures += 1
    log.info("%d/%d files ready in %s", len(items) - failures, len(items), target.relative_to(REPO_ROOT))
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/samples/milestone1_coco.yaml")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()
    setup_logging()
    return 1 if download_manifest(args.manifest, args.force) else 0


if __name__ == "__main__":
    sys.exit(main())
