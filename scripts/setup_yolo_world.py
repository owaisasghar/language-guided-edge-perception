#!/usr/bin/env python
"""One-time setup for the YOLO-World backend (Milestone 2). Needs the `yolo` extra installed.

1. Downloads the YOLO-World v2 weights listed in configs/weights/yolo_world.yaml (SHA-256 verified).
2. Writes a project-local Ultralytics settings file (models/ultralytics/Ultralytics/settings.json):
   analytics off, weights/runs/datasets directories inside models/ultralytics/. The user's global
   ~/.config/Ultralytics settings are not read or modified.
3. Downloads the CLIP ViT-B/32 text encoder into that weights directory (the `clip` package verifies
   the SHA-256 embedded in its download URL).

After this, YOLO-World runs fully offline.

Usage:
    python scripts/setup_yolo_world.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from download_samples import download_manifest

from lgep.perception import ultralytics_env
from lgep.utils.logging import setup_logging

log = logging.getLogger("setup_yolo_world")
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/weights/yolo_world.yaml")
    ap.add_argument("--ultralytics-dir", default="models/ultralytics")
    args = ap.parse_args()
    setup_logging()

    if download_manifest(args.manifest):
        return 1

    config_dir = REPO_ROOT / args.ultralytics_dir
    settings_file = ultralytics_env.write_settings(config_dir)
    log.info("Ultralytics settings: %s", settings_file.relative_to(REPO_ROOT))

    import clip

    clip_dir = ultralytics_env.weights_dir(config_dir) / "clip"
    try:
        clip.load("ViT-B/32", device="cpu", download_root=str(clip_dir))  # raises on checksum mismatch
    except Exception:
        log.exception("CLIP ViT-B/32 download/verification failed")
        return 1
    log.info("CLIP ViT-B/32 ready: %s", (clip_dir / ultralytics_env.CLIP_FILE).relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
