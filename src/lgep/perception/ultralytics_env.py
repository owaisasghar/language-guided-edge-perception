"""Isolate Ultralytics from the user's global settings before it is imported.

Ultralytics reads ``<YOLO_CONFIG_DIR>/Ultralytics/settings.json`` once, at import time, and derives its
weights directory (where the CLIP text encoder is cached) from it. Without isolation it would use the
user's global settings (``~/.config/Ultralytics``), send usage analytics, and pip-install missing
packages at runtime. This module points it at a project-local config directory instead, with
analytics and auto-install off.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

CLIP_FILE = "ViT-B-32.pt"


def configure(config_dir: Path) -> None:
    """Set the Ultralytics environment variables. Must run before ``import ultralytics``."""
    env = {
        "YOLO_CONFIG_DIR": str(config_dir),
        "YOLO_OFFLINE": "1",  # disables analytics and online checks; weights are pre-downloaded
        "YOLO_AUTOINSTALL": "False",  # never pip-install at runtime
    }
    if "ultralytics" in sys.modules:
        stale = {k: v for k, v in env.items() if os.environ.get(k) != v}
        if stale:
            log.warning("ultralytics was imported before configure(); settings not applied: %s", sorted(stale))
        return
    for k, v in env.items():
        os.environ.setdefault(k, v)


def weights_dir(config_dir: Path) -> Path:
    """Where the project-local settings put Ultralytics' weights (the CLIP cache lives below it)."""
    return config_dir / "weights"


def write_settings(config_dir: Path) -> Path:
    """Create/update the project-local settings file. Run in a fresh process (see setup_yolo_world.py)."""
    configure(config_dir)
    from ultralytics.utils import SETTINGS

    SETTINGS.update({
        "weights_dir": str(weights_dir(config_dir)),
        "runs_dir": str(config_dir / "runs"),
        "datasets_dir": str(config_dir / "datasets"),
        "sync": False,
    })
    return Path(SETTINGS.file)
