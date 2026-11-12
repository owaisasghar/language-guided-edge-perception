"""Project-wide logging setup (console + optional file)."""

from __future__ import annotations

import logging
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the root logger once; safe to call again to add a log file."""
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(_FMT, datefmt="%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    # third-party libraries are noisy at INFO
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "huggingface_hub", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
