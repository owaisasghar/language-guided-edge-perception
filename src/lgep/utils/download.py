"""Checksum-verified downloads for datasets and model weights."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: Path) -> str:
    return file_digest(path, "sha256")


class ChecksumError(RuntimeError):
    pass


def download_file(url: str, dst: Path, *, sha256_hex: str | None = None, md5_hex: str | None = None,
                  force: bool = False) -> Path:
    """Download ``url`` to ``dst`` atomically (via ``.part``) and verify the given checksum(s).

    An existing file is kept (and still verified) unless ``force`` is set. A file that fails
    verification is never deleted automatically, so it can be inspected.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        log.info("exists   %s", dst)
    else:
        log.info("download %s -> %s", url, dst)
        tmp = dst.with_name(dst.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dst)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
    for algo, expected in (("sha256", sha256_hex), ("md5", md5_hex)):
        if expected:
            got = file_digest(dst, algo)
            if got != expected.lower():
                raise ChecksumError(f"{algo} mismatch for {dst}: expected {expected}, got {got} "
                                    "(delete the file and re-run)")
    return dst
