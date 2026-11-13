#!/usr/bin/env python
"""Fetch parts of AI City Challenge datasets (CityFlowV2, CityFlow-NL) without downloading whole archives.

The datasets are single zip files on Google Drive (16-17 GB each). Google Drive serves byte ranges, so
the zip's table of contents can be read remotely and only the needed members fetched (each verified by
the zip's CRC-32). Members already on disk with the right size are skipped.

Licence: downloading means accepting NVIDIA's Dataset License Agreement for the AI City Challenge data
(non-commercial, academic use only, also for models trained on it; no redistribution except minimal,
redacted portions in research papers; see docs/references.md). ``--accept-license`` is required and is
recorded in ``<root>/LICENSE_ACCEPTED.txt``. Nothing from these datasets may be committed.

Usage:
    python scripts/download_aicity.py cityflowv2 --list
    python scripts/download_aicity.py cityflowv2 --accept-license --include "validation/*/*/gt/*" "validation/*/*/vdo.avi"
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

from lgep.utils.logging import setup_logging

log = logging.getLogger("download_aicity")
REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
DATASETS = {
    "cityflowv2": {"id": "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC", "file": "AICity22_Track1_MTMC_Tracking.zip",
                   "size": 16_912_529_116, "page": "http://www.aicitychallenge.org/2022-track1-download/",
                   "license": "http://www.aicitychallenge.org/wp-content/uploads/2022/02/Dataset-License-AIC2022.pdf (v. February 9, 2022)"},
    "cityflow_nl": {"id": "1_uY17_mBGCIlImJvPcbw4MW-b8tB4ZwR", "file": "AICity23_Track2_NL_Retrieval.zip",
                    "size": 16_241_024_725, "page": "http://www.aicitychallenge.org/2023-track2-download/",
                    "license": "http://www.aicitychallenge.org/wp-content/uploads/2023/01/Dataset-License-AIC2023-12Jan2023.pdf (v. January 12, 2023)"},
}


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP byte ranges, with a read-ahead buffer and retries."""

    def __init__(self, url: str, size: int, chunk: int = 8 << 20, retries: int = 5):
        self.url, self.size, self.chunk, self.retries = url, size, chunk, retries
        self.pos, self._buf, self._buf_start = 0, b"", 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise OSError(f"expected HTTP 206 (partial content), got {r.status}")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
                self.bytes_fetched += len(data)
                return data
            except OSError as e:
                if attempt == self.retries - 1:
                    raise
                wait = 2 ** attempt
                log.warning("range %d-%d failed (%s); retrying in %d s", start, end, e, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        out = []
        while n > 0 and self.pos < self.size:
            off = self.pos - self._buf_start
            if not 0 <= off < len(self._buf):
                self._buf_start = self.pos
                self._buf = self._fetch(self.pos, min(self.size, self.pos + max(self.chunk, n)) - 1)
                off = 0
            piece = self._buf[off:off + n]
            out.append(piece)
            self.pos += len(piece)
            n -= len(piece)
        return b"".join(out)


def open_remote_zip(name: str) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    spec = DATASETS[name]
    f = HttpRangeFile(DRIVE.format(spec["id"]), spec["size"])
    return zipfile.ZipFile(f), f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--root", default="data/aicity")
    ap.add_argument("--list", action="store_true", help="print the archive's contents and exit")
    ap.add_argument("--include", nargs="*", default=[], help="glob patterns of members to fetch")
    ap.add_argument("--accept-license", action="store_true", help="required to fetch: you accept NVIDIA's licence")
    ap.add_argument("--workers", type=int, default=6, help="parallel connections (Google Drive limits each one)")
    args = ap.parse_args()
    setup_logging()
    spec = DATASETS[args.dataset]
    root = REPO_ROOT / args.root / args.dataset
    try:
        z, f = open_remote_zip(args.dataset)
        infos = [i for i in z.infolist() if not i.is_dir()]
        if args.list:
            for i in infos:
                print(f"{i.file_size:14,d}  {i.filename}")
            print(f"{len(infos)} files, {sum(i.file_size for i in infos) / 1e9:.2f} GB uncompressed")
            return 0
        if not args.accept_license:
            log.error("Fetching requires --accept-license (NVIDIA Dataset License Agreement: %s)", spec["license"])
            return 2
        root.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.root / "LICENSE_ACCEPTED.txt").open("a").write(
            f"{date.today().isoformat()}  {args.dataset}  accepted by the project owner: {spec['license']}\n")
        chosen = [i for i in infos if any(fnmatch.fnmatch(i.filename, p) for p in args.include)]
        total = sum(i.file_size for i in chosen)
        free = shutil.disk_usage(root).free
        if total > free - 5e9:
            raise RuntimeError(f"{total / 1e9:.1f} GB selected but only {free / 1e9:.1f} GB free (keeping 5 GB spare)")
        log.info("%s: %d of %d files selected (%.2f GB)", args.dataset, len(chosen), len(infos), total / 1e9)
        todo = [i for i in chosen if not ((root / i.filename).is_file() and (root / i.filename).stat().st_size == i.file_size)]
        todo.sort(key=lambda i: -i.file_size)  # big files first: better balance across workers
        log.info("%d files already present, %d to fetch with %d connections", len(chosen) - len(todo), len(todo), args.workers)
        local, lock, t0 = threading.local(), threading.Lock(), time.monotonic()
        progress = {"files": 0, "bytes": 0}
        handles = []

        def fetch(info: zipfile.ZipInfo) -> None:
            if not hasattr(local, "zip"):  # one remote zip (and HTTP connection state) per worker thread
                local.zip, local.file = open_remote_zip(args.dataset)
                with lock:
                    handles.append(local.file)
            dst = root / info.filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            with local.zip.open(info) as src, tmp.open("wb") as out:  # zipfile checks the CRC-32 at the end
                shutil.copyfileobj(src, out, 8 << 20)
            tmp.replace(dst)
            with lock:
                progress["files"] += 1
                progress["bytes"] += info.file_size
                if progress["files"] % 10 == 0 or progress["files"] == len(todo):
                    rate = sum(h.bytes_fetched for h in handles) / 1e6 / max(time.monotonic() - t0, 1e-6)
                    log.info("%4d/%d files, %.2f GB, %.1f MB/s", progress["files"], len(todo),
                             progress["bytes"] / 1e9, rate)

        with ThreadPoolExecutor(args.workers) as pool:
            for fut in [pool.submit(fetch, i) for i in todo]:
                fut.result()
    except Exception:
        log.exception("AI City download failed")
        return 1
    log.info("done: %s", root.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
