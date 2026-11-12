"""Owns all camera readers: start/stop, newest frames, health status, cameras added or removed at runtime."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable

from lgep.ingestion.reader import CameraReader
from lgep.ingestion.source import CameraConfig, Frame, open_capture

log = logging.getLogger(__name__)


class StreamManager:
    """N cameras, one capture thread each. Consumers poll ``latest()`` and block on ``wait_for_frame()``."""

    def __init__(self, cameras: Iterable[CameraConfig] = (), capture_factory: Callable = open_capture):
        self._factory = capture_factory
        self._readers: dict[str, CameraReader] = {}
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._started = False
        for cfg in cameras:
            self.add_camera(cfg)

    # ------------------------------------------------------------------------------------- lifecycle
    def start(self) -> StreamManager:
        with self._lock:
            self._started = True
            readers = list(self._readers.values())
        for r in readers:
            r.start()
        log.info("Started %d camera(s): %s", len(readers), ", ".join(r.cfg.camera_id for r in readers))
        return self

    def stop(self) -> None:
        with self._lock:
            self._started = False
            readers = list(self._readers.values())
        for r in readers:  # signal all first, then join, so shutdown takes one timeout, not N
            r._stop.set()
        for r in readers:
            r.stop()
        self._new_frame.set()  # wake a waiting consumer
        log.info("Stopped %d camera(s)", len(readers))

    def __enter__(self) -> StreamManager:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -------------------------------------------------------------------------------------- cameras
    def add_camera(self, cfg: CameraConfig) -> None:
        with self._lock:
            if cfg.camera_id in self._readers:
                raise ValueError(f"camera {cfg.camera_id!r} already exists")
            r = CameraReader(cfg, self._factory, on_frame=self._notify)
            self._readers[cfg.camera_id] = r
            started = self._started
        if started:
            r.start()

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            r = self._readers.pop(camera_id)
        r.stop()

    @property
    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._readers)

    def reader(self, camera_id: str) -> CameraReader:
        return self._readers[camera_id]

    # --------------------------------------------------------------------------------------- frames
    def _notify(self, _camera_id: str) -> None:
        self._new_frame.set()

    def wait_for_frame(self, timeout: float) -> bool:
        """Block until any camera has published a frame since the last call (or ``timeout``)."""
        got = self._new_frame.wait(timeout)
        self._new_frame.clear()
        return got

    def latest(self, camera_id: str, newer_than: int = 0, captured_after: float | None = None) -> Frame | None:
        r = self._readers.get(camera_id)
        return r.latest(newer_than, captured_after) if r else None

    def status(self) -> dict[str, dict]:
        with self._lock:
            readers = list(self._readers.values())
        return {r.cfg.camera_id: r.stats() for r in readers}
