"""Live state for the dashboard: the newest processed frame of each camera, and an event stream.

``LiveHub``: the perception worker hands over each processed frame with its tracks (a reference, no copy, no
encoding). A viewer asks for "a frame newer than N" (long poll); the first viewer of a new frame encodes it
once (downscale to ``stream_width``, draw boxes and IDs, JPEG), the others get the cached bytes. Nothing is
encoded while nobody watches, and at most ``max_fps`` frames per camera are encoded.

``EventBus``: thread-safe fan-out of JSON-able messages (store commits, status) to the asyncio queues of
connected Server-Sent-Events clients; a slow client loses its oldest messages, never blocks a producer.
"""

from __future__ import annotations

import asyncio
import colorsys
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    stream_width: int = 960  # px; frames are downscaled before drawing and encoding
    jpeg_quality: int = 80
    max_fps: float = 10.0  # per camera, encodes at most this often (viewers then get the cached frame)
    draw: bool = True
    long_poll_timeout_s: float = 2.0

    @classmethod
    def from_dict(cls, d: dict | None) -> LiveConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown live config keys: {sorted(unknown)}")
        return cls(**d)


def colour(n: int) -> tuple[int, int, int]:
    """A stable, distinct BGR colour per ID (golden-ratio hue steps)."""
    r, g, b = colorsys.hsv_to_rgb((n * 0.618033988749895) % 1.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class _Slot:
    def __init__(self):
        self.seq = 0
        self.frame = None
        self.tracks: list = []
        self.provisional: dict[str, tuple[int, float]] = {}
        self.processed_wall = 0.0
        self.jpeg: tuple[int, bytes, dict] | None = None  # (seq, bytes, meta) of the last encode
        self.last_encode = 0.0
        self.encode_lock = threading.Lock()


class LiveHub:
    def __init__(self, cfg: LiveConfig | None = None):
        self.cfg = cfg or LiveConfig()
        self._slots: dict[str, _Slot] = {}
        self._cond = threading.Condition()
        self.encodes = 0
        self.encode_ms_total = 0.0
        self.served = 0

    @property
    def cameras(self) -> list[str]:
        return sorted(self._slots)

    def update(self, frame, tracks, provisional: dict[str, tuple[int, float]] | None = None) -> None:
        """Called by the perception worker after each processed frame (cheap: stores references)."""
        with self._cond:
            s = self._slots.get(frame.camera_id)
            if s is None:
                s = self._slots[frame.camera_id] = _Slot()
            s.seq += 1
            s.frame, s.tracks, s.provisional = frame, list(tracks), dict(provisional or {})
            s.processed_wall = time.time()
            self._cond.notify_all()

    def tracks(self, camera_id: str) -> dict | None:
        """The camera's current tracks (JSON-able), without image."""
        with self._cond:
            s = self._slots.get(camera_id)
            if s is None or s.frame is None:
                return None
            f, tracks, prov = s.frame, s.tracks, s.provisional
        return {"camera_id": camera_id, "seq": s.seq, "frame_index": f.index, "t_wall": f.t_wall,
                "size": [int(f.image.shape[1]), int(f.image.shape[0])],
                "tracks": [{"key": t.key, "local_id": t.local_id, "category": t.phrase, "score": round(float(t.score), 3),
                            "box": [round(float(v), 1) for v in t.box_xyxy],
                            "provisional_global_id": prov.get(t.key, (None, None))[0]} for t in tracks]}

    def wait_jpeg(self, camera_id: str, after_seq: int = 0, timeout: float | None = None
                  ) -> tuple[int, bytes, dict] | None:
        """The newest frame as JPEG once it is newer than ``after_seq`` (waits up to ``timeout``); None on timeout.
        Raises KeyError for an unknown camera."""
        timeout = self.cfg.long_poll_timeout_s if timeout is None else timeout
        end = time.monotonic() + timeout
        with self._cond:
            if camera_id not in self._slots:
                raise KeyError(camera_id)
            s = self._slots[camera_id]
            while s.seq <= after_seq:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(left)
        with s.encode_lock:
            cached = s.jpeg
            fresh = cached is not None and cached[0] > after_seq  # newer than what this viewer has
            if fresh and cached[0] == s.seq:
                self.served += 1
                return cached
            wait = (1.0 / self.cfg.max_fps if self.cfg.max_fps > 0 else 0.0) - (time.monotonic() - s.last_encode)
            if wait > 0:  # encode budget for this camera not yet available
                if fresh:
                    self.served += 1
                    return cached
                time.sleep(wait)
            with self._cond:
                seq, frame, tracks, prov = s.seq, s.frame, s.tracks, s.provisional
            t0 = time.perf_counter()
            data = self._encode(frame, tracks, prov)
            ms = (time.perf_counter() - t0) * 1e3
            meta = {"camera_id": camera_id, "frame_index": frame.index, "t_wall": frame.t_wall,
                    "age_ms": round((time.monotonic() - frame.t_capture) * 1e3, 1), "encode_ms": round(ms, 2),
                    "tracks": len(tracks)}
            s.jpeg, s.last_encode = (seq, data, meta), time.monotonic()
            self.encodes += 1
            self.encode_ms_total += ms
            self.served += 1
            return s.jpeg

    def _encode(self, frame, tracks, prov) -> bytes:
        import cv2

        img = frame.image
        h, w = img.shape[:2]
        scale = min(1.0, self.cfg.stream_width / w)
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # also a copy: the shared frame is never drawn on
        if self.cfg.draw:
            for t in tracks:
                x1, y1, x2, y2 = (int(round(v * scale)) for v in t.box_xyxy)
                gid = prov.get(t.key, (None, None))[0]
                c = colour(gid if gid is not None else t.local_id + 7919)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                label = f"{t.local_id}" + (f" G{gid}?" if gid is not None else "")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = y1 - th - 6 if y1 - th - 6 >= 0 else y1  # above the box, or inside it at the image top
                cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + th + 6), c, -1)
                cv2.putText(img, label, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    def stats(self) -> dict:
        return {"cameras": len(self._slots), "encodes": self.encodes, "frames_served": self.served,
                "encode_ms_mean": round(self.encode_ms_total / max(self.encodes, 1), 2)}


class EventBus:
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.published = self.dropped = 0

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def subscribers(self) -> int:
        return len(self._subs)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, msg: dict) -> None:
        """Thread-safe; returns immediately."""
        loop = self._loop
        if loop is None or loop.is_closed() or not self._subs:
            return
        self.published += 1
        try:
            loop.call_soon_threadsafe(self._fanout, msg)
        except RuntimeError:  # loop closed during shutdown
            pass

    def _fanout(self, msg: dict) -> None:
        for q in list(self._subs):
            if q.full():
                q.get_nowait()
                self.dropped += 1
            q.put_nowait(msg)
