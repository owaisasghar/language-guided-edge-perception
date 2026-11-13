"""Tracklet segments: one tracker key, possibly several finished tracklets.

ByteTrack keeps a lost track for a number of *frames* (``track_buffer``, scaled by the configured frame rate);
the Re-ID service and the store end a track after ``close_after_s`` *seconds* without it. When processing
slows down (a busy CPU, fewer frames per second than configured), a track can be re-found by the tracker
after it was already ended downstream. Its key then comes back. Such a key starts a new segment,
``cam01/17~2``, so every finished tracklet has a unique key (needed by the identity manager and the store).

Shared by the Re-ID service and the recorder, so both use the same segment keys (thread-safe: with Re-ID in
its own worker, Milestone 8, they run in different threads). ``close`` is idempotent: closing a segment that is
not current does nothing.
"""

from __future__ import annotations

import threading


class TrackSegments:
    def __init__(self):
        self._current: dict[str, str] = {}  # tracker key -> open segment key
        self._count: dict[str, int] = {}  # tracker key -> segments started
        self.reopened = 0
        self._lock = threading.Lock()

    def current(self, key: str) -> str:
        """The open segment of tracker ``key`` (a new one if its last segment was closed)."""
        with self._lock:
            seg = self._current.get(key)
            if seg is None:
                n = self._count.get(key, 0) + 1
                self._count[key] = n
                seg = self._current[key] = key if n == 1 else f"{key}~{n}"
                if n > 1:
                    self.reopened += 1
            return seg

    def close(self, segment: str) -> None:
        base = segment.split("~", 1)[0]
        with self._lock:
            if self._current.get(base) == segment:
                del self._current[base]


class SegmentTrack:
    """A ``Track`` seen under its segment key (same attributes, different ``key``)."""

    __slots__ = ("key", "camera_id", "local_id", "phrase", "box_xyxy", "score", "start_frame")

    def __init__(self, track, key: str):
        self.key = key
        self.camera_id, self.local_id, self.phrase = track.camera_id, track.local_id, track.phrase
        self.box_xyxy, self.score, self.start_frame = track.box_xyxy, track.score, track.start_frame


def as_segments(segments: TrackSegments, tracks) -> list:
    out = []
    for t in tracks:
        seg = segments.current(t.key)
        out.append(t if seg == t.key else SegmentTrack(t, seg))
    return out
