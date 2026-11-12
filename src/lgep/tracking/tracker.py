"""Per-camera multi-object tracking: ByteTrack per phrase, persistent per-camera local IDs.

A ``Track`` is identified by ``(camera_id, local_id)``. The local ID is unique within its camera and is
never reused (see ``IdAllocator``). It is **not** a cross-camera identity: linking tracks across cameras
is Re-ID's job (Milestone 5), which assigns separate global identities.

Association is per phrase: each phrase of the query ("person", "white truck", ...) has its own ByteTrack
state, so a box is only ever matched to tracks of the same phrase. All phrases of one camera draw IDs from
the same allocator, so IDs are unique across phrases too.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lgep.perception.types import DetectionResult
from lgep.tracking.bytetrack import BYTETracker
from lgep.tracking.ids import IdAllocator

log = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    # ByteTrack parameters; defaults are ByteTrack's own (tools/track.py, MOT17).
    track_thresh: float = 0.6  # detections above this start/extend tracks; 0.1 < score <= this: 2nd association
    track_buffer: int = 30  # frames (at 30 fps) a lost track is kept for re-identification by motion
    match_thresh: float = 0.9  # max association cost (1 - IoU x score) in the first association
    mot20: bool = False  # True: no score fusion in the first association (ByteTrack's crowded-scene setting)
    frame_rate: float = 30.0  # of the frames actually fed to the tracker; scales track_buffer
    # Output filters (applied to reported tracks only, never to the tracker state), as in ByteTrack.
    min_box_area: float = 100.0  # px^2; ByteTrack's tools/track.py default
    max_aspect_ratio: float | None = None  # drop boxes with w/h above this; ByteTrack uses 1.6 for pedestrians
    # Persistence of local IDs across restarts: one JSON file per camera in this directory (None = off).
    id_state_dir: str | None = None
    id_block: int = 1000

    @classmethod
    def from_dict(cls, d: dict | None) -> TrackerConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown tracker config keys: {sorted(unknown)}")
        return cls(**d)


@dataclass(frozen=True)
class Track:
    camera_id: str
    local_id: int  # unique within the camera, never reused; NOT a cross-camera identity
    phrase: str
    box_xyxy: tuple[float, float, float, float]  # Kalman-filtered box in image pixels
    score: float  # score of the detection matched in this frame
    start_frame: int  # camera frame index at which the track was started

    @property
    def key(self) -> str:
        return f"{self.camera_id}/{self.local_id}"


@dataclass
class TrackingResult:
    camera_id: str
    frame_index: int  # 1-based count of frames fed to this camera's tracker
    tracks: list[Track] = field(default_factory=list)
    tracker_ms: float = 0.0  # CPU time of the association step for this frame (all phrases)


class CameraTracker:
    """Tracks the detections of one camera. Feed every processed frame in order, with or without detections."""

    def __init__(self, camera_id: str, cfg: TrackerConfig | None = None):
        if not camera_id or "/" in camera_id:
            raise ValueError(f"camera_id must be non-empty and contain no '/': {camera_id!r}")
        self.camera_id = camera_id
        self.cfg = cfg or TrackerConfig()
        state = Path(self.cfg.id_state_dir) / f"{camera_id}.json" if self.cfg.id_state_dir else None
        self.ids = IdAllocator(state, block=self.cfg.id_block)
        self.frame_index = 0
        self._trackers: dict[str, BYTETracker] = {}
        self._offsets: dict[str, int] = {}  # camera frame index = tracker frame id + offset

    @property
    def phrases(self) -> list[str]:
        return list(self._trackers)

    def reset(self) -> None:
        """Forget all tracks (e.g. after a stream gap); later tracks get new, never-used IDs."""
        self._trackers.clear()
        self._offsets.clear()

    def _tracker(self, phrase: str) -> BYTETracker:
        t = self._trackers.get(phrase)
        if t is None:
            c = self.cfg
            t = BYTETracker(c.track_thresh, c.track_buffer, c.match_thresh, c.mot20, c.frame_rate, self.ids)
            self._trackers[phrase] = t
            self._offsets[phrase] = self.frame_index - 1  # its frame 1 is the current camera frame
        return t

    def update(self, detections: dict[str, tuple[np.ndarray, np.ndarray]]) -> TrackingResult:
        """``detections``: phrase -> (boxes (N, 4) x1 y1 x2 y2 px, scores (N,)), for every phrase of the
        current query, including phrases with no detections in this frame. Trackers of phrases that are no
        longer in the query are dropped (their tracks end)."""
        self.frame_index += 1
        t0 = time.perf_counter()
        for gone in set(self._trackers) - set(detections):
            log.info("%s: phrase %r left the query; its tracks end", self.camera_id, gone)
            del self._trackers[gone], self._offsets[gone]
        tracks: list[Track] = []
        for phrase, (boxes, scores) in detections.items():
            tracker = self._tracker(phrase)
            for st in tracker.update(boxes, scores):
                x, y, w, h = st.tlwh
                if not self._keep(w, h):
                    continue
                tracks.append(Track(self.camera_id, int(st.track_id), phrase, (x, y, x + w, y + h),
                                    float(st.score), st.start_frame + self._offsets[phrase]))
        return TrackingResult(self.camera_id, self.frame_index, tracks, (time.perf_counter() - t0) * 1e3)

    def update_from_result(self, result: DetectionResult) -> TrackingResult:
        """Convenience wrapper: group a detector's output by phrase and call :meth:`update`."""
        dets = {}
        for i, phrase in enumerate(result.phrases):
            sel = [d for d in result.detections if d.phrase_id == i]
            boxes = np.array([d.box_xyxy for d in sel], dtype=np.float64).reshape(-1, 4)
            dets[phrase] = (boxes, np.array([d.score for d in sel], dtype=np.float64))
        return self.update(dets)

    def _keep(self, w: float, h: float) -> bool:
        # Same test and order as ByteTrack's evaluator: area > min_box_area and not "vertical" (w/h > 1.6).
        if w * h <= self.cfg.min_box_area:
            return False
        return not (self.cfg.max_aspect_ratio is not None and h > 0 and w / h > self.cfg.max_aspect_ratio)
