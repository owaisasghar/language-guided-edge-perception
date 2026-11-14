"""Per-camera multi-object tracking: ByteTrack, OC-SORT or BoT-SORT per phrase, persistent per-camera local IDs.

A ``Track`` is identified by ``(camera_id, local_id)``. The local ID is unique within its camera and is
never reused (see ``IdAllocator``). It is **not** a cross-camera identity: linking tracks across cameras
is Re-ID's job (Milestone 5), which assigns separate global identities.

Association is per phrase: each phrase of the query ("person", "white truck", ...) has its own tracker
state, so a box is only ever matched to tracks of the same phrase. All phrases of one camera draw IDs from
the same allocator, so IDs are unique across phrases too.

Trackers (``TrackerConfig.algorithm``; official implementations, vendored with equality tests):
* ``bytetrack`` (Zhang et al., ECCV 2022): Kalman motion + IoU, a second association for low-score boxes.
* ``ocsort`` (Cao et al., CVPR 2023): observation-centric SORT; re-updates the motion model over occlusions
  and adds a direction-consistency cost. Tracks are reported after ``min_hits`` consecutive matches.
* ``botsort`` (Aharon et al., 2022): ByteTrack's association with a width/height Kalman state and camera-
  motion compensation (sparse optical flow on the frame, computed once per camera frame); optional
  appearance features.
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

ALGORITHMS = ("bytetrack", "ocsort", "botsort")
# Each tracker's own MOT17 settings (official evaluation scripts). The shared fields of TrackerConfig give the
# high-score threshold (track_thresh) and how long lost tracks are kept (track_buffer, scaled by frame rate).
OCSORT_DEFAULTS = {"iou_threshold": 0.3, "min_hits": 3, "delta_t": 3, "inertia": 0.2, "asso_func": "iou",
                   "use_byte": False}
BOTSORT_DEFAULTS = {"new_track_delta": 0.1,  # new tracks need track_thresh + 0.1 (BoT-SORT's MOT17 rule)
                    "track_low_thresh": 0.1, "match_thresh": 0.8, "proximity_thresh": 0.5,
                    "appearance_thresh": 0.25, "cmc": "sparseOptFlow", "cmc_downscale": 2}

log = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    algorithm: str = "bytetrack"  # bytetrack | ocsort | botsort
    # ByteTrack parameters; defaults are ByteTrack's own (tools/track.py, MOT17). track_thresh and track_buffer
    # apply to all three trackers (OC-SORT: det_thresh, max_age; BoT-SORT: track_high_thresh, track_buffer).
    track_thresh: float = 0.6  # detections above this start/extend tracks; 0.1 < score <= this: 2nd association
    track_buffer: int = 30  # frames (at 30 fps) a lost track is kept for re-identification by motion
    match_thresh: float = 0.9  # max association cost (1 - IoU x score) in the first association
    mot20: bool = False  # True: no score fusion in the first association (ByteTrack's crowded-scene setting)
    frame_rate: float = 30.0  # of the frames actually fed to the tracker; scales track_buffer
    # Output filters (applied to reported tracks only, never to the tracker state), as in ByteTrack.
    min_box_area: float = 100.0  # px^2; ByteTrack's tools/track.py default
    max_aspect_ratio: float | None = None  # drop boxes with w/h above this; ByteTrack uses 1.6 for pedestrians
    # phrases the aspect-ratio filter applies to (None = all). Cars are often wider than 1.6 x their height.
    aspect_filter_phrases: list[str] | None = field(default_factory=lambda: ["person"])
    # phrases tracked together under one name, e.g. {"vehicle": ["car", "truck", "bus"]}: open-vocabulary
    # detectors flip between "car" and "truck" for the same van, which would split its track per label
    phrase_groups: dict[str, list[str]] = field(default_factory=dict)
    # OC-SORT / BoT-SORT parameters that differ from ByteTrack's (keys of OCSORT_DEFAULTS / BOTSORT_DEFAULTS)
    ocsort: dict = field(default_factory=dict)
    botsort: dict = field(default_factory=dict)
    # Persistence of local IDs across restarts: one JSON file per camera in this directory (None = off).
    id_state_dir: str | None = None
    id_block: int = 1000

    @classmethod
    def from_dict(cls, d: dict | None) -> TrackerConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown tracker config keys: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"tracker.algorithm must be one of {ALGORITHMS}, got {self.algorithm!r}")
        for name, given, known in (("ocsort", self.ocsort, OCSORT_DEFAULTS), ("botsort", self.botsort, BOTSORT_DEFAULTS)):
            unknown = set(given or {}) - set(known)
            if unknown:
                raise ValueError(f"Unknown tracker.{name} keys: {sorted(unknown)}")

    @property
    def lost_frames(self) -> int:
        """Frames a lost track is kept, as ByteTrack computes it: track_buffer at 30 fps, scaled by frame rate."""
        return int(self.frame_rate / 30.0 * self.track_buffer)


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
    motion_ms: float = 0.0  # CPU time of camera-motion estimation (BoT-SORT), not included in tracker_ms


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
        self.cfg.validate()
        self._trackers: dict = {}
        self._offsets: dict[str, int] = {}  # camera frame index = tracker frame id + offset
        self._new_motion = None  # factory of BoT-SORT's camera-motion estimator (one per camera)
        self._warned_no_frame = False
        if self.cfg.algorithm == "botsort":
            b = {**BOTSORT_DEFAULTS, **self.cfg.botsort}
            if b["cmc"] not in ("none", None):
                from lgep.tracking.botsort import GMC

                self._new_motion = lambda: GMC(b["cmc"], downscale=b["cmc_downscale"])
        self._motion = self._new_motion() if self._new_motion else None

    @property
    def phrases(self) -> list[str]:
        return list(self._trackers)

    @property
    def needs_frames(self) -> bool:
        """True if update() should get the camera image (BoT-SORT's camera-motion compensation)."""
        return self._motion is not None

    def reset(self) -> None:
        """Forget all tracks (e.g. after a stream gap); later tracks get new, never-used IDs."""
        self._trackers.clear()
        self._offsets.clear()
        self._motion = self._new_motion() if self._new_motion else None  # the next frame has no predecessor

    def _tracker(self, phrase: str):
        t = self._trackers.get(phrase)
        if t is None:
            c = self.cfg
            if c.algorithm == "ocsort":
                from lgep.tracking.ocsort import OCSort

                o = {**OCSORT_DEFAULTS, **c.ocsort}
                t = OCSort(c.track_thresh, max_age=c.lost_frames, min_hits=o["min_hits"],
                           iou_threshold=o["iou_threshold"], delta_t=o["delta_t"], asso_func=o["asso_func"],
                           inertia=o["inertia"], use_byte=o["use_byte"], id_allocator=self.ids)
            elif c.algorithm == "botsort":
                from lgep.tracking.botsort import BoTSORT

                b = {**BOTSORT_DEFAULTS, **c.botsort}
                t = BoTSORT(track_high_thresh=c.track_thresh, track_low_thresh=b["track_low_thresh"],
                            new_track_thresh=c.track_thresh + b["new_track_delta"], track_buffer=c.track_buffer,
                            match_thresh=b["match_thresh"], proximity_thresh=b["proximity_thresh"],
                            appearance_thresh=b["appearance_thresh"], with_reid=False, mot20=c.mot20,
                            frame_rate=c.frame_rate, id_allocator=self.ids)
            else:
                t = BYTETracker(c.track_thresh, c.track_buffer, c.match_thresh, c.mot20, c.frame_rate, self.ids)
            self._trackers[phrase] = t
            self._offsets[phrase] = self.frame_index - 1  # its frame 1 is the current camera frame
        return t

    def update(self, detections: dict[str, tuple[np.ndarray, np.ndarray]], image: np.ndarray | None = None,
               rgb: bool = True, warp: np.ndarray | None = None) -> TrackingResult:
        """``detections``: phrase -> (boxes (N, 4) x1 y1 x2 y2 px, scores (N,)), for every phrase of the
        current query, including phrases with no detections in this frame. Trackers of phrases that are no
        longer in the query are dropped (their tracks end).

        ``image`` (H x W x 3, RGB unless ``rgb`` is False): the frame, used by BoT-SORT's camera-motion
        compensation (once per camera frame, shared by all phrases); ignored by the other trackers. ``warp``:
        a precomputed 2 x 3 camera motion to use instead (evaluation)."""
        self.frame_index += 1
        motion_ms = 0.0
        if self._motion is not None and warp is None:
            if image is None:
                if not self._warned_no_frame:
                    log.warning("%s: BoT-SORT camera-motion compensation needs the frames; running without it",
                                self.camera_id)
                    self._warned_no_frame = True
            else:
                tm = time.perf_counter()
                warp = self._motion.apply(image, rgb=rgb)
                motion_ms = (time.perf_counter() - tm) * 1e3
        t0 = time.perf_counter()
        for gone in set(self._trackers) - set(detections):
            log.info("%s: phrase %r left the query; its tracks end", self.camera_id, gone)
            del self._trackers[gone], self._offsets[gone]
        tracks: list[Track] = []
        for phrase, (boxes, scores) in detections.items():
            tracker = self._tracker(phrase)
            out = tracker.update(boxes, scores, warp=warp) if self.cfg.algorithm == "botsort" else tracker.update(boxes, scores)
            for st in out:
                x, y, w, h = st.tlwh
                if not self._keep(w, h, phrase):
                    continue
                tracks.append(Track(self.camera_id, int(st.track_id), phrase, (x, y, x + w, y + h),
                                    float(st.score), st.start_frame + self._offsets[phrase]))
        return TrackingResult(self.camera_id, self.frame_index, tracks, (time.perf_counter() - t0) * 1e3, motion_ms)

    def update_from_result(self, result: DetectionResult, image: np.ndarray | None = None, rgb: bool = True) -> TrackingResult:
        """Convenience wrapper: group a detector's output by phrase and call :meth:`update`."""
        group_of = {p: g for g, members in self.cfg.phrase_groups.items() for p in members}
        boxes: dict[str, list] = {}
        scores: dict[str, list] = {}
        for phrase in result.phrases:  # every phrase (group) of the query, also without detections
            name = group_of.get(phrase, phrase)
            boxes.setdefault(name, [])
            scores.setdefault(name, [])
        for d in result.detections:
            name = group_of.get(d.phrase, d.phrase)
            boxes[name].append(d.box_xyxy)
            scores[name].append(d.score)
        return self.update({k: (np.array(boxes[k], dtype=np.float64).reshape(-1, 4),
                                 np.array(scores[k], dtype=np.float64)) for k in boxes}, image, rgb)

    def _keep(self, w: float, h: float, phrase: str = "person") -> bool:
        # Same test and order as ByteTrack's evaluator: area > min_box_area and not "vertical" (w/h > 1.6); the
        # aspect test only for pedestrian phrases (``aspect_filter_phrases``), never for vehicles.
        if w * h <= self.cfg.min_box_area:
            return False
        phrases = self.cfg.aspect_filter_phrases
        if self.cfg.max_aspect_ratio is None or (phrases is not None and phrase not in phrases):
            return True
        return not (h > 0 and w / h > self.cfg.max_aspect_ratio)
