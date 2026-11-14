"""Asynchronous multi-camera perception: capture threads feed one shared detector and per-camera trackers.

Capture (one thread per camera, ``lgep.ingestion``) is decoupled from processing (this worker): cameras
keep decoding at their own rate while the worker takes each camera's newest frame, so a slow detector
never makes a camera fall behind; it skips frames instead, and every skip is counted.

Scheduling: round-robin over cameras that have a frame newer than the last one processed and whose
per-camera rate limit (``process_fps``) allows another frame. One detector (one model on the GPU) serves
all cameras. With ``batch_size`` > 1 (Milestone 8) the worker takes every camera that is ready at that
moment, up to ``batch_size``, and detects them in one call (``detector.detect_batch``); it never waits
for a camera that is not ready, so batching adds no waiting time to latency.

When a camera's connection epoch changes (reconnect, or a looping file restarting) the camera's tracker
is reset: tracks never span a gap in the video, and new tracks get new, never-reused local IDs.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from lgep.ingestion.manager import StreamManager
from lgep.ingestion.source import Frame
from lgep.perception.prompt import normalize_phrases
from lgep.perception.types import DetectionResult
from lgep.tracking import CameraTracker, TrackerConfig, TrackingResult

log = logging.getLogger(__name__)
ResultCallback = Callable[[Frame, DetectionResult, TrackingResult], None]


@dataclass
class PipelineConfig:
    prompt: str | list[str] = "person"
    process_fps: float = 10.0  # per camera; <= 0 processes every new frame
    tracker: dict = field(default_factory=dict)  # TrackerConfig fields; frame_rate defaults to process_fps
    window: int = 600  # samples kept per camera for latency / rate statistics
    batch_size: int = 1  # frames of different cameras detected in one call (1 = one frame at a time)

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown pipeline config keys: {sorted(unknown)}")
        return cls(**d)


class _CameraRuntime:
    def __init__(self, camera_id: str, tracker_cfg: TrackerConfig, window: int):
        self.tracker = CameraTracker(camera_id, tracker_cfg)
        self.last_index = 0
        self.epoch: int | None = None
        self.next_due = 0.0
        self.processed = 0
        self.errors = 0
        self.resets = 0
        self.done_at = deque(maxlen=window)  # completion times -> processed fps
        self.latency_ms = deque(maxlen=window)  # frame decoded -> tracks ready
        self.wait_ms = deque(maxlen=window)  # frame decoded -> processing started
        self.detector_ms = deque(maxlen=window)
        self.tracker_ms = deque(maxlen=window)
        self.batch = deque(maxlen=window)  # size of the detector call this camera's frame was part of

    def clear(self) -> None:
        for d in (self.done_at, self.latency_ms, self.wait_ms, self.detector_ms, self.tracker_ms, self.batch):
            d.clear()


def _pct(values, q) -> float | None:
    return None if not values else round(float(np.percentile(np.fromiter(values, float), q)), 2)


class PerceptionPipeline:
    def __init__(self, manager: StreamManager, detector, cfg: PipelineConfig,
                 on_result: ResultCallback | None = None, clock: Callable[[], float] = time.monotonic):
        self.manager = manager
        self.detector = detector
        self.cfg = cfg
        self.phrases = normalize_phrases(cfg.prompt)
        self.on_result = on_result
        self._clock = clock
        self._interval = 1.0 / cfg.process_fps if cfg.process_fps and cfg.process_fps > 0 else 0.0
        frame_rate = cfg.process_fps if cfg.process_fps and cfg.process_fps > 0 else 30.0
        self._tracker_cfg = TrackerConfig.from_dict({"frame_rate": frame_rate, **cfg.tracker})
        self._cams: dict[str, _CameraRuntime] = {}
        self._rr = 0
        self._started_at: float | None = None

    def _runtime(self, camera_id: str) -> _CameraRuntime:
        rt = self._cams.get(camera_id)
        if rt is None:
            rt = self._cams[camera_id] = _CameraRuntime(camera_id, self._tracker_cfg, self.cfg.window)
        return rt

    # ------------------------------------------------------------------------------------------ loop
    def step(self) -> bool:
        """Process the next due camera (or, with batching, every due camera up to ``batch_size``). True if
        anything was processed."""
        if self.cfg.batch_size > 1:
            return self._step_batch()
        ids = self.manager.camera_ids
        now = self._clock()
        for k in range(len(ids)):
            cid = ids[(self._rr + k) % len(ids)]
            rt = self._runtime(cid)
            if rt.next_due > now:
                continue
            # Rate-limited cameras: take the first frame captured at or after the due time and process it as it
            # arrives. Taking whatever is newest at the due time adds the age of that frame (up to one camera
            # frame period) to the latency. The processing rate is unchanged.
            fresh = rt.next_due if self._interval and rt.processed else None
            frame = self.manager.latest(cid, rt.last_index, fresh)
            if frame is None:
                continue
            self._rr = (self._rr + k + 1) % len(ids)
            self._process(rt, frame, now)
            return True
        return False

    def run(self, stop: threading.Event | None = None, duration_s: float | None = None,
            idle_wait_s: float = 0.02) -> None:
        """Process until ``stop`` is set or ``duration_s`` has passed."""
        stop = stop or threading.Event()
        self._started_at = self._started_at or self._clock()
        end = None if duration_s is None else self._clock() + duration_s
        while not stop.is_set() and (end is None or self._clock() < end):
            if not self.step():
                self.manager.wait_for_frame(idle_wait_s)

    def _step_batch(self) -> bool:
        ids = self.manager.camera_ids
        now = self._clock()
        batch: list[tuple[_CameraRuntime, Frame]] = []
        last = None
        for k in range(len(ids)):
            cid = ids[(self._rr + k) % len(ids)]
            rt = self._runtime(cid)
            if rt.next_due > now:
                continue
            fresh = rt.next_due if self._interval and rt.processed else None
            frame = self.manager.latest(cid, rt.last_index, fresh)
            if frame is None:
                continue
            batch.append((rt, frame))
            last = k
            if len(batch) == self.cfg.batch_size:
                break
        if not batch:
            return False
        self._rr = (self._rr + last + 1) % len(ids)
        for rt, frame in batch:
            self._admit(rt, frame, now)
        t0 = self._clock()
        try:
            if hasattr(self.detector, "detect_batch"):
                dets = self.detector.detect_batch([f.image for _, f in batch], self.phrases)
            else:
                dets = [self.detector.detect(f.image, self.phrases) for _, f in batch]
        except Exception:
            for rt, frame in batch:
                rt.errors += 1
            log.exception("detection of a batch of %d frames failed", len(batch))
            return True
        for (rt, frame), det in zip(batch, dets):
            try:
                trk = rt.tracker.update_from_result(det, frame.image if rt.tracker.needs_frames else None)
            except Exception:
                rt.errors += 1
                log.exception("%s: tracking frame %d failed", frame.camera_id, frame.index)
                continue
            self._finish(rt, frame, det, trk, t0, len(batch))
        return True

    def _admit(self, rt: _CameraRuntime, frame: Frame, now: float) -> None:
        """Epoch check and rate-limit bookkeeping for a frame about to be processed."""
        if rt.epoch is not None and frame.epoch != rt.epoch:
            rt.tracker.reset()
            rt.resets += 1
            log.info("%s: video interrupted (epoch %d -> %d); tracker reset", frame.camera_id, rt.epoch, frame.epoch)
        rt.epoch = frame.epoch
        rt.last_index = frame.index
        if self._interval:  # keep the cadence when on time; restart it (no catch-up burst) when late
            nxt = rt.next_due + self._interval
            rt.next_due = nxt if nxt > now else now + self._interval

    def _finish(self, rt: _CameraRuntime, frame: Frame, det, trk, t0: float, batch: int) -> None:
        t1 = self._clock()
        rt.processed += 1
        rt.done_at.append(t1)
        rt.latency_ms.append((t1 - frame.t_capture) * 1e3)
        rt.wait_ms.append((t0 - frame.t_capture) * 1e3)
        rt.detector_ms.append(det.timings.total_ms)
        rt.tracker_ms.append(trk.tracker_ms)
        rt.batch.append(batch)
        if self.on_result:
            try:
                self.on_result(frame, det, trk)
            except Exception:
                log.exception("%s: result callback failed", frame.camera_id)

    def _process(self, rt: _CameraRuntime, frame: Frame, now: float) -> None:
        self._admit(rt, frame, now)
        t0 = self._clock()
        try:
            det = self.detector.detect(frame.image, self.phrases)
            trk = rt.tracker.update_from_result(det, frame.image if rt.tracker.needs_frames else None)
        except Exception:
            rt.errors += 1
            log.exception("%s: processing frame %d failed", frame.camera_id, frame.index)
            return
        self._finish(rt, frame, det, trk, t0, 1)

    # ----------------------------------------------------------------------------------------- stats
    def stats(self) -> dict:
        cams = {}
        for cid, rt in self._cams.items():
            d = rt.done_at
            fps = (len(d) - 1) / (d[-1] - d[0]) if len(d) > 1 and d[-1] > d[0] else 0.0
            cams[cid] = {"processed": rt.processed, "processed_fps": round(fps, 2), "errors": rt.errors,
                         "tracker_resets": rt.resets, "epoch": rt.epoch,
                         "latency_ms_p50": _pct(rt.latency_ms, 50), "latency_ms_p90": _pct(rt.latency_ms, 90),
                         "latency_ms_p99": _pct(rt.latency_ms, 99),
                         "batch_mean": None if not rt.batch else round(float(np.mean(rt.batch)), 2),
                         "latency_ms_max": None if not rt.latency_ms else round(max(rt.latency_ms), 2),
                         "wait_ms_p50": _pct(rt.wait_ms, 50), "detector_ms_p50": _pct(rt.detector_ms, 50),
                         "tracker_ms_p50": _pct(rt.tracker_ms, 50),
                         "next_local_id": rt.tracker.ids.next_id}
        return {"target_process_fps": self.cfg.process_fps, "phrases": self.phrases, "batch_size": self.cfg.batch_size,
                "cameras": cams}

    def clear_stats(self) -> None:
        """Forget the rolling latency / rate samples (benchmarks: measure one window exactly)."""
        for rt in self._cams.values():
            rt.clear()
