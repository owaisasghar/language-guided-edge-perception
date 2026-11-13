"""Online Re-ID for the multi-camera pipeline: embed sampled crops, close idle tracks, assign global IDs.

Plug ``ReIDService.on_result`` into ``PerceptionPipeline(on_result=...)``. Per processed frame it

1. updates every track's time span and picks the tracks worth a crop (``TrackFeatureBank``),
2. embeds those crops in one batch (``ClipEmbedder``),
3. closes the tracks of that camera not seen for ``close_after_s`` and links each closed tracklet to a
   global identity (``IdentityManager``).

Times are ``Frame.t_capture`` (one monotonic clock for all cameras of the process), so transit times
between cameras are consistent. Call ``close_all()`` at shutdown.

With an ``index`` (lgep.query.index.TrackIndex), every closed track is also added to it for language
queries (Milestone 6): its mean CLIP embedding, body-region embeddings of its best crop (people), a thumbnail.

``provisional(camera_id)`` (Milestone 7, live display): for the camera's open tracks with enough crops, the
identity each would join if it ended now (``IdentityManager.peek``), refreshed at most every
``provisional_every_s`` per track. Global IDs are final only when the track ends.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

import numpy as np

from lgep.reid.features import SamplingConfig, TrackFeatureBank, TrackletRecord, crop
from lgep.reid.identity import AssociationConfig, AssociationRecord, CameraTopology, IdentityManager
from lgep.tracking.segments import TrackSegments, as_segments

log = logging.getLogger(__name__)


class ReIDService:
    def __init__(self, embedder, sampling: SamplingConfig | None = None, association: AssociationConfig | None = None,
                 topology: CameraTopology | None = None, close_after_s: float = 3.0, crop_pad: float = 0.0,
                 index=None, thumbnail_height: int = 128, provisional_every_s: float = 1.0,
                 segments: TrackSegments | None = None, assign_segments: bool = True):
        self.embedder = embedder
        self.bank = TrackFeatureBank(sampling)
        self.identities = IdentityManager(association, topology)
        self.close_after_s = close_after_s
        self.crop_pad = crop_pad
        self.index = index
        self.thumbnail_height = thumbnail_height
        self._wall_offset = time.time() - time.monotonic()  # Frame.t_capture is time.monotonic()
        self._lock = threading.Lock()
        self.crops_embedded = 0
        self.embed_ms_total = 0.0
        self.frames = 0
        self.segments = segments or TrackSegments()  # a tracker key re-found after it ended -> new tracklet key
        # False when the caller assigns segment keys at frame time (Re-ID in its own thread lags behind the
        # perception worker; asking the registry later would give an answer for a different moment)
        self.assign_segments = assign_segments
        self._finished: dict[str, float] = {}  # tracklet key -> when it was finished (ignore later sightings)
        self.ignored_after_finish = 0
        self.provisional_every_s = provisional_every_s
        self._provisional: dict[str, tuple[float, tuple[int, float] | None]] = {}  # key -> (computed at, result)
        self.provisional_ms_total = 0.0

    def on_result(self, frame, det, trk) -> list[AssociationRecord]:
        image = frame.image
        h, w = image.shape[:2]
        with self._lock:
            tracks = as_segments(self.segments, trk.tracks) if self.assign_segments else list(trk.tracks)
            if self._finished:
                kept = [t for t in tracks if t.key not in self._finished]
                self.ignored_after_finish += len(tracks) - len(kept)
                tracks = kept
            want = self.bank.observe(frame.camera_id, frame.index, frame.t_capture, tracks, (w, h))
            if want:
                t0 = time.perf_counter()
                crops = [crop(image, tr.box_xyxy, self.crop_pad) for tr in want]
                feats = self.embedder.embed_images(crops)
                self.embed_ms_total += (time.perf_counter() - t0) * 1e3
                self.crops_embedded += len(want)
                for tr, f, c in zip(want, feats, crops):
                    self.bank.add(tr.key, frame.index, f, c if self.index is not None else None)
            self.frames += 1
            return self._close(frame.camera_id, frame.t_capture)

    def _close(self, camera_id: str, now: float) -> list[AssociationRecord]:
        stale = [k for k, r in self.bank.tracks.items()
                 if r.camera_id == camera_id and now - r.t_end > self.close_after_s]
        return self._finish(sorted(stale, key=lambda k: self.bank.tracks[k].t_end))

    def close_all(self) -> list[AssociationRecord]:
        with self._lock:
            return self._finish(sorted(self.bank.tracks, key=lambda k: self.bank.tracks[k].t_end))

    def provisional(self, camera_id: str, now: float) -> dict[str, tuple[int, float]]:
        """Open track key -> (global ID it would join now, appearance score), for tracks with a match."""
        t0 = time.perf_counter()
        out = {}
        with self._lock:
            center = self.bank.camera_mean(camera_id) if self.bank.cfg.center else None
            for key, rec in self.bank.tracks.items():
                if rec.camera_id != camera_id:
                    continue
                cached = self._provisional.get(key)
                if cached is None or now - cached[0] >= self.provisional_every_s:
                    res = self.identities.peek(dataclasses.replace(rec, center=center)) if rec.embeddings else None
                    cached = self._provisional[key] = (now, res)
                if cached[1] is not None:
                    out[key] = cached[1]
        self.provisional_ms_total += (time.perf_counter() - t0) * 1e3
        return out

    def _finish(self, keys: list[str]) -> list[AssociationRecord]:
        for k in keys:
            self._provisional.pop(k, None)
            if self.assign_segments:
                self.segments.close(k)
        recs = [self.bank.pop(k) for k in keys]
        for r in recs:
            self._finished[r.key] = r.t_end
        if len(self._finished) > 20000:  # forget tracklets finished long ago (bounded memory)
            newest = max(self._finished.values())
            self._finished = {k: t for k, t in self._finished.items() if newest - t < 600}
        out = [self.identities.assign(r) for r in recs]
        if self.index is not None:
            self._index(recs)
        return out

    def _index(self, recs: list[TrackletRecord]) -> None:
        from lgep.query.attributes import region_crop
        from lgep.query.index import TrackEntry

        recs = [r for r in recs if r.mean_embedding is not None]
        jobs = [(i, reg) for i, r in enumerate(recs) if r.phrase == "person" and r.best_crop is not None
                for reg in ("upper", "lower", "head")]
        feats = self.embedder.embed_images([region_crop(recs[i].best_crop, reg) for i, reg in jobs]) if jobs else []
        regions: dict[int, dict] = {}
        for (i, reg), f in zip(jobs, feats):
            regions.setdefault(i, {})[reg] = f
        for i, r in enumerate(recs):
            thumb = None
            if r.best_crop is not None:
                from PIL import Image

                h, w = r.best_crop.shape[:2]
                scale = self.thumbnail_height / max(h, 1)
                thumb = np.asarray(Image.fromarray(r.best_crop).resize((max(1, round(w * scale)), self.thumbnail_height)))
            self.index.add(TrackEntry(r.key, r.camera_id, r.local_id, self.identities.global_id_of(r.key), r.phrase,
                                      r.t_start, r.t_end, r.t_start + self._wall_offset, r.t_end + self._wall_offset,
                                      r.n_samples, {"full": r.mean_embedding, **regions.get(i, {})}, thumb))

    def global_id(self, track_key: str) -> int | None:
        return self.identities.global_id_of(track_key)

    def stats(self) -> dict:
        ids = self.identities.identities.values()
        decisions = {}
        for r in self.identities.records:
            decisions[r.decision] = decisions.get(r.decision, 0) + 1
        return {"frames": self.frames, "crops_embedded": self.crops_embedded,
                "embed_ms_per_frame": round(self.embed_ms_total / max(self.frames, 1), 2),
                "provisional_ms_per_frame": round(self.provisional_ms_total / max(self.frames, 1), 3),
                "tracks_refound_after_end": self.segments.reopened,
                "sightings_ignored_after_finish": self.ignored_after_finish,
                "open_tracks": len(self.bank.tracks), "global_identities": len(self.identities.identities),
                "multi_camera_identities": sum(1 for i in ids if len(i.cameras) > 1), "decisions": decisions}


class ReIDWorker:
    """Runs a ``ReIDService`` in its own thread (Milestone 8), so the perception worker only hands frames over.

    ``submit`` never blocks: when the queue is full the oldest waiting frame is dropped (and counted); Re-ID then
    samples fewer crops of that moment, but detection and tracking keep their rate. Decisions are passed to
    ``on_records`` from the worker thread; ``provisional(camera)`` returns the latest provisional IDs computed
    for that camera. ``stop`` processes what is queued, then ends every open track (``close_all``).
    """

    def __init__(self, service: ReIDService, on_records=None, queue_size: int = 16):
        import queue
        from collections import deque

        self.service = service
        self.on_records = on_records
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._provisional: dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="reid", daemon=True)
        self.submitted = self.dropped = self.processed = 0
        self.lag_ms = deque(maxlen=2000)  # frame captured -> its Re-ID done

    def start(self) -> ReIDWorker:
        self._thread.start()
        return self

    def submit(self, frame, det, trk) -> None:
        import queue

        self.submitted += 1
        while True:
            try:
                self._q.put_nowait((frame, det, trk))
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def provisional(self, camera_id: str) -> dict:
        return self._provisional.get(camera_id, {})

    def _run(self) -> None:
        import queue

        while not (self._stop.is_set() and self._q.empty()):
            try:
                frame, det, trk = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                records = self.service.on_result(frame, det, trk)
                self._provisional[frame.camera_id] = self.service.provisional(frame.camera_id, frame.t_capture)
                self.processed += 1
                self.lag_ms.append((time.monotonic() - frame.t_capture) * 1e3)
                if records and self.on_records:
                    self.on_records(records)
            except Exception:
                log.exception("%s: Re-ID failed", frame.camera_id)

    def stop(self, timeout: float = 30.0) -> list[AssociationRecord]:
        self._stop.set()
        self._thread.join(timeout)
        return self.service.close_all()

    def stats(self) -> dict:
        lag = np.fromiter(self.lag_ms, float) if self.lag_ms else None
        return {**self.service.stats(), "queued": self._q.qsize(), "submitted": self.submitted,
                "dropped": self.dropped, "processed": self.processed,
                "lag_ms_p50": None if lag is None else round(float(np.percentile(lag, 50)), 1),
                "lag_ms_p90": None if lag is None else round(float(np.percentile(lag, 90)), 1)}
