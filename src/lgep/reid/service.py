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
                 segments: TrackSegments | None = None):
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
        self.provisional_every_s = provisional_every_s
        self._provisional: dict[str, tuple[float, tuple[int, float] | None]] = {}  # key -> (computed at, result)
        self.provisional_ms_total = 0.0

    def on_result(self, frame, det, trk) -> list[AssociationRecord]:
        image = frame.image
        h, w = image.shape[:2]
        with self._lock:
            tracks = as_segments(self.segments, trk.tracks)
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
            self.segments.close(k)
        recs = [self.bank.pop(k) for k in keys]
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
                "open_tracks": len(self.bank.tracks), "global_identities": len(self.identities.identities),
                "multi_camera_identities": sum(1 for i in ids if len(i.cameras) > 1), "decisions": decisions}
