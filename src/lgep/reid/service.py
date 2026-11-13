"""Online Re-ID for the multi-camera pipeline: embed sampled crops, close idle tracks, assign global IDs.

Plug ``ReIDService.on_result`` into ``PerceptionPipeline(on_result=...)``. Per processed frame it

1. updates every track's time span and picks the tracks worth a crop (``TrackFeatureBank``),
2. embeds those crops in one batch (``ClipEmbedder``),
3. closes the tracks of that camera not seen for ``close_after_s`` and links each closed tracklet to a
   global identity (``IdentityManager``).

Times are ``Frame.t_capture`` (one monotonic clock for all cameras of the process), so transit times
between cameras are consistent. Call ``close_all()`` at shutdown.
"""

from __future__ import annotations

import logging
import threading
import time

from lgep.reid.features import SamplingConfig, TrackFeatureBank, crop
from lgep.reid.identity import AssociationConfig, AssociationRecord, CameraTopology, IdentityManager

log = logging.getLogger(__name__)


class ReIDService:
    def __init__(self, embedder, sampling: SamplingConfig | None = None, association: AssociationConfig | None = None,
                 topology: CameraTopology | None = None, close_after_s: float = 3.0, crop_pad: float = 0.0):
        self.embedder = embedder
        self.bank = TrackFeatureBank(sampling)
        self.identities = IdentityManager(association, topology)
        self.close_after_s = close_after_s
        self.crop_pad = crop_pad
        self._lock = threading.Lock()
        self.crops_embedded = 0
        self.embed_ms_total = 0.0
        self.frames = 0

    def on_result(self, frame, det, trk) -> list[AssociationRecord]:
        image = frame.image
        h, w = image.shape[:2]
        with self._lock:
            want = self.bank.observe(frame.camera_id, frame.index, frame.t_capture, trk.tracks, (w, h))
            if want:
                t0 = time.perf_counter()
                feats = self.embedder.embed_images([crop(image, tr.box_xyxy, self.crop_pad) for tr in want])
                self.embed_ms_total += (time.perf_counter() - t0) * 1e3
                self.crops_embedded += len(want)
                for tr, f in zip(want, feats):
                    self.bank.add(tr.key, frame.index, f)
            self.frames += 1
            return self._close(frame.camera_id, frame.t_capture)

    def _close(self, camera_id: str, now: float) -> list[AssociationRecord]:
        stale = [k for k, r in self.bank.tracks.items()
                 if r.camera_id == camera_id and now - r.t_end > self.close_after_s]
        return [self.identities.assign(self.bank.pop(k)) for k in sorted(stale, key=lambda k: self.bank.tracks[k].t_end)]

    def close_all(self) -> list[AssociationRecord]:
        with self._lock:
            keys = sorted(self.bank.tracks, key=lambda k: self.bank.tracks[k].t_end)
            return [self.identities.assign(self.bank.pop(k)) for k in keys]

    def global_id(self, track_key: str) -> int | None:
        return self.identities.global_id_of(track_key)

    def stats(self) -> dict:
        ids = self.identities.identities.values()
        decisions = {}
        for r in self.identities.records:
            decisions[r.decision] = decisions.get(r.decision, 0) + 1
        return {"frames": self.frames, "crops_embedded": self.crops_embedded,
                "embed_ms_per_frame": round(self.embed_ms_total / max(self.frames, 1), 2),
                "open_tracks": len(self.bank.tracks), "global_identities": len(self.identities.identities),
                "multi_camera_identities": sum(1 for i in ids if len(i.cameras) > 1), "decisions": decisions}
