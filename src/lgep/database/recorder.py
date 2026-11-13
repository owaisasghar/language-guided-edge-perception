"""Turns the live pipeline's output into store operations (runs in the perception worker; must stay cheap).

* ``Recorder.observe(frame, tracking)``: opens a track the first time its key appears, stores one box per
  ``points_every_s``, and closes tracks of that camera not seen for ``close_after_s`` (the same rule as the
  Re-ID service, on the same clock).
* ``Recorder.associations(records)``: the Re-ID service's decisions (global IDs, with evidence).
* ``PersistentTrackIndex``: the Milestone 6 track index, whose entries are also written to the store
  (thumbnail + CLIP embeddings), so search works across restarts; ``load_index`` rebuilds it.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass

import numpy as np

from lgep.database.store import MetadataStore
from lgep.query.index import REGIONS, TrackEntry, TrackIndex
from lgep.tracking.segments import TrackSegments, as_segments

log = logging.getLogger(__name__)


@dataclass
class _OpenTrack:
    camera_id: str
    last_seen: float  # monotonic (Frame.t_capture)
    wall_end: float
    frames: int
    max_score: float
    last_point: float  # wall time of the last stored point


class Recorder:
    """``segments``: share the Re-ID service's ``TrackSegments`` so both store the same tracklet keys."""

    def __init__(self, store: MetadataStore, segments: TrackSegments | None = None):
        self.store = store
        self.cfg = store.cfg
        self.segments = segments or TrackSegments()
        self._open: dict[str, _OpenTrack] = {}

    @property
    def open_tracks(self) -> int:
        return len(self._open)

    def observe(self, frame, tracking) -> None:
        now, wall = frame.t_capture, frame.t_wall
        every = self.cfg.points_every_s
        for t in as_segments(self.segments, tracking.tracks):
            o = self._open.get(t.key)
            if o is None:
                o = self._open[t.key] = _OpenTrack(t.camera_id, now, wall, 0, t.score, -1e18)
                self.store.submit(("open", t.key, t.camera_id, t.local_id, t.phrase, wall, float(t.score)))
            o.last_seen, o.wall_end = now, wall
            o.frames += 1
            o.max_score = max(o.max_score, float(t.score))
            if wall - o.last_point >= every:
                x1, y1, x2, y2 = (float(v) for v in t.box_xyxy)
                self.store.submit(("point", t.key, wall, x1, y1, x2, y2, float(t.score)))
                self.store.submit(("update", t.key, wall, o.frames, o.max_score))
                o.last_point = wall
        stale = [k for k, o in self._open.items()
                 if o.camera_id == frame.camera_id and now - o.last_seen > self.cfg.close_after_s]
        for k in stale:
            self._close(k)

    def _close(self, key: str) -> None:
        o = self._open.pop(key)
        self.segments.close(key)
        self.store.submit(("close", key, o.wall_end, o.frames, o.max_score))

    def close_all(self) -> None:
        for k in list(self._open):
            self._close(k)

    def associations(self, records) -> None:
        for r in records:
            ev = {k: v for k, v in r.evidence.items() if k in ("top", "candidates", "samples")}
            self.store.submit(("assoc", r.track_key, r.global_id, r.decision, r.score, json.dumps(ev),
                               r.evidence.get("samples")))

    def camera_event(self, type_: str, t: float, camera_id: str, data: dict) -> None:
        self.store.submit(("event", type_, t, camera_id, data))


def encode_thumbnail(rgb: np.ndarray | None, quality: int = 85) -> bytes | None:
    if rgb is None:
        return None
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pack_embeddings(embeddings: dict[str, np.ndarray]) -> tuple[bytes, str, int]:
    """Region embeddings -> float16 bytes, region list, dimension (float16 halves the size; scores change by
    less than 1e-3, see tests)."""
    regions = [r for r in REGIONS if r in embeddings]
    arr = np.stack([embeddings[r] for r in regions]).astype(np.float16)
    return arr.tobytes(), ",".join(regions), int(arr.shape[1])


def unpack_embeddings(blob: bytes, regions: str, dim: int) -> dict[str, np.ndarray]:
    arr = np.frombuffer(blob, np.float16).reshape(-1, dim).astype(np.float32)
    return {r: arr[i] for i, r in enumerate(regions.split(","))}


class PersistentTrackIndex(TrackIndex):
    """``TrackIndex`` whose new entries are also written to the store (current run)."""

    def __init__(self, store: MetadataStore, entries: list[TrackEntry] | None = None):
        super().__init__(entries)
        self.store = store

    def add(self, entry: TrackEntry) -> None:
        entry.run_id = self.store.run_id
        super().add(entry)
        blob, regions, dim = pack_embeddings(entry.embeddings)
        thumb = encode_thumbnail(entry.thumbnail) if self.store.cfg.thumbnails else None
        self.store.submit(("media", entry.key, thumb, blob, regions, dim))


def load_index(store: MetadataStore, since_wall: float | None = None) -> list[TrackEntry]:
    """Index entries of earlier runs (closed tracks with embeddings), oldest first; thumbnails stay in the DB."""
    sql = ("SELECT t.id, t.run_id, t.key, t.camera_id, t.local_id, t.category, t.t_start, t.t_end, t.samples, "
           "i.global_id, m.embeddings, m.regions, m.dim FROM tracks t JOIN track_media m ON m.track_id = t.id "
           "LEFT JOIN identities i ON i.id = t.identity_id WHERE m.embeddings IS NOT NULL")
    args: tuple = ()
    if since_wall is not None:
        sql += " AND t.t_end >= ?"
        args = (since_wall,)
    entries = []
    with store.reader() as conn:
        for r in conn.execute(sql + " ORDER BY t.t_end", args):
            entries.append(TrackEntry(r["key"], r["camera_id"], r["local_id"], r["global_id"], r["category"],
                                      r["t_start"], r["t_end"], r["t_start"], r["t_end"], r["samples"] or 0,
                                      unpack_embeddings(r["embeddings"], r["regions"], r["dim"]), None,
                                      run_id=r["run_id"], track_id=r["id"]))
    return entries
