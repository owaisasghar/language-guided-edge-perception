"""Track index for language queries: one entry per finished local track, searchable across cameras.

Each entry keeps what a query needs without the video: camera, local and global ID, category (the detector
phrase), time span (monotonic and wall clock), the mean CLIP embedding of the track's sampled crops, CLIP
embeddings of the best crop's body regions (for clothing colours), and a small thumbnail.

``TrackIndex.search(query)``: filter by category, camera and time, score with ``QueryScorer`` (description
similarity + colour attributes), rank, and report for each hit the cameras where its global identity was
seen ("which cameras saw this vehicle?").
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from lgep.query.attributes import QueryScorer
from lgep.query.parser import StructuredQuery

REGIONS = ("full", "upper", "lower", "head")


@dataclass
class TrackEntry:
    key: str
    camera_id: str
    local_id: int
    global_id: int | None
    category: str
    t_start: float  # monotonic seconds (one clock per process)
    t_end: float
    wall_start: float  # Unix time
    wall_end: float
    samples: int
    embeddings: dict[str, np.ndarray] = field(default_factory=dict, repr=False)  # region -> (D,)
    thumbnail: np.ndarray | None = field(default=None, repr=False)


@dataclass
class SearchHit:
    rank: int
    score: float
    text: float
    attributes: float
    entry: TrackEntry
    identity_cameras: list[str]

    def row(self) -> dict:
        e = self.entry
        return {"rank": self.rank, "score": round(self.score, 4), "text": round(self.text, 4),
                "attributes": round(self.attributes, 4), "track": e.key, "camera": e.camera_id, "global_id": e.global_id,
                "category": e.category, "seen_in_cameras": self.identity_cameras,
                "start": datetime.fromtimestamp(e.wall_start).isoformat(timespec="seconds"),
                "duration_s": round(e.t_end - e.t_start, 1)}


class TrackIndex:
    def __init__(self, entries: list[TrackEntry] | None = None):
        self.entries: list[TrackEntry] = list(entries or [])

    def add(self, entry: TrackEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    # -------------------------------------------------------------------------------------------- search
    def _time_ok(self, e: TrackEntry, q: StructuredQuery, now: float) -> bool:
        tf = q.time
        if tf is None:
            return True
        if tf.last_s is not None and e.wall_end < now - tf.last_s:
            return False

        def clock(t: float) -> str:
            return datetime.fromtimestamp(t).strftime("%H:%M")

        if tf.after and clock(e.wall_end) < tf.after:
            return False
        return not (tf.before and clock(e.wall_start) > tf.before)

    def search(self, q: StructuredQuery, scorer: QueryScorer, k: int = 10, now: float | None = None,
               one_per_identity: bool = True) -> list[SearchHit]:
        """Top-``k`` tracks for ``q``. With ``one_per_identity``, each global identity appears once (its best
        track), so one object seen many times does not fill the list."""
        now = time.time() if now is None else now
        cams = {c.lower() for c in q.cameras}
        cand = [e for e in self.entries
                if (not q.categories or e.category in q.categories)
                and (not cams or e.camera_id.lower() in cams)
                and self._time_ok(e, q, now) and "full" in e.embeddings]
        if not cand:
            return []
        need = scorer.regions_needed(q)
        embs = {r: np.stack([e.embeddings.get(r, e.embeddings["full"]) for e in cand]) for r in need}
        s = scorer.score(q, embs, None)
        order, seen = [], set()
        for j in np.argsort(-s["total"], kind="stable"):
            gid = cand[j].global_id
            if one_per_identity and gid is not None:
                if gid in seen:
                    continue
                seen.add(gid)
            order.append(j)
            if len(order) == k:
                break
        cams_of = {}
        for e in self.entries:
            if e.global_id is not None:
                cams_of.setdefault(e.global_id, set()).add(e.camera_id)
        return [SearchHit(i + 1, float(s["total"][j]), float(s["text"][j]), float(s["attributes"][j]), cand[j],
                          sorted(cams_of.get(cand[j].global_id, {cand[j].camera_id})))
                for i, j in enumerate(order)]

    # --------------------------------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        from PIL import Image

        d = Path(directory)
        (d / "thumbnails").mkdir(parents=True, exist_ok=True)
        meta, arrays = [], {}
        for i, e in enumerate(self.entries):
            m = {k: v for k, v in asdict(e).items() if k not in ("embeddings", "thumbnail")}
            m["regions"] = sorted(e.embeddings)
            meta.append(m)
            for r, v in e.embeddings.items():
                arrays[f"{i}_{r}"] = v
            if e.thumbnail is not None:
                Image.fromarray(e.thumbnail).save(d / "thumbnails" / f"{e.key.replace('/', '_')}.jpg", quality=85)
        np.savez_compressed(d / "embeddings.npz", **arrays)
        (d / "entries.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> TrackIndex:
        from PIL import Image

        d = Path(directory)
        meta = json.loads((d / "entries.json").read_text(encoding="utf-8"))
        arrays = np.load(d / "embeddings.npz")
        entries = []
        for i, m in enumerate(meta):
            regions = m.pop("regions")
            thumb = d / "thumbnails" / f"{m['key'].replace('/', '_')}.jpg"
            e = TrackEntry(**m, embeddings={r: arrays[f"{i}_{r}"] for r in regions},
                           thumbnail=np.asarray(Image.open(thumb).convert("RGB")) if thumb.is_file() else None)
            entries.append(e)
        return cls(entries)
