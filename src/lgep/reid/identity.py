"""Global identities: link finished local tracks (tracklets) within and across cameras.

Each finished tracklet is compared with the existing global identities and either joins the best one or
starts a new one. Every decision is stored as an ``AssociationRecord`` with its evidence, and identities
can be merged or split afterwards. Local track IDs are never changed: a global identity is a separate
object that *refers to* local tracks (project rule).

Cues (each can be switched off for the RQ4 ablation):

* **appearance**: cosine similarity between the tracklet's descriptor and the identity's members
  (``max`` or ``mean`` over members); a match needs at least ``appearance_threshold``;
* **time and camera layout** (``use_time``): within one camera, a person cannot be two tracks at once and the
  gap since the identity's last track in that camera must be at most ``max_gap_same_camera_s``; across
  cameras, the transit time (start of the new tracklet minus end of the identity's track in the other
  camera) must lie in that camera pair's window (``CameraTopology``), and an identity cannot be in two
  non-overlapping cameras at the same time;
* **phrase** (``use_phrase``): only tracklets of the same query phrase (class) are linked;
* **attributes** (``use_attributes``): tracklets carrying attribute distributions (e.g. CLIP colour
  probabilities) are linked only if the distributions overlap: Bhattacharyya coefficient
  sum(sqrt(p * q)) >= ``attribute_min_overlap`` with every member that has attributes.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from lgep.reid.features import TrackletRecord

log = logging.getLogger(__name__)


@dataclass
class TransitWindow:
    min_s: float
    max_s: float
    overlapping: bool = False  # the cameras see a common area: the same object may be in both at once


class CameraTopology:
    """Allowed transit times between cameras. Pairs not listed use ``default`` (None = never linked)."""

    def __init__(self, pairs: dict[tuple[str, str], TransitWindow] | None = None,
                 default: TransitWindow | None = TransitWindow(0.0, 300.0)):
        self.pairs = dict(pairs or {})
        self.default = default

    @classmethod
    def from_dict(cls, d: dict | None) -> CameraTopology:
        d = d or {}
        default = d.get("default", {"min_s": 0.0, "max_s": 300.0})
        pairs = {}
        for p in d.get("pairs", []):
            w = TransitWindow(p["min_s"], p["max_s"], p.get("overlapping", False))
            pairs[(p["from"], p["to"])] = w
            if p.get("symmetric", True):
                pairs.setdefault((p["to"], p["from"]), w)
        return cls(pairs, None if default is None else TransitWindow(**default))

    def window(self, cam_from: str, cam_to: str) -> TransitWindow | None:
        return self.pairs.get((cam_from, cam_to), self.default)


@dataclass
class AssociationConfig:
    appearance_threshold: float = 0.80  # cosine similarity needed to link (CLIP features sit close together)
    member_reduction: str = "max"  # "max" | "mean" over the identity's member descriptors
    use_time: bool = True
    use_phrase: bool = True
    max_gap_same_camera_s: float = 30.0
    min_samples: int = 1  # tracklets with fewer appearance samples are not linked (they get their own ID)
    use_attributes: bool = False
    attribute_min_overlap: float = 0.5

    @classmethod
    def from_dict(cls, d: dict | None) -> AssociationConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown association config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.member_reduction not in ("max", "mean"):
            raise ValueError("member_reduction must be 'max' or 'mean'")
        return cfg


@dataclass
class Member:
    key: str
    camera_id: str
    phrase: str
    t_start: float
    t_end: float
    descriptor: np.ndarray | None
    attributes: np.ndarray | None = None


@dataclass
class GlobalIdentity:
    global_id: int
    members: list[Member] = field(default_factory=list)

    @property
    def cameras(self) -> list[str]:
        return sorted({m.camera_id for m in self.members})

    @property
    def t_start(self) -> float:
        return min(m.t_start for m in self.members)

    @property
    def t_end(self) -> float:
        return max(m.t_end for m in self.members)


@dataclass
class AssociationRecord:
    track_key: str
    global_id: int
    decision: str  # "matched" | "new" | "new_low_evidence" | "merged" | "split"
    score: float | None
    evidence: dict


class IdentityManager:
    def __init__(self, cfg: AssociationConfig | None = None, topology: CameraTopology | None = None):
        self.cfg = cfg or AssociationConfig()
        self.topology = topology or CameraTopology()
        self.identities: dict[int, GlobalIdentity] = {}
        self.records: list[AssociationRecord] = []
        self._of_track: dict[str, int] = {}
        self._ids = itertools.count(1)

    def global_id_of(self, track_key: str) -> int | None:
        return self._of_track.get(track_key)

    # ------------------------------------------------------------------------------------ feasibility
    def _feasible(self, m: Member, t: TrackletRecord) -> tuple[bool, str, float | None]:
        if m.camera_id == t.camera_id:
            if t.t_start <= m.t_end and m.t_start <= t.t_end:
                return False, "same camera, overlapping in time", None
            gap = t.t_start - m.t_end if t.t_start > m.t_end else m.t_start - t.t_end
            if gap > self.cfg.max_gap_same_camera_s:
                return False, f"same camera, gap {gap:.1f} s too long", gap
            return True, "same camera", gap
        if t.t_start >= m.t_end:
            w, transit = self.topology.window(m.camera_id, t.camera_id), t.t_start - m.t_end
        elif m.t_start >= t.t_end:
            w, transit = self.topology.window(t.camera_id, m.camera_id), m.t_start - t.t_end
        else:  # both cameras at the same time
            w = self.topology.window(m.camera_id, t.camera_id)
            return (w is not None and w.overlapping), "simultaneous in both cameras", 0.0
        if w is None:
            return False, "camera pair not connected", transit
        return w.min_s <= transit <= w.max_s, f"transit {transit:.1f} s in [{w.min_s}, {w.max_s}]", transit

    def _score(self, ident: GlobalIdentity, t: TrackletRecord, desc: np.ndarray) -> tuple[float | None, dict]:
        ev = {"global_id": ident.global_id}
        if self.cfg.use_phrase and any(m.phrase != t.phrase for m in ident.members):
            return None, {**ev, "rejected": "different phrase"}
        if self.cfg.use_time:
            for m in ident.members:
                ok, why, gap = self._feasible(m, t)
                if not ok:
                    return None, {**ev, "rejected": why}
        if self.cfg.use_attributes and t.attributes is not None:
            for m in ident.members:
                if m.attributes is not None:
                    overlap = float(np.sum(np.sqrt(np.clip(m.attributes, 0, None) * np.clip(t.attributes, 0, None))))
                    if overlap < self.cfg.attribute_min_overlap:
                        return None, {**ev, "rejected": f"attributes differ (overlap {overlap:.2f})"}
        sims = [float(desc @ m.descriptor) for m in ident.members if m.descriptor is not None]
        if not sims:
            return None, {**ev, "rejected": "no appearance"}
        s = max(sims) if self.cfg.member_reduction == "max" else float(np.mean(sims))
        return s, {**ev, "appearance": round(s, 4)}

    # ------------------------------------------------------------------------------------ association
    def assign(self, t: TrackletRecord) -> AssociationRecord:
        """Link a finished tracklet to the best feasible identity, or start a new one."""
        if t.key in self._of_track:
            raise ValueError(f"{t.key} was already assigned")
        desc = t.descriptor
        member = Member(t.key, t.camera_id, t.phrase, t.t_start, t.t_end, desc, t.attributes)
        if desc is None or t.n_samples < self.cfg.min_samples:
            return self._new(member, "new_low_evidence", None, {"samples": t.n_samples})
        best, best_s, considered = None, -1.0, []
        for ident in self.identities.values():
            s, ev = self._score(ident, t, desc)
            considered.append(ev)
            if s is not None and s > best_s:
                best, best_s = ident, s
        evidence = {"samples": t.n_samples, "candidates": len(considered),
                    "top": sorted((e for e in considered if "appearance" in e), key=lambda e: -e["appearance"])[:3]}
        if best is not None and best_s >= self.cfg.appearance_threshold:
            best.members.append(member)
            self._of_track[t.key] = best.global_id
            rec = AssociationRecord(t.key, best.global_id, "matched", round(best_s, 4), evidence)
            self.records.append(rec)
            return rec
        return self._new(member, "new", None if best is None else round(best_s, 4), evidence)

    def _new(self, member: Member, decision: str, score, evidence) -> AssociationRecord:
        gid = next(self._ids)
        self.identities[gid] = GlobalIdentity(gid, [member])
        self._of_track[member.key] = gid
        rec = AssociationRecord(member.key, gid, decision, score, evidence)
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------------------------------ revisions
    def merge(self, keep: int, absorb: int, reason: str = "manual") -> None:
        """Move all of ``absorb``'s tracklets into ``keep`` (e.g. after an operator's correction)."""
        a = self.identities.pop(absorb)
        self.identities[keep].members.extend(a.members)
        for m in a.members:
            self._of_track[m.key] = keep
            self.records.append(AssociationRecord(m.key, keep, "merged", None, {"from": absorb, "reason": reason}))

    def split(self, track_key: str, reason: str = "manual") -> int:
        """Detach one tracklet into a new identity; returns its new global ID."""
        gid = self._of_track[track_key]
        ident = self.identities[gid]
        m = next(x for x in ident.members if x.key == track_key)
        ident.members.remove(m)
        if not ident.members:
            del self.identities[gid]
        new = self._new(m, "split", None, {"from": gid, "reason": reason})
        return new.global_id
