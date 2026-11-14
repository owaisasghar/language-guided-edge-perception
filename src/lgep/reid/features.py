"""Track-level appearance: choose a few good crops per local track and keep their embeddings.

Embedding every box on every frame is wasteful and noisy (occluded, truncated and tiny boxes give poor
features). Per track, a crop is sampled only if

* the box is at least ``min_height_px`` tall and its detection score is at least ``min_score``,
* it does not touch the image border (truncated objects),
* at most ``max_occlusion`` of its area is covered by other boxes in the same frame,
* at least ``every_n_frames`` frames have passed since this track's previous sample,

and at most ``max_samples`` embeddings are kept per track (evenly thinned when full). A track's descriptor
is the L2-normalised mean of its embeddings.

Mean-centring (``center``): every crop from one camera shares a large common component (same background,
same kind of object), which pushes all CLIP cosine similarities towards 1. Subtracting that camera's mean
embedding before comparing (a calibration from earlier footage, or online a running mean) separates
identities better (measured, docs/milestone5.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SamplingConfig:
    min_height_px: float = 64.0
    min_score: float = 0.5
    border_margin_px: float = 4.0
    max_occlusion: float = 0.3
    every_n_frames: int = 5
    max_samples: int = 16
    center: bool = True  # subtract the camera's mean embedding (running mean online) before comparing

    @classmethod
    def from_dict(cls, d: dict | None) -> SamplingConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown sampling config keys: {sorted(unknown)}")
        return cls(**d)


def occlusion(boxes: np.ndarray) -> np.ndarray:
    """For each box (N, 4 xyxy), the largest fraction of its area covered by any single other box."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    np.fill_diagonal(inter, 0)
    return (inter / area[:, None]).max(axis=1)


@dataclass
class TrackletRecord:
    """Everything known about one local track: its time span and appearance samples."""

    key: str  # "<camera_id>/<local_id>"
    camera_id: str
    local_id: int
    phrase: str
    first_frame: int
    last_frame: int
    t_start: float  # seconds (stream time or wall time; one clock per deployment)
    t_end: float
    frames_seen: int = 1
    sample_frames: list[int] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    center: np.ndarray | None = None  # camera mean embedding to subtract (set when the tracklet is closed)
    # Milestone 9: embeddings of a dedicated Re-ID model (e.g. FastReID) of the same crops, for identity matching
    # only; ``embeddings`` (CLIP) stay for search, attributes and the index. Empty: identity uses ``embeddings``.
    identity_embeddings: list[np.ndarray] = field(default_factory=list)
    identity_center: np.ndarray | None = None
    attributes: np.ndarray | None = None  # e.g. CLIP colour probabilities (for the attribute cue)
    best_crop: np.ndarray | None = None  # the tallest sampled crop (thumbnail, attribute checks)
    best_crop_frame: int | None = None

    @property
    def mean_embedding(self) -> np.ndarray | None:
        """L2-normalised mean of the raw (not centred) embeddings: the track's appearance in CLIP space,
        comparable with text embeddings."""
        if not self.embeddings:
            return None
        m = np.mean(self.embeddings, axis=0)
        return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)

    @property
    def descriptor(self) -> np.ndarray | None:
        """The appearance used for identity: mean-centred, L2-normalised mean of the identity model's
        embeddings if there are any, else of the CLIP embeddings."""
        if self.identity_embeddings:
            emb, center = self.identity_embeddings, self.identity_center
        else:
            emb, center = self.embeddings, self.center
        if not emb:
            return None
        e = np.asarray(emb, np.float64)
        if center is not None:
            e = e - center
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        m = e.mean(axis=0)
        return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)

    @property
    def n_samples(self) -> int:
        return len(self.embeddings)


class TrackFeatureBank:
    """Per-track records and appearance samples for one or many cameras."""

    def __init__(self, cfg: SamplingConfig | None = None):
        self.cfg = cfg or SamplingConfig()
        self.tracks: dict[str, TrackletRecord] = {}
        self._sum: dict[str, np.ndarray] = {}  # per camera: running sum / count of all embeddings
        self._count: dict[str, int] = {}
        self._isum: dict[str, np.ndarray] = {}  # the same for identity-model embeddings
        self._icount: dict[str, int] = {}

    def camera_mean(self, camera_id: str, identity: bool = False) -> np.ndarray | None:
        s, c = (self._isum, self._icount) if identity else (self._sum, self._count)
        n = c.get(camera_id, 0)
        return None if not n else (s[camera_id] / n).astype(np.float32)

    def observe(self, camera_id: str, frame_index: int, t: float, tracks, image_size: tuple[int, int]) -> list:
        """Update time spans for this frame's tracks; return the tracks whose crop should be embedded now."""
        w, h = image_size
        c = self.cfg
        boxes = np.array([tr.box_xyxy for tr in tracks], dtype=np.float64).reshape(-1, 4)
        occ = occlusion(boxes)
        want = []
        for tr, box, o in zip(tracks, boxes, occ):
            rec = self.tracks.get(tr.key)
            if rec is None:
                rec = self.tracks[tr.key] = TrackletRecord(tr.key, camera_id, tr.local_id, tr.phrase, frame_index,
                                                           frame_index, t, t)
            else:
                rec.last_frame, rec.t_end = frame_index, t
                rec.frames_seen += 1
            x1, y1, x2, y2 = box
            if (y2 - y1 < c.min_height_px or tr.score < c.min_score or o > c.max_occlusion
                    or x1 < c.border_margin_px or y1 < c.border_margin_px
                    or x2 > w - c.border_margin_px or y2 > h - c.border_margin_px):
                continue
            if rec.sample_frames and frame_index - rec.sample_frames[-1] < c.every_n_frames:
                continue
            want.append(tr)
        return want

    def add(self, key: str, frame_index: int, embedding: np.ndarray, crop_image: np.ndarray | None = None,
            identity: np.ndarray | None = None) -> None:
        """One sampled crop: its (CLIP) embedding and, optionally, an identity model's embedding of it."""
        rec = self.tracks[key]
        if crop_image is not None and (rec.best_crop is None or crop_image.shape[0] > rec.best_crop.shape[0]):
            rec.best_crop, rec.best_crop_frame = np.ascontiguousarray(crop_image), frame_index
        e = np.asarray(embedding, np.float32)
        rec.sample_frames.append(frame_index)
        rec.embeddings.append(e)
        self._sum[rec.camera_id] = self._sum.get(rec.camera_id, 0) + e.astype(np.float64)
        self._count[rec.camera_id] = self._count.get(rec.camera_id, 0) + 1
        if identity is not None:
            i = np.asarray(identity, np.float32)
            rec.identity_embeddings.append(i)
            self._isum[rec.camera_id] = self._isum.get(rec.camera_id, 0) + i.astype(np.float64)
            self._icount[rec.camera_id] = self._icount.get(rec.camera_id, 0) + 1
        if len(rec.embeddings) > self.cfg.max_samples:  # keep the spread over time: drop every second sample
            rec.embeddings = rec.embeddings[::2]
            rec.sample_frames = rec.sample_frames[::2]
            rec.identity_embeddings = rec.identity_embeddings[::2]  # same samples kept

    def pop(self, key: str) -> TrackletRecord | None:
        """Remove a finished track; with ``center``, attach the camera's current mean embedding."""
        rec = self.tracks.pop(key, None)
        if rec is not None and self.cfg.center:
            rec.center = self.camera_mean(rec.camera_id)
            if rec.identity_embeddings:
                rec.identity_center = self.camera_mean(rec.camera_id, identity=True)
        return rec


def crop(image: np.ndarray, box_xyxy, pad: float = 0.0) -> np.ndarray:
    """Crop (optionally padded by a fraction of the box size), clipped to the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    x1, y1 = max(0, int(round(x1 - px))), max(0, int(round(y1 - py)))
    x2, y2 = min(w, int(round(x2 + px))), min(h, int(round(y2 + py)))
    return image[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
