"""The vendored ByteTrack must behave exactly like the official implementation.

``test_matches_official_bytetrack`` runs the unmodified official code (a checkout of
github.com/ifzhang/ByteTrack at the pinned commit in ``third_party/ByteTrack``; skipped if absent) and our
copy on the same synthetic detections, and requires identical track IDs and boxes on every frame.
Its only stand-ins are ``cython_bbox`` (no Python 3.12 wheels), replaced by a line-by-line port of the
Cython loop below, and the ``np.float`` alias the official code still uses.
"""

from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from lgep.tracking.bytetrack import BYTETracker
from lgep.tracking.bytetrack.matching import bbox_ious

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "third_party/ByteTrack"
OFFICIAL_COMMIT = "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99"


def cython_bbox_reference(boxes, query_boxes):
    """Line-by-line port of cython_bbox.bbox_overlaps (bbox.pyx), the IoU the official code uses."""
    n, k = boxes.shape[0], query_boxes.shape[0]
    overlaps = np.zeros((n, k), dtype=np.float64)
    for kk in range(k):
        box_area = (query_boxes[kk, 2] - query_boxes[kk, 0] + 1) * (query_boxes[kk, 3] - query_boxes[kk, 1] + 1)
        for nn in range(n):
            iw = min(boxes[nn, 2], query_boxes[kk, 2]) - max(boxes[nn, 0], query_boxes[kk, 0]) + 1
            if iw > 0:
                ih = min(boxes[nn, 3], query_boxes[kk, 3]) - max(boxes[nn, 1], query_boxes[kk, 1]) + 1
                if ih > 0:
                    ua = float((boxes[nn, 2] - boxes[nn, 0] + 1) * (boxes[nn, 3] - boxes[nn, 1] + 1)
                               + box_area - iw * ih)
                    overlaps[nn, kk] = iw * ih / ua
    return overlaps


def test_bbox_ious_equals_cython_bbox_definition():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, (60, 2))
    a = np.hstack([xy, xy + rng.uniform(1, 80, (60, 2))])
    b = np.vstack([a[:20] + rng.normal(0, 3, (20, 4)), a[:5],  # overlapping and identical boxes
                   [[0, 0, 9, 9], [10, 0, 19, 9], [9, 0, 18, 9]]])  # touching edges: +1 px convention
    assert np.array_equal(bbox_ious(a, b), cython_bbox_reference(a, b))
    edge = bbox_ious(np.array([[0.0, 0, 9, 9]]), np.array([[10.0, 0, 19, 9], [9.0, 0, 18, 9]]))
    assert edge[0, 0] == 0.0 and edge[0, 1] == pytest.approx(10 / 190)


def scenario(seed: int, frames: int = 400, objects: int = 12):
    """Synthetic detections: moving boxes, noise, occlusions of 5-60 frames (around ByteTrack's 30-frame
    buffer), low-score boxes, false positives, crossing paths and objects entering late."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform([0, 0], [1200, 600], (objects, 2))
    vel = rng.normal(0, 4, (objects, 2))
    size = rng.uniform([20, 50], [80, 200], (objects, 2))
    birth = rng.integers(1, frames // 2, objects)
    hidden = np.zeros(objects, int)
    out = []
    for f in range(1, frames + 1):
        pos += vel
        rows = []
        for i in range(objects):
            if f < birth[i]:
                continue
            if hidden[i] > 0:
                hidden[i] -= 1
                continue
            if rng.random() < 0.03:
                hidden[i] = rng.integers(5, 60)
            x1, y1 = pos[i] + rng.normal(0, 2, 2)
            w, h = size[i] * rng.uniform(0.9, 1.1, 2)
            rows.append([x1, y1, x1 + w, y1 + h, rng.choice([rng.uniform(0.6, 0.98), rng.uniform(0.1, 0.6)],
                                                              p=[0.8, 0.2])])
        for _ in range(rng.poisson(1.5)):  # false positives
            x1, y1 = rng.uniform(0, 1200), rng.uniform(0, 600)
            rows.append([x1, y1, x1 + rng.uniform(15, 60), y1 + rng.uniform(30, 120), rng.uniform(0.05, 0.8)])
        out.append(np.array(rows, dtype=np.float64).reshape(-1, 5))
    return out


@pytest.fixture(scope="module")
def official():
    if not (OFFICIAL / "yolox/tracker/byte_tracker.py").is_file():
        pytest.skip(f"official ByteTrack not checked out at {OFFICIAL} (see docs/milestone3.md)")
    names = ("yolox", "yolox.tracker", "yolox.tracker.byte_tracker", "yolox.tracker.basetrack",
             "yolox.tracker.kalman_filter", "yolox.tracker.matching", "cython_bbox")
    saved = {n: sys.modules.get(n) for n in names}
    added_float = not hasattr(np, "float")
    if added_float:
        np.float = float  # removed in NumPy 1.24; the official code still uses it
    sys.modules["cython_bbox"] = types.SimpleNamespace(bbox_overlaps=cython_bbox_reference)
    pkg = types.ModuleType("yolox")
    pkg.__path__ = [str(OFFICIAL / "yolox")]  # skip yolox/__init__.py (it imports the whole detector)
    sys.modules["yolox"] = pkg
    try:
        yield importlib.import_module("yolox.tracker.byte_tracker")
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
        if added_float:
            del np.float


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("params", [dict(track_thresh=0.6, match_thresh=0.9, mot20=False),
                                    dict(track_thresh=0.4, match_thresh=0.8, mot20=True)])
def test_matches_official_bytetrack(official, seed, params):
    args = Namespace(track_buffer=30, **params)
    official.BaseTrack._count = 0  # the official counter is process-wide
    ref = official.BYTETracker(args, frame_rate=30)
    ours = BYTETracker(frame_rate=30, track_buffer=30, **params)
    for f, dets in enumerate(scenario(seed), start=1):
        expected = ref.update(dets.copy(), (720, 1280), (720, 1280))  # same image and input size: scale 1
        got = ours.update(dets[:, :4], dets[:, 4])
        assert [t.track_id for t in got] == [t.track_id for t in expected], f"frame {f}"
        for g, e in zip(got, expected):
            assert np.array_equal(g.tlwh, e.tlwh), f"frame {f}, track {g.track_id}"
            assert g.score == e.score and g.start_frame == e.start_frame
    # The scenario must actually exercise the paths that differ in implementation.
    assert len(ref.removed_stracks) > len(ours.removed_ids)
    assert ref.lost_stracks or len(ref.removed_stracks) > 10


def test_removed_ids_stay_bounded():
    tracker = BYTETracker()
    peak = 0
    for dets in scenario(5, frames=1500, objects=40):
        tracker.update(dets[:, :4], dets[:, 4])
        assert len(tracker.removed_ids) <= len(tracker.tracked_stracks) + len(tracker.lost_stracks)
        peak = max(peak, len(tracker.removed_ids))
    assert peak < 100 < tracker.next_id() - 1  # many tracks were created; few removed IDs are kept
