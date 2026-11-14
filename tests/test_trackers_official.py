"""The vendored OC-SORT and BoT-SORT must behave exactly like the official implementations (Milestone 9).

Each test runs the unmodified official code (checkouts in ``third_party/OC_SORT`` and ``third_party/BoT-SORT``
at the pinned commits; skipped if absent) and our copy on the same synthetic detections (the ByteTrack test's
scenarios: motion, occlusions around the track buffer, low-score boxes, false positives), and requires the same
track IDs and identical boxes on every frame.

Stand-ins for the official code's imports only: ``cython_bbox`` (no Python 3.12 wheels; a line-by-line port
of its loop), FilterPy's ``reshape_z`` (OC-SORT; not installed), ``fast_reid`` (BoT-SORT's encoder; the
features are given per frame instead), matplotlib if missing, and the ``np.float`` alias.
"""

from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from test_bytetrack import cython_bbox_reference, scenario

from lgep.tracking.botsort import GMC, BoTSORT
from lgep.tracking.ocsort import OCSort
from lgep.tracking.ocsort.kalmanfilter import reshape_z

REPO = Path(__file__).resolve().parents[1]
OCSORT_DIR = REPO / "third_party/OC_SORT"
BOTSORT_DIR = REPO / "third_party/BoT-SORT"
OCSORT_COMMIT = "8462e7e729a93ccd3bd995c0a79a890336cb3a0b"
BOTSORT_COMMIT = "251985436d6712aaf682aaaf5f71edb4987224bd"


class _Imports:
    """Temporarily install stand-in modules and packages; restore ``sys.modules`` afterwards."""

    def __init__(self, names):
        self.saved = {n: sys.modules.get(n) for n in names}
        self.added_float = not hasattr(np, "float")
        if self.added_float:
            np.float = float  # removed in NumPy 1.24; the official code still uses it

    def package(self, name, path):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]  # skip the package's __init__ (it imports detectors / Re-ID models)
        sys.modules[name] = pkg

    def restore(self):
        for n, m in self.saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
        if self.added_float:
            del np.float


@pytest.fixture(scope="module")
def official_ocsort():
    if not (OCSORT_DIR / "trackers/ocsort_tracker/ocsort.py").is_file():
        pytest.skip(f"official OC-SORT not checked out at {OCSORT_DIR} (see docs/milestone9.md)")
    names = ("trackers", "trackers.ocsort_tracker", "trackers.ocsort_tracker.ocsort",
             "trackers.ocsort_tracker.association", "trackers.ocsort_tracker.kalmanfilter",
             "filterpy", "filterpy.stats", "filterpy.common")
    imp = _Imports(names)
    imp.package("trackers", OCSORT_DIR / "trackers")
    imp.package("trackers.ocsort_tracker", OCSORT_DIR / "trackers/ocsort_tracker")
    sys.modules["filterpy"] = types.ModuleType("filterpy")
    sys.modules["filterpy.stats"] = types.SimpleNamespace(logpdf=None)
    sys.modules["filterpy.common"] = types.SimpleNamespace(pretty_str=None, reshape_z=reshape_z)
    try:
        yield importlib.import_module("trackers.ocsort_tracker.ocsort")
    finally:
        imp.restore()


@pytest.fixture(scope="module")
def official_botsort():
    if not (BOTSORT_DIR / "tracker/bot_sort.py").is_file():
        pytest.skip(f"official BoT-SORT not checked out at {BOTSORT_DIR} (see docs/milestone9.md)")
    names = ("tracker", "tracker.bot_sort", "tracker.matching", "tracker.gmc", "tracker.basetrack",
             "tracker.kalman_filter", "cython_bbox", "fast_reid", "fast_reid.fast_reid_interfece",
             "matplotlib", "matplotlib.pyplot")
    imp = _Imports(names)
    imp.package("tracker", BOTSORT_DIR / "tracker")
    sys.modules["cython_bbox"] = types.SimpleNamespace(bbox_overlaps=cython_bbox_reference)
    sys.modules["fast_reid"] = types.ModuleType("fast_reid")
    sys.modules["fast_reid.fast_reid_interfece"] = types.SimpleNamespace(FastReIDInterface=None)
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")
        sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
    try:
        yield importlib.import_module("tracker.bot_sort")
    finally:
        imp.restore()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("params", [dict(det_thresh=0.6, iou_threshold=0.3, use_byte=False),
                                    dict(det_thresh=0.4, iou_threshold=0.2, use_byte=True, asso_func="giou")])
def test_ocsort_matches_official(official_ocsort, seed, params):
    ref = official_ocsort.OCSort(max_age=30, min_hits=3, delta_t=3, inertia=0.2, **params)  # resets the count
    ours = OCSort(max_age=30, min_hits=3, delta_t=3, inertia=0.2, **params)
    n_out = 0
    for f, dets in enumerate(scenario(seed), start=1):
        expected = ref.update(dets.copy(), (720, 1280), (720, 1280))  # same image and input size: scale 1
        got = ours.update(dets[:, :4], dets[:, 4])
        assert [t.track_id for t in got] == [int(r[4]) for r in expected], f"frame {f}"
        for g, e in zip(got, expected):
            assert np.array_equal(g.tlbr, e[:4]), f"frame {f}, track {g.track_id}"
        n_out += len(got)
    assert n_out > 500 and ref.trackers  # the scenario produced tracks


def _botsort_args(**kw):
    a = dict(track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=30, match_thresh=0.8,
             proximity_thresh=0.5, appearance_thresh=0.25, with_reid=False, mot20=False)
    a.update(kw)
    return a


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("variant", ["iou", "camera_motion", "reid"])
def test_botsort_matches_official(official_botsort, seed, variant, monkeypatch):
    rng = np.random.default_rng(100 + seed)
    # appearance only counts where IoU >= 0.5 (proximity_thresh): it changes decisions only in a crowded scene
    frames, objects = scenario(seed, objects=80 if variant == "reid" else 12, with_ids=True)
    params = _botsort_args(with_reid=variant == "reid", **({"track_high_thresh": 0.5, "new_track_thresh": 0.6,
                                                             "match_thresh": 0.9} if seed % 2 else {}))
    # per frame: a small similarity warp (camera motion) and one appearance feature per detection
    warps = [np.eye(2, 3) if variant != "camera_motion" else
             np.array([[1 + rng.normal(0, 0.002), rng.normal(0, 0.002), rng.normal(0, 3)],
                       [rng.normal(0, 0.002), 1 + rng.normal(0, 0.002), rng.normal(0, 3)]]) for _ in frames]
    # each object has its own appearance (+ noise), false positives a random one: so appearance can override IoU
    looks = rng.normal(0, 1, (int(max(o.max(initial=0) for o in objects)) + 1, 16))
    feats = [np.where((o >= 0)[:, None], looks[np.maximum(o, 0)] + rng.normal(0, 0.15, (len(o), 16)),
                      rng.normal(0, 1, (len(o), 16))) for o in objects]

    class Encoder:  # stands in for FastReIDInterface: the official code asks it for the kept boxes' features
        def __init__(self):
            self.frame = 0

        def inference(self, img, dets):
            d = frames[self.frame - 1]
            keep = (d[:, 4] > params["track_low_thresh"])
            keep_hi = d[keep][:, 4] > params["track_high_thresh"]
            return feats[self.frame - 1][keep][keep_hi].copy()

    enc = Encoder()
    monkeypatch.setattr(official_botsort, "FastReIDInterface", lambda *a, **k: enc)
    ref = official_botsort.BoTSORT(Namespace(**params, cmc_method="none", name="synthetic", ablation=False,
                                             fast_reid_config=None, fast_reid_weights=None, device="cpu"),
                                   frame_rate=30)
    ours = BoTSORT(**params, frame_rate=30)
    plain = BoTSORT(**{**params, "with_reid": False}, frame_rate=30)  # neither appearance nor camera motion
    gmc_calls = iter(warps)
    monkeypatch.setattr(ref.gmc, "apply", lambda img, dets: next(gmc_calls))
    differs = 0
    for f, dets in enumerate(frames, start=1):
        enc.frame = f
        expected = ref.update(dets.copy(), None)
        got = ours.update(dets[:, :4], dets[:, 4], warp=warps[f - 1], features=feats[f - 1].copy())
        assert [t.track_id for t in got] == [t.track_id for t in expected], f"frame {f}"
        for g, e in zip(got, expected):
            assert np.array_equal(g.tlwh, e.tlwh), f"frame {f}, track {g.track_id}"
            assert g.score == e.score and g.start_frame == e.start_frame and g.is_activated == e.is_activated
        base = plain.update(dets[:, :4], dets[:, 4])
        differs += [(t.track_id, *t.tlwh) for t in base] != [(t.track_id, *t.tlwh) for t in got]
    # the scenario exercises the paths whose implementation differs (removed tracks, lost tracks), and the
    # variant's input (appearance, camera motion) actually changes the result
    assert len(ref.removed_stracks) > len(ours.removed_ids)
    assert ref.lost_stracks or len(ref.removed_stracks) > 10
    assert (differs > 0) == (variant != "iou")


def test_botsort_removed_ids_stay_bounded():
    tracker = BoTSORT()
    peak = 0
    for dets in scenario(5, frames=1500, objects=40):
        tracker.update(dets[:, :4], dets[:, 4])
        assert len(tracker.removed_ids) <= len(tracker.tracked_stracks) + len(tracker.lost_stracks)
        peak = max(peak, len(tracker.removed_ids))
    assert peak < 100 < tracker.next_id() - 1


def _textured(seed: int, h: int = 360, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    import cv2

    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (5, 5), 0)
    return img


def test_gmc_recovers_a_camera_shift():
    gmc = GMC("sparseOptFlow")
    img = _textured(0)
    assert np.array_equal(gmc.apply(img), np.eye(2, 3))  # first frame: no motion
    h = gmc.apply(np.roll(img, (4, -6), axis=(0, 1)))  # content moves 6 px left, 4 px down
    assert h[0, 2] == pytest.approx(-6, abs=0.5) and h[1, 2] == pytest.approx(4, abs=0.5)
    assert h[:, :2] == pytest.approx(np.eye(2), abs=0.01)


def test_gmc_survives_frames_without_texture():
    """The official code raises on a frame without corners (and on the frame after it); a camera must not stop."""
    gmc = GMC("sparseOptFlow")
    black = np.zeros((360, 640, 3), np.uint8)
    for img in (black, black, _textured(1), black, _textured(2)):
        h = gmc.apply(img)
        assert h.shape == (2, 3) and np.isfinite(h).all()
    assert np.array_equal(GMC("none").apply(black), np.eye(2, 3))
