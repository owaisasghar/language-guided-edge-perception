"""Per-camera tracking: persistent local IDs, occlusion handling, per-phrase association, filters."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lgep.perception.types import Detection, DetectionResult, StageTimings
from lgep.tracking import CameraTracker, IdAllocator, TrackerConfig


def box(x, y, w=40.0, h=100.0):
    return [x, y, x + w, y + h]


def frame(*boxes_scores, phrase="person"):
    b = np.array([bs[0] for bs in boxes_scores], dtype=float).reshape(-1, 4)
    s = np.array([bs[1] for bs in boxes_scores], dtype=float)
    return {phrase: (b, s)}


def ids(result):
    return sorted(t.local_id for t in result.tracks)


def test_id_allocator_never_reuses_ids_across_restarts(tmp_path):
    state = tmp_path / "cam.json"
    a = IdAllocator(state, block=3)
    assert [a() for _ in range(4)] == [1, 2, 3, 4]
    assert json.loads(state.read_text())["reserved_until"] == 6  # second block reserved at ID 4
    b = IdAllocator(state, block=3)  # "restart": continues after everything that may have been used
    assert b() == 7
    assert IdAllocator()() == 1  # no state file: restarts at 1


def test_track_keeps_its_id_through_a_short_occlusion_and_not_a_long_one():
    t = CameraTracker("cam01")
    walk = [box(100 + 5 * f, 200) for f in range(200)]
    first = [t.update(frame((walk[f], 0.9))) for f in range(20)]
    assert ids(first[-1]) == [1]
    for f in range(20, 30):  # 10 frames without the detection (< 30-frame buffer)
        assert ids(t.update(frame())) == []
    assert ids(t.update(frame((walk[30], 0.9)))) == [1]  # re-found by motion, same ID
    for _ in range(40):  # 40 frames > buffer: the track is removed
        t.update(frame())
    t.update(frame((walk[71], 0.9)))
    assert ids(t.update(frame((walk[72], 0.9)))) == [2]  # a new track, a new ID; 1 is never reused


def test_cameras_have_independent_id_spaces():
    a, b = CameraTracker("cam01"), CameraTracker("cam02")
    ra = rb = None
    for f in range(3):
        ra = a.update(frame((box(10 + f, 10), 0.9), (box(300 + f, 10), 0.9)))
        rb = b.update(frame((box(50 + f, 50), 0.9)))
    assert ids(ra) == [1, 2] and ids(rb) == [1]
    assert {tr.key for tr in ra.tracks} == {"cam01/1", "cam01/2"}  # local keys, not global identities


def test_phrases_are_associated_separately_but_share_the_id_space():
    t = CameraTracker("cam01")
    same_box = box(100, 100)
    r = None
    for _ in range(3):
        r = t.update({**frame((same_box, 0.9), phrase="person"), **frame((same_box, 0.9), phrase="bicycle")})
    assert sorted((tr.phrase, tr.local_id) for tr in r.tracks) == [("bicycle", 2), ("person", 1)]
    r = t.update(frame((same_box, 0.9), phrase="person"))  # "bicycle" left the query
    assert t.phrases == ["person"] and [tr.phrase for tr in r.tracks] == ["person"]


def test_start_frame_is_in_camera_frames_for_a_phrase_added_later():
    t = CameraTracker("cam01")
    for _ in range(5):
        t.update(frame())
    r = None
    for f in range(3):
        r = t.update({**frame(), **frame((box(10 + f, 10), 0.9), phrase="car")})
    assert [tr.start_frame for tr in r.tracks] == [6]  # tracker created on camera frame 6


def test_output_filters_do_not_change_tracker_state():
    cfg = TrackerConfig(min_box_area=5000, max_aspect_ratio=1.6)
    t = CameraTracker("cam01", cfg)
    small, wide = box(0, 0, 40, 100), box(500, 0, 200, 100)  # area 4000; w/h 2.0
    r = None
    for _ in range(3):
        r = t.update(frame((small, 0.9), (wide, 0.9), (box(800, 0, 60, 120), 0.9)))
    assert len(r.tracks) == 1 and r.tracks[0].box_xyxy[0] == pytest.approx(800, abs=1)
    assert len(t._trackers["person"].tracked_stracks) == 3  # filtered boxes are still tracked


def test_reset_keeps_ids_increasing():
    t = CameraTracker("cam01")
    for _ in range(2):
        t.update(frame((box(0, 0), 0.9)))
    t.reset()
    t.update(frame((box(0, 0), 0.9)))
    assert ids(t.update(frame((box(0, 0), 0.9)))) == [2]


def test_update_from_detector_result():
    dets = [Detection(box_xyxy=tuple(box(10, 10)), score=0.9, phrase_id=0, phrase="person"),
            Detection(box_xyxy=tuple(box(300, 10)), score=0.8, phrase_id=1, phrase="car")]
    res = DetectionResult(phrases=["person", "car", "bus"], prompt="person . car . bus", image_size=(640, 480),
                          input_size=(640, 480), detections=dets, timings=StageTimings(0, 0, 0))
    t = CameraTracker("cam01")
    r = t.update_from_result(res)
    assert sorted(t.phrases) == ["bus", "car", "person"] and len(r.tracks) == 2


def test_config_and_camera_id_validation():
    with pytest.raises(ValueError, match="Unknown tracker config"):
        TrackerConfig.from_dict({"track_treshold": 0.5})
    with pytest.raises(ValueError):
        CameraTracker("site/cam01")



def test_aspect_ratio_filter_only_for_people():
    from lgep.tracking.tracker import CameraTracker, TrackerConfig

    tr = CameraTracker("cam01", TrackerConfig(max_aspect_ratio=1.6))
    wide = np.array([[100.0, 100.0, 300.0, 180.0]])  # 200 x 80: w/h 2.5, like a car seen from the side
    out = None
    for _ in range(3):
        out = tr.update({"person": (wide, np.array([0.95])), "car": (wide + 400, np.array([0.95]))})
    assert [t.phrase for t in out.tracks] == ["car"]  # the wide "person" box is dropped, the car kept
    every = CameraTracker("cam01", TrackerConfig(max_aspect_ratio=1.6, aspect_filter_phrases=None))
    for _ in range(3):
        out = every.update({"car": (wide, np.array([0.95]))})
    assert out.tracks == []



def test_phrase_groups_track_label_flicker_as_one_object():
    from lgep.perception.types import Detection, DetectionResult
    from lgep.tracking.tracker import CameraTracker, TrackerConfig

    def frame(label):
        d = Detection(box_xyxy=(100.0, 100.0, 300.0, 180.0), score=0.9, phrase_id=["car", "truck"].index(label),
                      phrase=label)
        return DetectionResult(phrases=["car", "truck"], prompt="car . truck", image_size=(640, 480),
                               input_size=(640, 480), detections=[d])

    grouped = CameraTracker("cam01", TrackerConfig(phrase_groups={"vehicle": ["car", "truck"]}))
    split = CameraTracker("cam01", TrackerConfig())
    ids_g, ids_s = set(), set()
    for label in ["car", "car", "truck", "car", "truck", "truck"]:  # the detector flips its label
        ids_g |= {(t.phrase, t.local_id) for t in grouped.update_from_result(frame(label)).tracks}
        ids_s |= {(t.phrase, t.local_id) for t in split.update_from_result(frame(label)).tracks}
    assert ids_g == {("vehicle", 1)}  # one track for the whole time
    assert len(ids_s) > 1  # per-label trackers split it
