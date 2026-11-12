"""YOLO-World backend: our pipeline must reproduce Ultralytics' own ``predict()``.

Skipped when the `yolo` extra is not installed or scripts/setup_yolo_world.py has not been run.
Never downloads anything.
"""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from lgep.perception import ultralytics_env

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "models/yolo_world/yolov8s-worldv2.pt"
CLIP = ultralytics_env.weights_dir(REPO / "models/ultralytics") / "clip" / ultralytics_env.CLIP_FILE
IMAGE = REPO / "data/samples/milestone1/000000210273.jpg"  # dense traffic scene
PHRASES = ["person", "bus", "car", "truck"]

pytestmark = [
    pytest.mark.weights,
    pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None, reason="`yolo` extra not installed"),
    pytest.mark.skipif(not (WEIGHTS.is_file() and CLIP.is_file()), reason="run scripts/setup_yolo_world.py"),
    pytest.mark.skipif(not IMAGE.is_file(), reason="run scripts/download_samples.py first"),
]


@pytest.fixture(scope="module")
def detector():
    from lgep.perception.yolo_world import YoloWorldConfig, YoloWorldDetector

    return YoloWorldDetector(YoloWorldConfig(weights=str(WEIGHTS.relative_to(REPO)), device="auto"), base_dir=REPO)


def test_matches_ultralytics_predict(detector):
    from ultralytics import YOLOWorld

    ref = YOLOWorld(str(WEIGHTS), verbose=False)
    ref.set_classes(PHRASES)
    image = Image.open(IMAGE)
    expected = ref.predict(image, conf=0.25, iou=0.5, imgsz=640, max_det=300, device=str(detector.device),
                           verbose=False)[0].boxes
    got = detector.detect(image, PHRASES).detections
    assert len(got) == len(expected) > 5
    for d, box, score, cls in zip(got, expected.xyxy.tolist(), expected.conf.tolist(), expected.cls.tolist()):
        assert d.phrase_id == int(cls)
        assert d.score == pytest.approx(score, abs=1e-4)
        assert d.box_xyxy == pytest.approx(box, abs=0.05)


def test_text_encoded_once_per_phrase_list(detector):
    image = Image.open(IMAGE)
    detector.detect(image, ["bus"])
    n = len(detector.text_encode_ms)
    detector.detect(image, ["bus"])
    assert len(detector.text_encode_ms) == n
    detector.detect(image, ["car"])
    assert len(detector.text_encode_ms) == n + 1


def test_multi_label_keeps_at_least_single_label_boxes(detector):
    image = Image.open(IMAGE)
    single = detector.detect(image, PHRASES).detections
    detector.cfg.multi_label = True
    try:
        multi = detector.detect(image, PHRASES).detections
    finally:
        detector.cfg.multi_label = False
    assert len(multi) >= len(single)
    assert {d.phrase for d in multi} <= set(PHRASES)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@pytest.mark.parametrize("mode", ["val_rect", "square"])
def test_letterbox_modes_map_boxes_back_to_the_image(detector, mode):
    """Padding changes the network input, not the coordinate frame: confident boxes must stay in place.

    A wrong scale or pad offset in ``ratio_pad`` would shift every box by up to 80 px or scale it by ~5 %.
    """
    from lgep.perception.yolo_world import YoloWorldConfig, YoloWorldDetector, val_rect_shape

    image = Image.open(IMAGE)
    ref = detector.detect(image, PHRASES)
    other = YoloWorldDetector(YoloWorldConfig(weights=str(WEIGHTS.relative_to(REPO)), letterbox=mode),
                              base_dir=REPO)
    got = other.detect(image, PHRASES)
    w, h = image.size
    expected = val_rect_shape(h, w, 640, other.stride)[::-1] if mode == "val_rect" else (640, 640)
    assert got.input_size == expected != ref.input_size
    confident = [d for d in ref.detections if d.score > 0.5]
    assert len(confident) >= 3
    for d in confident:
        same = [g.box_xyxy for g in got.detections if g.phrase_id == d.phrase_id]
        assert max((_iou(d.box_xyxy, g) for g in same), default=0.0) > 0.85, d
