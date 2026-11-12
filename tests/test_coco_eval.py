"""COCO evaluation helpers on a tiny synthetic ground truth with known answers."""

import contextlib
import io

import pytest

pytest.importorskip("pycocotools")

from lgep.evaluation.coco import (  # noqa: E402
    category_names,
    evaluate_bbox,
    precision_recall_at,
    to_coco_results,
)
from lgep.perception.types import Detection, DetectionResult  # noqa: E402


@pytest.fixture()
def gt():
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = {
        "images": [{"id": 1, "width": 200, "height": 200}, {"id": 2, "width": 200, "height": 200}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 50, 40], "area": 2000, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [100, 50, 40, 80], "area": 3200, "iscrowd": 0},
        ],
        "categories": [{"id": 3, "name": "Car"}, {"id": 1, "name": "person"}],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _det(image_id, cat_id, bbox, score):
    return {"image_id": image_id, "category_id": cat_id, "bbox": bbox, "score": score}


def test_category_names_sorted_by_id_and_lowercased(gt):
    assert category_names(gt) == (["person", "car"], [1, 3])


def test_to_coco_results_converts_xyxy_and_phrase_ids():
    r = DetectionResult(phrases=["person", "car"], prompt="", image_size=(200, 200), input_size=(200, 200),
                        detections=[Detection(box_xyxy=(10, 10, 60, 50), score=0.9, phrase_id=1, phrase="car")])
    assert to_coco_results(7, r, [1, 3]) == [_det(7, 3, [10, 10, 50, 40], 0.9)]


def test_perfect_detections_give_ap_one(gt):
    res = [_det(1, 3, [10, 10, 50, 40], 0.9), _det(2, 1, [100, 50, 40, 80], 0.8)]
    out = evaluate_bbox(gt, res, [1, 2], report_classes=["car"])
    assert out["metrics"]["AP"] == pytest.approx(1.0)
    assert out["per_class"]["car"]["AP"] == pytest.approx(1.0)


def test_precision_recall_at_threshold(gt):
    res = [_det(1, 3, [10, 10, 50, 40], 0.9),  # TP
           _det(2, 1, [100, 50, 40, 80], 0.3),  # TP, low score
           _det(2, 3, [0, 150, 30, 30], 0.6)]   # FP (no car in image 2)
    hi = precision_recall_at(gt, res, [1, 2], score_thr=0.5)
    assert (hi["tp"], hi["fp"], hi["precision"], hi["recall"]) == (1, 1, 0.5, 0.5)
    lo = precision_recall_at(gt, res, [1, 2], score_thr=0.1)
    assert (lo["tp"], lo["fp"], lo["recall"]) == (2, 1, 1.0)
    none = precision_recall_at(gt, res, [1, 2], score_thr=0.95)
    assert none["tp"] == 0 and none["recall"] == 0.0
