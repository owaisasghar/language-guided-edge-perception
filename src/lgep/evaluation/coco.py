"""COCO bounding-box evaluation with pycocotools (the reference implementation).

Detections are evaluated as a ranked list (low score threshold, up to 300 per image), which is what
AP needs. Precision/recall at a fixed operating threshold are derived from the same COCOeval matching
(IoU 0.5, crowd regions ignored), so all numbers come from one protocol.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lgep.perception.types import DetectionResult

# Order of COCOeval.stats, as printed by COCOeval.summarize().
STAT_NAMES = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")


def load_coco(annotation_file: str | Path):
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(annotation_file))


def category_names(coco) -> tuple[list[str], list[int]]:
    """COCO category names (lower-case, as in the official Grounding DINO evaluation) and their ids."""
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    return [c["name"].lower() for c in cats], [c["id"] for c in cats]


def to_coco_results(image_id: int, result: DetectionResult, cat_ids: Sequence[int]) -> list[dict]:
    """Detections (phrase_id indexes ``cat_ids``) -> COCO result dicts with xywh boxes."""
    out = []
    for d in result.detections:
        x1, y1, x2, y2 = d.box_xyxy
        out.append({"image_id": image_id, "category_id": cat_ids[d.phrase_id],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(d.score, 5)})
    return out


def _coco_eval(coco_gt, results: list[dict], image_ids: Sequence[int]):
    from pycocotools.cocoeval import COCOeval

    if not results:
        raise ValueError("No detections to evaluate")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = list(image_ids)
    return ev


def evaluate_bbox(coco_gt, results: list[dict], image_ids: Sequence[int],
                  report_classes: Sequence[str] = ()) -> dict:
    """Standard COCO metrics, plus AP / AP50 for the named classes."""
    ev = _coco_eval(coco_gt, results, image_ids)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    metrics = {k: round(float(v), 4) for k, v in zip(STAT_NAMES, ev.stats)}

    names, cat_ids = category_names(coco_gt)
    precision = ev.eval["precision"]  # [T iou, R recall, K class, A area, M maxDets]
    per_class = {}
    for name in report_classes:
        k = ev.params.catIds.index(cat_ids[names.index(name)])
        p_all = precision[:, :, k, 0, -1]  # area=all, maxDets=100
        p50 = precision[0, :, k, 0, -1]
        per_class[name] = {"AP": round(float(p_all[p_all > -1].mean()), 4) if (p_all > -1).any() else None,
                           "AP50": round(float(p50[p50 > -1].mean()), 4) if (p50 > -1).any() else None}
    return {"metrics": metrics, "per_class": per_class, "summary_text": buf.getvalue()}


def precision_recall_at(coco_gt, results: list[dict], image_ids: Sequence[int], score_thr: float,
                        iou: float = 0.5) -> dict:
    """Precision / recall of the detections with score >= ``score_thr`` (all classes pooled).

    Uses COCOeval's greedy matching per image and class (crowd GT and detections matched to crowd GT
    are ignored), area = all, at most 100 detections per image and class.
    """
    kept = [r for r in results if r["score"] >= score_thr]
    n_gt = sum(1 for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=list(image_ids))) if not a["iscrowd"])
    if not kept:
        return {"score_thr": score_thr, "precision": None, "recall": 0.0, "tp": 0, "fp": 0, "num_gt": n_gt}
    ev = _coco_eval(coco_gt, kept, image_ids)
    ev.params.iouThrs = np.array([iou])
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.params.maxDets = [100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
    tp = fp = 0
    for e in ev.evalImgs:
        if e is None:
            continue
        matched = e["dtMatches"][0] > 0
        ignored = e["dtIgnore"][0].astype(bool)
        tp += int((matched & ~ignored).sum())
        fp += int((~matched & ~ignored).sum())
    return {"score_thr": score_thr, "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / n_gt, 4) if n_gt else None, "tp": tp, "fp": fp, "num_gt": n_gt}
