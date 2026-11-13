"""AI City (CityFlowV2) helpers: transit windows from ground truth, oracle track IDs, prediction files."""

from __future__ import annotations

import numpy as np
import pytest

from lgep.evaluation.aicity import iou_matrix, learn_transit_windows, oracle_track_ids, write_predictions

CLOCK = {"c001": {"offset_s": 0.0, "fps": 10.0}, "c002": {"offset_s": 5.0, "fps": 10.0},
         "c003": {"offset_s": 0.0, "fps": 10.0}}


def gt_rows(cam: int, vid: int, frames) -> list[list[int]]:
    return [[cam, vid, f, 10, 10, 20, 20] for f in frames]


def test_transit_windows_sequential_and_overlapping():
    gt = np.array(
        gt_rows(1, 7, range(1, 11))          # c001: t 0.0-0.9
        + gt_rows(2, 7, range(21, 31))       # c002: t 5 + 2.0-2.9 = 7.0-7.9 -> transit 6.1 s
        + gt_rows(1, 8, range(1, 11))
        + gt_rows(2, 8, range(51, 61))       # transit 5 + 5.0 - 0.9 = 9.1 s
        + gt_rows(1, 9, range(1, 51))        # c001 0.0-4.9 and c003 2.0-6.9: both at once
        + gt_rows(3, 9, range(21, 71)))
    w = learn_transit_windows(gt, CLOCK, margin_s=1.0)
    assert w[("c001", "c002")]["min_s"] == pytest.approx(5.1)
    assert w[("c001", "c002")]["max_s"] == pytest.approx(10.1)
    assert w[("c001", "c002")]["n"] == 2 and not w[("c001", "c002")]["overlapping"]
    assert ("c002", "c001") not in w  # never observed in that direction
    for pair in [("c001", "c003"), ("c003", "c001")]:
        assert w[pair]["overlapping"] and (w[pair]["min_s"], w[pair]["max_s"]) == (0.0, 1.0)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10]], float)
    iou = iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(1.0) and iou[0, 1] == pytest.approx(50 / 150) and iou[1].max() == 0


def test_oracle_track_ids_majority_vote():
    gt = np.array(gt_rows(1, 5, range(1, 6)) + [[1, 6, f, 200, 200, 20, 20] for f in range(1, 6)])
    rows = np.array([[f, 1, 10, 10, 20, 20] for f in range(1, 6)]          # on vehicle 5 throughout
                    + [[f, 2, 201, 201, 20, 20] for f in range(1, 4)]      # on vehicle 6
                    + [[f, 2, 10, 11, 20, 20] for f in range(4, 6)]        # then switches to 5: majority 6
                    + [[f, 3, 500, 500, 20, 20] for f in range(1, 6)], float)  # never on an annotated car
    assert oracle_track_ids(rows, gt) == {1: 5, 2: 6}


def test_write_predictions_keeps_multi_camera_ids_and_clips(tmp_path):
    rows_a = np.array([[1, 1, -5.0, 10.0, 30.0, 20.0, 0.9], [1, 2, 0, 0, 10, 10, 0.9]])
    rows_b = np.array([[3, 4, 95.0, 50.0, 20.0, 20.0, 0.8]])
    ids = {("c001", 1): 100, ("c002", 4): 100, ("c001", 2): 101}  # 101 is seen by one camera only
    n = write_predictions(tmp_path / "p.txt", [("c001", rows_a, (100, 100)), ("c002", rows_b, (100, 100))], ids)
    lines = (tmp_path / "p.txt").read_text().split("\n")
    assert n == 2
    assert lines[0] == "1 100 1 0 10 25 20 -1 -1"   # x -5 clipped to 0, width shrinks
    assert lines[1] == "2 100 3 95 50 4 20 -1 -1"   # right edge clipped to w - 1 = 99
