"""MOT evaluation helpers: ByteTrack's half split, GT/result files, and TrackEval scoring.

The TrackEval tests are skipped when TrackEval is not installed (`pip install -e ".[eval]"`).
"""

from __future__ import annotations

import importlib.util

import pytest

from lgep.evaluation.mot import load_sequence, split_frames, write_gt_split, write_mot_results

HAS_TRACKEVAL = importlib.util.find_spec("trackeval") is not None


@pytest.mark.parametrize("n", [600, 525, 837, 1050])  # MOT17-02, -09, -05, -04
def test_half_split_matches_bytetrack(n):
    # ByteTrack tools/convert_mot17_to_coco.py, 0-based i: train_half [0, n//2], val_half [n//2 + 1, n - 1]
    train, val = split_frames(n, "train_half"), split_frames(n, "val_half")
    assert list(train) == [i + 1 for i in range(0, n // 2 + 1)]
    assert list(val) == [i + 1 for i in range(n // 2 + 1, n)]
    assert list(split_frames(n, "full")) == list(train) + list(val)


def make_sequence(root, name="SEQ-01", length=6):
    d = root / name
    (d / "gt").mkdir(parents=True)
    (d / "seqinfo.ini").write_text(f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={length}\n"
                                   "imWidth=640\nimHeight=480\nimExt=.jpg\n")
    rows = []
    for f in range(1, length + 1):  # two pedestrians walking right, plus one static person (class 7)
        rows.append(f"{f},1,{10 + 5 * f},20,40,100,1,1,1.0")
        rows.append(f"{f},2,{300 - 5 * f},40,40,100,1,1,0.8")
        rows.append(f"{f},3,500,50,30,90,0,7,1.0")
    (d / "gt/gt.txt").write_text("\n".join(rows) + "\n")
    return load_sequence(d)


def test_write_gt_split_renumbers_frames(tmp_path):
    seq = make_sequence(tmp_path / "src")
    n = write_gt_split(seq, "val_half", tmp_path / "gt")  # frames 5..6 of 6
    lines = (tmp_path / "gt/SEQ-01/gt/gt.txt").read_text().splitlines()
    assert n == 2 and {int(x.split(",")[0]) for x in lines} == {1, 2}
    assert lines[0] == "1,1,35,20,40,100,1,1,1.0"  # original frame 5, otherwise unchanged


def test_write_mot_results_format(tmp_path):
    p = tmp_path / "r.txt"
    write_mot_results(p, [(1, 7, 10.04, 20.06, 30.0, 40.0, 0.876)])
    assert p.read_text() == "1,7,10.0,20.1,30.0,40.0,0.88,-1,-1,-1\n"


@pytest.mark.skipif(not HAS_TRACKEVAL, reason="TrackEval not installed")
def test_ground_truth_as_tracker_scores_perfectly(tmp_path):
    from lgep.evaluation.mot import evaluate_mot

    seq = make_sequence(tmp_path / "src", length=10)
    gt = tmp_path / "gt"
    write_gt_split(seq, "full", gt)
    rows = [(int(f), int(i), float(x), float(y), float(w), float(h), 1.0)
            for f, i, x, y, w, h, *_ in (r.split(",") for r in (gt / "SEQ-01/gt/gt.txt").read_text().split())
            if int(i) != 3]  # pedestrians only; the static person (class 7) is a distractor
    write_mot_results(tmp_path / "trk/perfect/data/SEQ-01.txt", rows)
    m = evaluate_mot(gt, tmp_path / "trk", "perfect", {"SEQ-01": 10})["combined"]
    assert m["HOTA"] == pytest.approx(1.0) and m["MOTA"] == pytest.approx(1.0) and m["IDF1"] == pytest.approx(1.0)
    assert m["IDSW"] == 0 and m["CLR_FP"] == 0 and m["CLR_FN"] == 0


@pytest.mark.skipif(not HAS_TRACKEVAL, reason="TrackEval not installed")
def test_identity_swap_is_counted(tmp_path):
    from lgep.evaluation.mot import evaluate_mot

    seq = make_sequence(tmp_path / "src", length=10)
    gt = tmp_path / "gt"
    write_gt_split(seq, "full", gt)
    rows = []
    for r in (gt / "SEQ-01/gt/gt.txt").read_text().split():
        f, i, x, y, w, h, *_ = r.split(",")
        if int(i) == 3:
            continue
        tid = int(i) if int(f) <= 5 else 3 - int(i)  # the two IDs are swapped from frame 6 on
        rows.append((int(f), tid, float(x), float(y), float(w), float(h), 1.0))
    write_mot_results(tmp_path / "trk/swap/data/SEQ-01.txt", rows)
    m = evaluate_mot(gt, tmp_path / "trk", "swap", {"SEQ-01": 10})["combined"]
    assert m["IDSW"] == 2 and m["MOTA"] == pytest.approx(1 - 2 / 20) and m["IDF1"] == pytest.approx(0.5)
