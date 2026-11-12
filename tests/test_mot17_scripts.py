"""Helpers of scripts/detect_mot17.py and scripts/eval_mot17.py that decide what gets evaluated."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from detect_mot17 import cached, oracle  # noqa: E402
from eval_mot17 import load_detections  # noqa: E402

from lgep.evaluation.mot import load_sequence  # noqa: E402


def make_seq(tmp_path):
    d = tmp_path / "MOT17-99"
    (d / "gt").mkdir(parents=True)
    (d / "seqinfo.ini").write_text("[Sequence]\nseqLength=3\nimWidth=640\nimHeight=480\nframeRate=30\n")
    (d / "gt/gt.txt").write_text("\n".join([
        "2,1,10,20,30,40,1,1,1.0",    # pedestrian, scored            -> kept
        "1,2,50,60,20,50,1,1,0.5",    # pedestrian, scored            -> kept (sorted to frame 1)
        "1,3,90,90,10,10,0,1,1.0",    # pedestrian, not scored (0)    -> dropped
        "3,4,5,5,40,80,1,7,1.0",      # static person (distractor)    -> dropped
        "3,5,5,5,60,30,1,3,1.0",      # car                           -> dropped
    ]) + "\n")
    return load_sequence(d)


def test_oracle_keeps_only_scored_pedestrians(tmp_path):
    o = oracle(make_seq(tmp_path))
    assert o["frame"].tolist() == [1, 2]
    assert o["boxes"].tolist() == [[50, 60, 70, 110], [10, 20, 40, 60]]  # x1 y1 x2 y2
    assert o["scores"].tolist() == [1.0, 1.0]


def test_load_detections_groups_by_frame(tmp_path):
    p = tmp_path / "d.npz"
    np.savez(p, frame=np.array([1, 1, 3], np.int32), boxes=np.arange(12, dtype=np.float32).reshape(3, 4),
             scores=np.array([0.9, 0.2, 0.5], np.float32))
    d = load_detections(p)
    assert sorted(d) == [1, 3] and len(d[1][0]) == 2 and d[3][1].tolist() == [0.5]


def test_cache_must_match_model_prompt_and_all_sequences(tmp_path):
    cfg = {"data": {"sequences": ["A", "B"]}, "model": {"backend": "x", "conf": 0.05}, "prompt": "person"}
    meta = {"sequences_requested": ["A", "B"], "config": cfg}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "A.npz").write_bytes(b"")
    assert not cached(tmp_path, cfg)  # B.npz missing
    (tmp_path / "B.npz").write_bytes(b"")
    assert cached(tmp_path, cfg)
    assert not cached(tmp_path, {**cfg, "model": {"backend": "x", "conf": 0.1}})
    assert not cached(tmp_path, {**cfg, "data": {"sequences": ["A"]}})  # e.g. a one-sequence smoke test
    assert not cached(tmp_path, {**cfg, "prompt": "pedestrian"})
