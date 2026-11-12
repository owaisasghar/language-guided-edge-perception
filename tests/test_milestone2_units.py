"""Unit tests for Milestone 2 building blocks that need no model weights."""

import pytest
import torch

from lgep.perception.factory import build_detector
from lgep.perception.grounding_dino import assign_phrases, phrase_scores, top_pairs
from lgep.utils.config import load_config


def test_phrase_scores_max_and_mean():
    probs = torch.tensor([[0.9, 0.1, 0.4, 0.0], [0.2, 0.3, 0.8, 0.6]])
    spans = [[0, 1], [2, 3]]
    assert torch.allclose(phrase_scores(probs, spans, "max"), torch.tensor([[0.9, 0.4], [0.3, 0.8]]))
    assert torch.allclose(phrase_scores(probs, spans, "mean"), torch.tensor([[0.5, 0.2], [0.25, 0.7]]))
    score, idx = assign_phrases(probs, spans, "max")
    assert idx.tolist() == [0, 1] and torch.allclose(score, torch.tensor([0.9, 0.8]))


def test_top_pairs_can_return_one_query_twice():
    scores = torch.tensor([[0.9, 0.8], [0.1, 0.7]])  # query 0 is good for both phrases
    s, q, p = top_pairs(scores, k=3)
    assert s.tolist() == pytest.approx([0.9, 0.8, 0.7])
    assert q.tolist() == [0, 0, 1] and p.tolist() == [0, 1, 1]
    assert len(top_pairs(scores, k=10)[0]) == 4  # k is capped at Q * P


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown detector backend"):
        build_detector({"backend": "nope"})


def test_config_extends_deep_merges(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nmodel: {x: 1, y: 2}\nlist: [1, 2]\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nmodel: {y: 3}\nlist: [9]\n")
    cfg = load_config(tmp_path / "child.yaml", ["model.x=5"])
    assert cfg == {"a": 1, "model": {"x": 5, "y": 3}, "list": [9]}


def test_config_extends_cycle_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_milestone2_configs_resolve():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    prompts = load_config(repo / "configs/milestone2_prompts.yaml")
    assert len(prompts["prompts"]) == 5  # inherited from Milestone 1
    assert prompts["models"]["yoloworld_m"]["conf"] == 0.25  # YAML merge key from the yoloworld_s anchor
    coco = load_config(repo / "configs/milestone2_coco_eval.yaml")
    assert coco["models"]["gdino_b"]["multi_label"] is True
    assert coco["models"]["gdino_b"]["model_id"].endswith("grounding-dino-base")


def test_val_rect_shape_matches_ultralytics_validation():
    from lgep.perception.yolo_world import val_rect_shape

    # BaseDataset.set_rectangle at batch 1: ceil(shape * imgsz / stride + 0.5) * stride
    assert val_rect_shape(480, 640, 640, 32) == (512, 672)  # landscape COCO image
    assert val_rect_shape(640, 427, 640, 32) == (672, 448)  # portrait
    assert val_rect_shape(640, 640, 640, 32) == (672, 672)  # square still gets the half-stride pad
    assert val_rect_shape(375, 500, 640, 32) == (512, 672)  # small images are scaled up first


def test_yolo_world_rejects_unknown_letterbox():
    from lgep.perception.yolo_world import YoloWorldConfig

    assert YoloWorldConfig().letterbox == "minimal"  # = YOLOWorld.predict()
    with pytest.raises(ValueError, match="letterbox"):
        YoloWorldConfig(letterbox="stretch")
