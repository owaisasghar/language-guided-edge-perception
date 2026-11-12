from pathlib import Path

import pytest

from lgep.utils.config import apply_overrides, load_config

REPO = Path(__file__).resolve().parents[1]


def test_overrides_parse_yaml_types_and_do_not_mutate():
    cfg = {"device": "cuda", "model": {"box_threshold": 0.35, "nms_iou": 0.5}}
    out = apply_overrides(cfg, ["device=cpu", "model.box_threshold=0.3", "model.nms_iou=null", "new.key=[1, 2]"])
    assert out == {"device": "cpu", "model": {"box_threshold": 0.3, "nms_iou": None}, "new": {"key": [1, 2]}}
    assert cfg["device"] == "cuda"  # original untouched


def test_override_value_may_contain_equals():
    assert apply_overrides({}, ["prompt=a=b"]) == {"prompt": "a=b"}


@pytest.mark.parametrize("bad", ["no_equals", "model.box_threshold.x=1"])
def test_bad_overrides_raise(bad):
    with pytest.raises(ValueError):
        apply_overrides({"model": {"box_threshold": 0.35}}, [bad])


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_milestone1_config_is_consistent():
    """The shipped config must have 5 prompts with unique ids and a valid detector section."""
    from lgep.perception.grounding_dino import GroundingDinoConfig

    cfg = load_config(REPO / "configs/milestone1_grounding_dino.yaml")
    ids = [p["id"] for p in cfg["prompts"]]
    assert len(ids) == 5 and len(set(ids)) == 5
    GroundingDinoConfig.from_dict({**cfg["model"], "device": cfg["device"]})  # raises on unknown keys
    man = load_config(REPO / cfg["data"]["manifest"])
    assert len(man["images"]) == 5
    assert all(len(it["sha256"]) == 64 for it in man["images"])
