"""Integration tests with real Grounding DINO weights.

Skipped automatically when the weights are not in the local Hugging Face cache
(they are never downloaded by the test suite). CPU tests take ~10-20 s.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from lgep.perception.grounding_dino import GroundingDinoConfig, GroundingDinoDetector
from lgep.utils.gpu import cuda_usable

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
FOOD_TRUCK = REPO / "data/samples/milestone1/000000401862.jpg"


def _weights_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=REVISION), str)
    except Exception:
        return False


pytestmark = [
    pytest.mark.weights,
    pytest.mark.skipif(not _weights_cached(), reason="grounding-dino-tiny weights not in local HF cache"),
    pytest.mark.skipif(not FOOD_TRUCK.is_file(), reason="run scripts/download_samples.py first"),
]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


@pytest.fixture(scope="module")
def cpu_detector():
    cfg = GroundingDinoConfig(model_id=MODEL_ID, revision=REVISION, device="cpu", local_files_only=True)
    return GroundingDinoDetector(cfg)


def test_output_contract(cpu_detector):
    res = cpu_detector.detect(Image.open(FOOD_TRUCK), "white truck . red car")
    assert res.phrases == ["white truck", "red car"] and res.prompt == "white truck . red car ."
    w, h = res.image_size
    for d in res.detections:
        assert d.score > cpu_detector.cfg.box_threshold
        assert 0 <= d.box_xyxy[0] <= d.box_xyxy[2] <= w and 0 <= d.box_xyxy[1] <= d.box_xyxy[3] <= h
        assert d.phrase == res.phrases[d.phrase_id]
    assert res.timings.total_ms > 0


def test_blue_shirt_person_regression(cpu_detector):
    """Regression on observed behaviour (2026-11-11): the top box is the man in the blue T-shirt."""
    res = cpu_detector.detect(Image.open(FOOD_TRUCK), "person wearing a blue shirt")
    assert res.detections, "expected at least one detection"
    top = res.detections[0]
    assert _iou(top.box_xyxy, (195, 199, 274, 431)) > 0.7


def test_phrase_scores_never_exceed_upstream_scores(cpu_detector):
    """Our phrase-restricted score is a max over a subset of tokens, so every box we keep
    must also be kept by the upstream HF post-processing at the same threshold."""
    proc, model = cpu_detector.processor, cpu_detector.model
    image = Image.open(FOOD_TRUCK).convert("RGB")
    ours = cpu_detector.detect(image, "person . truck")
    inputs = proc(images=image, text="person . truck .", return_tensors="pt")
    with torch.inference_mode():
        out = model(**inputs)
    ref = proc.post_process_grounded_object_detection(
        out, inputs["input_ids"], threshold=cpu_detector.cfg.box_threshold,
        text_threshold=cpu_detector.cfg.text_threshold, target_sizes=[image.size[::-1]])[0]
    ref_boxes = ref["boxes"].tolist()
    assert len(ours.detections) <= len(ref_boxes)
    for d in ours.detections:
        assert max(_iou(d.box_xyxy, rb) for rb in ref_boxes) > 0.99


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_usable(), reason="CUDA not usable")
def test_gpu_matches_cpu(cpu_detector):
    cfg = GroundingDinoConfig(model_id=MODEL_ID, revision=REVISION, device="cuda", local_files_only=True)
    gpu = GroundingDinoDetector(cfg)
    image = Image.open(FOOD_TRUCK)
    a = cpu_detector.detect(image, "person wearing a blue shirt").detections
    b = gpu.detect(image, "person wearing a blue shirt").detections
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert _iou(x.box_xyxy, y.box_xyxy) > 0.98 and abs(x.score - y.score) < 1e-2
