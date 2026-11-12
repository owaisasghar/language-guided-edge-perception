import torch
from PIL import Image

from lgep.perception.grounding_dino import _cxcywh_to_xyxy, assign_phrases
from lgep.perception.types import Detection, DetectionResult
from lgep.utils.visualization import draw_detections, label_text


def test_cxcywh_to_xyxy():
    b = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    assert torch.allclose(_cxcywh_to_xyxy(b), torch.tensor([[0.4, 0.3, 0.6, 0.7]]))


def test_assign_phrases_picks_best_phrase_per_query():
    # 3 queries x 6 text positions; phrase A = tokens [1, 2], phrase B = tokens [4]
    probs = torch.tensor([
        [0.0, 0.9, 0.1, 0.0, 0.2, 0.0],  # -> A (max 0.9)
        [0.0, 0.1, 0.1, 0.0, 0.7, 0.0],  # -> B
        [0.95, 0.1, 0.1, 0.0, 0.1, 0.0],  # high prob on [CLS] must NOT count
    ])
    score, idx = assign_phrases(probs, [[1, 2], [4]], "max")
    assert idx[:2].tolist() == [0, 1]
    assert torch.allclose(score[:2], torch.tensor([0.9, 0.7]))
    assert score[2] < 0.2  # CLS position ignored


def test_assign_phrases_mean_reduction():
    probs = torch.tensor([[0.0, 0.9, 0.1, 0.0, 0.6, 0.0]])
    score, idx = assign_phrases(probs, [[1, 2], [4]], "mean")
    assert idx.item() == 1 and abs(score.item() - 0.6) < 1e-6  # mean(A)=0.5 < B=0.6


def _result(dets):
    return DetectionResult(phrases=["red car"], prompt="red car .", image_size=(200, 100),
                           input_size=(200, 100), detections=dets)


def test_label_text_shows_token_label_only_when_different():
    d1 = Detection((0, 0, 1, 1), 0.64, 0, "red car", "white truck red car")
    d2 = Detection((0, 0, 1, 1), 0.5, 0, "red car", "red car")
    assert label_text(d1) == "red car 0.64 [white truck red car]"
    assert label_text(d2) == "red car 0.50"
    assert label_text(d1, show_tokens=False) == "red car 0.64"


def test_draw_detections_draws_box_and_header():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    out = draw_detections(img, _result([Detection((50, 40, 150, 90), 0.8, 0, "red car")]), header="hdr")
    assert out.size[0] == 200 and out.size[1] > 100  # header strip added on top
    strip = out.size[1] - 100
    assert out.getpixel((50, strip + 70)) != (0, 0, 0)  # left box edge painted
    assert img.getpixel((50, 70)) == (0, 0, 0)  # input not modified
