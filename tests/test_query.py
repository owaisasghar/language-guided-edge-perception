"""Language queries: parsing, attribute scoring, the track index."""

from __future__ import annotations

import time

import numpy as np
import pytest

from lgep.query import Clothing, parse_query
from lgep.query.attributes import QueryScorer, ScoreWeights, region_crop
from lgep.query.index import TrackEntry, TrackIndex


# --------------------------------------------------------------------------------------------- parser
@pytest.mark.parametrize("text, cats, colours, clothing, carried", [
    ("white truck", ["truck"], ["white"], [], []),
    ("Find all white trucks across the cameras", ["truck"], ["white"], [], []),
    ("a man in a blue shirt and black jeans", ["person"], [], [("shirt", "upper", "blue"), ("jeans", "lower", "black")], []),
    ("black shirt person holding umbrella", ["person"], [], [("shirt", "upper", "black")], ["umbrella"]),
    ("guy holding purp umbrella", ["person"], [], [], ["umbrella"]),  # the colour belongs to the umbrella
    ("woman in red", ["person"], [], [(None, "full", "red")], []),
    ("kid with green hat", ["person"], [], [("hat", "head", "green")], []),
    ("grey motorbikes", ["motorcycle"], ["gray"], [], []),
    ("silver pickup truck", ["truck"], ["silver"], [], []),
    ("people in the bus", ["person"], [], [], ["bus"]),
    ("any vehicle", ["car", "truck", "bus", "motorcycle"], [], [], []),
    ("large dark item next to car", [], [], [], ["car"]),  # unknown head: not the car
    ("navy jacket", ["person"], [], [("jacket", "upper", "blue")], []),  # clothing only: a person
    ("white shirt on the left", ["person"], [], [("shirt", "upper", "white")], []),
    ("orange shirt guy", ["person"], [], [("shirt", "upper", "orange")], []),  # a colour, not the fruit
    ("orange next to the apple", ["orange"], [], [], ["apple"]),  # the fruit
    ("baby elephant", ["elephant"], [], [], []),  # compound: the last noun is the head
    ("carrot bowl", ["bowl"], [], [], []),
    ("red bike", ["bicycle", "motorcycle"], ["red"], [], []),
    ("man in a red tie", ["person"], [], [("tie", "upper", "red")], []),
    ("man with glasses", ["person"], [], [], []),  # spectacles, not a wine glass
    ("brown bear", ["bear", "teddy bear"], ["brown"], [], []),
])
def test_parse_categories_colours_clothing(text, cats, colours, clothing, carried):
    q = parse_query(text)
    assert q.categories == cats
    assert q.object_colours == colours
    assert [(c.item, c.region, c.colour) for c in q.clothing] == clothing
    assert q.carried == carried


def test_parse_cameras_and_time():
    q = parse_query("red car on cam02 in the last 10 minutes")
    assert q.cameras == ["cam02"] and q.time.last_s == 600 and q.categories == ["car"]
    assert "cam02" not in q.free_text and "minutes" not in q.free_text
    q = parse_query("person between 2pm and 3:30pm on camera 3")
    assert (q.time.after, q.time.before, q.cameras) == ("14:00", "15:30", ["cam03"])
    assert parse_query("bus after 9:05").time.after == "09:05"
    assert parse_query("truck in the past hour").time.last_s == 3600


def test_numbers_that_are_not_times_do_not_break_parsing():
    q = parse_query("player between 27 and 31 in white shirt")  # jersey numbers, not clock times
    assert q.time is None and q.categories == ["person"] and q.clothing == [Clothing("shirt", "upper", "white")]
    assert parse_query("bus after 25:00").time is None


def test_detector_prompt_falls_back_to_the_description():
    assert parse_query("white truck").detector_prompt == "truck"
    assert parse_query("large dark item next to car").detector_prompt == "large dark item next to car"


# ------------------------------------------------------------------------------------------- scoring
class ColourEmbedder:
    """A fake CLIP: images are 'coloured' by their mean RGB; texts by the colour word they contain."""

    BASIS = {"red": 0, "green": 1, "blue": 2}

    def _vec(self, i):
        v = np.full(16, 0.01, np.float32)
        v[i] = 1.0
        return v / np.linalg.norm(v)

    def embed_images(self, crops):
        return np.stack([self._vec(int(np.argmax(c.reshape(-1, 3).mean(0)))) for c in crops])

    def embed_texts(self, texts):
        out = []
        for t in texts:
            hit = [i for w, i in self.BASIS.items() if w in t]
            out.append(self._vec(hit[0]) if hit else self._vec(15))
        return np.stack(out)


def person(top, bottom):
    img = np.zeros((100, 40, 3), np.uint8)
    img[:55, :, top] = 255
    img[55:, :, bottom] = 255
    return img


def test_region_crop_fractions():
    img = np.arange(100)[:, None, None].repeat(4, 1).repeat(3, 2)
    assert region_crop(img, "upper")[0, 0, 0] == 15 and region_crop(img, "upper").shape[0] == 45
    assert region_crop(img, "lower").shape[0] == 45


def test_clothing_colour_is_checked_on_the_right_region():
    s = QueryScorer(ColourEmbedder(), ScoreWeights(0, 0, 1), colours=("red", "green", "blue"))
    crops = [person(top=0, bottom=2), person(top=2, bottom=0)]  # red top / blue trousers, and the reverse
    q = parse_query("man in a red shirt")
    embs = s.embed_regions(crops, s.regions_needed(q))
    sc = s.score(q, embs)
    assert sc["attributes"][0] > 0.9 and sc["attributes"][1] < 0.1
    q2 = parse_query("man in blue pants")
    sc2 = s.score(q2, s.embed_regions(crops, s.regions_needed(q2)))
    assert sc2["attributes"][0] > 0.9 > sc2["attributes"][1]


def test_score_without_attributes_and_weights():
    s = QueryScorer(ColourEmbedder(), ScoreWeights(detection=2, text=0, attributes=1))
    q = parse_query("person")
    sc = s.score(q, s.embed_regions([person(0, 0)] * 2, {"full"}), det_scores=np.array([0.2, 0.5]))
    assert np.allclose(sc["attributes"], 1) and np.allclose(sc["total"], [1.4, 2.0])


# --------------------------------------------------------------------------------------------- index
def entry(key, cat, colour_idx, gid, wall_end, emb=ColourEmbedder()):
    cam, lid = key.split("/")
    return TrackEntry(key, cam, int(lid), gid, cat, 0.0, 5.0, wall_end - 5, wall_end, 3,
                      {"full": emb._vec(colour_idx)}, np.zeros((8, 4, 3), np.uint8))


def test_index_filters_ranks_and_reports_cameras(tmp_path):
    now = time.time()
    idx = TrackIndex([entry("cam01/1", "car", 0, 7, now - 30), entry("cam02/4", "car", 0, 7, now - 10),
                      entry("cam01/2", "car", 2, 8, now - 20), entry("cam01/3", "truck", 0, 9, now - 5),
                      entry("cam03/1", "car", 0, 10, now - 4000)])
    s = QueryScorer(ColourEmbedder(), ScoreWeights(0, 1, 1), colours=("red", "green", "blue"))
    hits = idx.search(parse_query("red car in the last 10 minutes"), s, now=now, one_per_identity=False)
    assert [h.entry.key for h in hits][:2] in (["cam01/1", "cam02/4"], ["cam02/4", "cam01/1"])
    assert hits[-1].entry.key == "cam01/2"  # the blue car ranks last; the truck and the old car are filtered
    assert {h.entry.key for h in hits} == {"cam01/1", "cam02/4", "cam01/2"}
    grouped = idx.search(parse_query("red car in the last 10 minutes"), s, now=now)
    assert [h.entry.global_id for h in grouped] == [7, 8]  # ID 7 once (its best track), then the blue car
    assert hits[0].identity_cameras == ["cam01", "cam02"]  # global ID 7 was seen by both cameras
    assert [h.entry.key for h in idx.search(parse_query("red car on cam02"), s, now=now)] == ["cam02/4"]
    idx.save(tmp_path / "index")
    back = TrackIndex.load(tmp_path / "index")
    assert len(back) == 5 and np.allclose(back.entries[0].embeddings["full"], idx.entries[0].embeddings["full"])
    assert back.entries[0].thumbnail.shape == (8, 4, 3) and back.entries[0].global_id == 7


def test_clothing_dataclass_is_hashable_and_comparable():
    assert Clothing("shirt", "upper", "blue") == Clothing("shirt", "upper", "blue")
