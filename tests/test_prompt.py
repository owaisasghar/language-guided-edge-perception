import pytest

from lgep.perception.prompt import build_prompt, normalize_phrases, phrase_token_spans

CLS, SEP, PAD, DOT = 101, 102, 0, 1012


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("white truck", ["white truck"]),
        ("White Truck. red  car .", ["white truck", "red car"]),
        (["Person", " bus ", "person"], ["person", "bus"]),  # de-duplicated, order kept
        (["a. b", "c"], ["a", "b", "c"]),  # dots inside list items split too
        ("  person   wearing a\tblue shirt  ", ["person wearing a blue shirt"]),
    ],
)
def test_normalize_phrases(raw, expected):
    assert normalize_phrases(raw) == expected


@pytest.mark.parametrize("raw", ["", " . . ", []])
def test_normalize_phrases_rejects_empty(raw):
    with pytest.raises(ValueError):
        normalize_phrases(raw)


def test_build_prompt_matches_grounding_dino_format():
    assert build_prompt(["white truck", "red car"]) == "white truck . red car ."


def test_phrase_token_spans_basic():
    # [CLS] white truck . red car . [SEP]
    ids = [CLS, 2317, 4744, DOT, 2417, 2482, DOT, SEP]
    assert phrase_token_spans(ids, DOT, {CLS, SEP, PAD}) == [[1, 2], [4, 5]]


def test_phrase_token_spans_unterminated_and_padding():
    ids = [CLS, 2317, DOT, 2417, SEP, PAD, PAD]
    assert phrase_token_spans(ids, DOT, {CLS, SEP, PAD}) == [[1], [3]]


def test_phrase_token_spans_ignores_repeated_separators():
    ids = [CLS, DOT, 2317, DOT, DOT, 2417, DOT, SEP]
    assert phrase_token_spans(ids, DOT, {CLS, SEP, PAD}) == [[2], [5]]
