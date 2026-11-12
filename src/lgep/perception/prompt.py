"""Text-prompt utilities for grounding detectors.

Grounding DINO expects lowercase phrases separated by " . " and terminated by ".",
e.g. ``"white truck . red car ."``. This module normalises user phrases into that form
and maps each phrase back to its token positions so a box can be assigned to exactly
one phrase (instead of the free-text token label the model emits).

Full natural-language query parsing ("Find all white trucks across the cameras")
is Milestone 6 (src/lgep/query); this module only handles detector-ready phrases.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_WS = re.compile(r"\s+")


def normalize_phrases(phrases: str | Iterable[str]) -> list[str]:
    """Return clean, lowercase, de-duplicated phrases (order preserved).

    Accepts either one string (phrases separated by ".") or an iterable of strings.

    >>> normalize_phrases("White Truck. red  car .")
    ['white truck', 'red car']
    """
    if isinstance(phrases, str):
        parts: list[str] = phrases.split(".")
    else:
        parts = [p for item in phrases for p in str(item).split(".")]
    out: list[str] = []
    for p in parts:
        p = _WS.sub(" ", p).strip().lower()
        if p and p not in out:
            out.append(p)
    if not out:
        raise ValueError("Prompt contains no phrases after normalisation")
    return out


def build_prompt(phrases: Sequence[str]) -> str:
    """Join normalised phrases into Grounding DINO's caption format: "a . b ."."""
    return " . ".join(phrases) + " ."


def phrase_token_spans(
    input_ids: Sequence[int], separator_id: int, special_ids: Iterable[int]
) -> list[list[int]]:
    """Token positions belonging to each phrase, in prompt order.

    Positions holding special tokens ([CLS], [SEP], [PAD]) and the "." separator are
    excluded. A phrase that was truncated away by the tokenizer simply has no span, so
    callers must check ``len(spans) == len(phrases)``.
    """
    special = set(special_ids)
    spans: list[list[int]] = []
    current: list[int] = []
    for pos, tok in enumerate(input_ids):
        if tok in special:
            continue
        if tok == separator_id:
            if current:
                spans.append(current)
                current = []
            continue
        current.append(pos)
    if current:  # prompt not terminated with "."
        spans.append(current)
    return spans
