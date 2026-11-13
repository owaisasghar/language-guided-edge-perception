"""Attribute verification and query scoring with CLIP (zero-shot).

Milestones 1-2 measured that neither detector grounds colour reliably ("green bus" fires on a blue bus).
Here colours are verified separately:

* **object colour** ("white truck"): softmax of CLIP similarities between the crop and the prompts
  "a photo of a {colour} {category}." over the colour vocabulary -> P(requested colour);
* **clothing colour** ("blue shirt", "black jeans"): the same on the matching body region of a person crop
  (upper / lower / head), with prompts "a photo of a person wearing a {colour} {item}.";
* **description similarity**: cosine similarity between the crop and "a photo of {description}.".

``QueryScorer.score`` combines detection score, description similarity and attribute probability with
weights tuned on RefCOCO+ val (configs/milestone6_query.yaml); each part can be switched off for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lgep.query.parser import COLOURS, StructuredQuery

# Fractions of a person box (top, bottom) used for each clothing region.
REGIONS = {"full": (0.0, 1.0), "upper": (0.15, 0.6), "lower": (0.5, 0.95), "head": (0.0, 0.25)}
LOGIT_SCALE = 100.0  # CLIP's learned temperature for ViT-B models (exp(4.6052))


def region_crop(crop: np.ndarray, region: str) -> np.ndarray:
    top, bottom = REGIONS[region]
    h = crop.shape[0]
    a, b = int(round(top * h)), max(int(round(bottom * h)), int(round(top * h)) + 1)
    return crop[a:b]


@dataclass
class ScoreWeights:
    detection: float = 1.0
    text: float = 1.0
    attributes: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> ScoreWeights:
        return cls(**(d or {}))


class QueryScorer:
    """Scores candidate crops (from a detector or a track index) against a structured query."""

    def __init__(self, embedder, weights: ScoreWeights | None = None, colours=COLOURS):
        self.embedder = embedder
        self.weights = weights or ScoreWeights()
        self.colours = tuple(colours)
        self._text_cache: dict[str, np.ndarray] = {}

    def texts(self, prompts: list[str]) -> np.ndarray:
        missing = [p for p in prompts if p not in self._text_cache]
        if missing:
            for p, e in zip(missing, self.embedder.embed_texts(missing)):
                self._text_cache[p] = e
        return np.stack([self._text_cache[p] for p in prompts])

    def colour_probs(self, image_embs: np.ndarray, template: str) -> np.ndarray:
        """(N, n_colours) softmax over the colour prompts; ``template`` contains ``{colour}``."""
        t = self.texts([template.format(colour=c) for c in self.colours])
        logits = LOGIT_SCALE * image_embs @ t.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def attribute_checks(self, q: StructuredQuery) -> list[tuple[str, str, str]]:
        """(region, prompt template, colour) for every colour attribute in the query."""
        noun = q.categories[0] if q.categories else "object"
        checks = [("full", f"a photo of a {{colour}} {noun}.", c) for c in q.object_colours]
        for cl in q.clothing:
            item = cl.item or "clothes"
            checks.append((cl.region, f"a photo of a person wearing {{colour}} {item}.", cl.colour))
        return checks

    def regions_needed(self, q: StructuredQuery) -> set[str]:
        return {"full"} | {r for r, _, _ in self.attribute_checks(q)}

    def embed_regions(self, crops: list[np.ndarray], regions: set[str]) -> dict[str, np.ndarray]:
        """CLIP embeddings of every crop for each of ``regions`` (``score`` needs "full" plus the regions of the
        query's clothing attributes: ``regions_needed``)."""
        out = {}
        for r in sorted(regions):
            out[r] = self.embedder.embed_images([c if r == "full" else region_crop(c, r) for c in crops])
        return out

    def score(self, q: StructuredQuery, region_embs: dict[str, np.ndarray], det_scores: np.ndarray | None = None) -> dict:
        """Scores for N candidates: ``text`` (cosine to the description), ``attributes`` (mean probability of
        the requested colours, 1 if none), ``total`` (weighted sum). ``region_embs`` from ``embed_regions``."""
        full = region_embs["full"]
        n = len(full)
        text = full @ self.texts([f"a photo of {q.free_text}."])[0] if q.free_text else np.zeros(n)
        checks = self.attribute_checks(q)
        if checks:
            probs = [self.colour_probs(region_embs[r], tpl)[:, self.colours.index(c)] for r, tpl, c in checks]
            attrs = np.mean(probs, axis=0)
        else:
            attrs = np.ones(n)
        det = np.zeros(n) if det_scores is None else np.asarray(det_scores, float)
        w = self.weights
        total = w.detection * det + w.text * text + w.attributes * attrs
        return {"text": text, "attributes": attrs, "detection": det, "total": total, "checks": checks}
