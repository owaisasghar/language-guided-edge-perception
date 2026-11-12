"""Grounding DINO open-vocabulary detector (Hugging Face transformers backend).

Why this backend (see docs/milestone1.md for the full trade-off):
  * Weights are the official IDEA-Research checkpoints (Apache-2.0), pinned by commit.
  * Pure pip install: the official repo needs nvcc to compile its MSDeformAttn CUDA op,
    and this machine has no CUDA toolkit (the PyTorch wheels bundle only the runtime).
  * transformers can optionally load a pre-built deformable-attention kernel from the
    Hub (`kernels` package); Milestone 1 measures the plain PyTorch path.

Post-processing differs from `GroundingDinoProcessor.post_process_grounded_object_detection`
in one deliberate way: each box is assigned to exactly ONE query phrase, scored over that
phrase's tokens only ("max" mirrors the official demo, "mean" mirrors the official COCO
evaluation). The upstream helper scores against all tokens and returns a free-text token
label, which cannot tell you which phrase of "white truck . red car ." a box belongs to.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms

from lgep.perception.prompt import build_prompt, normalize_phrases, phrase_token_spans
from lgep.perception.types import Detection, DetectionResult, StageTimings
from lgep.utils.gpu import resolve_device, resolve_dtype, timed

log = logging.getLogger(__name__)


@dataclass
class GroundingDinoConfig:
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    revision: str | None = None  # pin a Hub commit for reproducibility
    device: str = "auto"
    dtype: str = "float32"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    phrase_score_reduction: str = "max"  # "max" | "mean"
    shortest_edge: int = 800
    longest_edge: int = 1333
    nms_iou: float | None = None  # class-aware NMS; None = raw DETR output (no NMS)
    max_detections: int = 100
    local_files_only: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> GroundingDinoConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        unknown = set(d) - set(known) - {"backend"}
        if unknown:
            raise ValueError(f"Unknown Grounding DINO config keys: {sorted(unknown)}")
        return cls(**known)


def _cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def assign_phrases(
    probs: torch.Tensor, spans: list[list[int]], reduction: str = "max"
) -> tuple[torch.Tensor, torch.Tensor]:
    """probs: (num_queries, num_text_positions) sigmoid scores.

    Returns (score, phrase_index) per query, where score is the best phrase score.
    """
    if not spans:
        raise ValueError("No phrase spans: prompt was empty or fully truncated")
    per_phrase = []
    for span in spans:
        tok = probs[:, span]
        per_phrase.append(tok.max(dim=1).values if reduction == "max" else tok.mean(dim=1))
    phrase_scores = torch.stack(per_phrase, dim=1)  # (Q, P)
    score, idx = phrase_scores.max(dim=1)
    return score, idx


class GroundingDinoDetector:
    """Load once, then call :meth:`detect` per image + phrase list."""

    def __init__(self, cfg: GroundingDinoConfig):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if cfg.phrase_score_reduction not in ("max", "mean"):
            raise ValueError("phrase_score_reduction must be 'max' or 'mean'")
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)
        if self.device.type == "cpu" and self.dtype != torch.float32:
            log.warning("%s is not supported for CPU inference here; using float32", cfg.dtype)
            self.dtype = torch.float32

        t0 = time.perf_counter()
        kw = {"revision": cfg.revision, "local_files_only": cfg.local_files_only}
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, **kw)
        self.processor.image_processor.size = {"shortest_edge": cfg.shortest_edge, "longest_edge": cfg.longest_edge}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model_id, dtype=self.dtype, **kw)
        self.model.to(self.device).eval()
        self.load_seconds = time.perf_counter() - t0

        tok = self.processor.tokenizer
        self._sep_id = tok.convert_tokens_to_ids(".")
        self._special_ids = set(tok.all_special_ids)
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        log.info("Loaded %s (%.1fM params, %s, %s) in %.1fs", cfg.model_id, self.num_parameters / 1e6,
                 self.device, str(self.dtype).replace("torch.", ""), self.load_seconds)

    @torch.inference_mode()
    def detect(self, image: Image.Image | np.ndarray, phrases: str | list[str]) -> DetectionResult:
        if isinstance(image, np.ndarray):  # assume RGB HxWx3 uint8
            image = Image.fromarray(image)
        image = image.convert("RGB")
        phrases = normalize_phrases(phrases)
        prompt = build_prompt(phrases)
        t: dict[str, float] = {}

        with timed(self.device, t, "pre"):
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        input_ids = inputs["input_ids"][0].tolist()
        spans = phrase_token_spans(input_ids, self._sep_id, self._special_ids)
        if len(spans) != len(phrases):
            raise ValueError(f"Prompt tokenised into {len(spans)} phrases, expected {len(phrases)} "
                             "(too long for the 256-token text encoder?)")

        with timed(self.device, t, "infer"):
            outputs = self.model(**inputs)

        with timed(self.device, t, "post"):
            detections = self._postprocess(outputs, input_ids, spans, phrases, image.size)

        h_in, w_in = inputs["pixel_values"].shape[-2:]
        return DetectionResult(
            phrases=phrases, prompt=prompt, image_size=image.size, input_size=(int(w_in), int(h_in)),
            detections=detections,
            timings=StageTimings(preprocess_ms=t["pre"], inference_ms=t["infer"], postprocess_ms=t["post"]),
        )

    def _postprocess(self, outputs, input_ids, spans, phrases, image_size) -> list[Detection]:
        cfg = self.cfg
        probs = outputs.logits[0].float().sigmoid()  # (Q, 256)
        boxes = outputs.pred_boxes[0].float()  # (Q, 4) normalised cxcywh

        score, idx = assign_phrases(probs, spans, cfg.phrase_score_reduction)
        keep = score > cfg.box_threshold
        score, idx, boxes, probs = score[keep], idx[keep], boxes[keep], probs[keep]

        w, h = image_size
        xyxy = _cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)

        if cfg.nms_iou is not None and len(score):
            k = batched_nms(xyxy, score, idx, cfg.nms_iou)
            score, idx, xyxy, probs = score[k], idx[k], xyxy[k], probs[k]

        order = score.argsort(descending=True)[: cfg.max_detections]
        score, idx, xyxy, probs = score[order].cpu(), idx[order].cpu(), xyxy[order].cpu(), probs[order].cpu()

        n_tok = len(input_ids)
        valid = torch.tensor([i not in self._special_ids and i != self._sep_id for i in input_ids])
        dets = []
        for s, p, b, pr in zip(score.tolist(), idx.tolist(), xyxy.tolist(), probs):
            mask = (pr[:n_tok] > cfg.text_threshold) & valid
            tokens = [input_ids[j] for j in mask.nonzero(as_tuple=True)[0].tolist()]
            dets.append(Detection(box_xyxy=tuple(b), score=s, phrase_id=p, phrase=phrases[p],
                                  token_label=self.processor.tokenizer.decode(tokens) if tokens else ""))
        return dets
