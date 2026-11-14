"""Appearance embeddings for object crops (and, from Milestone 6, for text) with CLIP.

Backend: CLIP via Hugging Face ``transformers`` (already a core dependency). Default weights: LAION's
ViT-B/32 trained on LAION-2B (MIT licence, safetensors), pinned to a commit. CLIP is a general-purpose
embedding, not a Re-ID model. Dedicated Re-ID models (FastReID, Milestone 9: ``backend: fastreid``,
lgep.reid.fastreid_embedder) sit behind the same ``embed_images`` contract.

Pre-processing (``crop_mode``, on the GPU, one batch): ``stretch`` resizes the crop to the square input
(default), ``center_crop`` is CLIP's own (short side to 224, centred square crop), ``letterbox`` pads to a
square with CLIP's mean colour first. Letterboxing looked right on paper (nothing is cut off) but measured
worst on MOT17 person crops: a tall person fills only ~40 % of the padded square (docs/milestone5.md).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from lgep.utils.gpu import resolve_device, resolve_dtype, timed

log = logging.getLogger(__name__)
CROP_MODES = ("letterbox", "stretch", "center_crop")


@dataclass
class EmbedderConfig:
    backend: str = "hf_clip"
    model_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    revision: str | None = "1a25a446712ba5ee05982a381eed697ef9b435cf"
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 64  # crops per forward pass
    crop_mode: str = "stretch"
    local_files_only: bool = False
    # backend fastreid: model_id is the config file inside the FastReID checkout, weights the checkpoint
    weights: str | None = None
    fastreid_root: str = "third_party/fast-reid"

    @classmethod
    def from_dict(cls, d: dict | None) -> EmbedderConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown embedder config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.backend not in ("hf_clip", "fastreid"):
            raise ValueError(f"Unknown embedder backend {cfg.backend!r} (available: hf_clip, fastreid)")
        if cfg.backend == "fastreid" and not cfg.weights:
            raise ValueError("embedder backend fastreid needs 'weights' (a FastReID checkpoint)")
        if cfg.crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {cfg.crop_mode!r}")
        return cfg


class ClipEmbedder:
    """``embed_images(crops) -> (N, D)`` L2-normalised float32; ``embed_texts(texts)`` likewise.

    Thread-safe: one lock serialises calls, so the pipeline (crops) and API requests (query texts) can share
    one model on the GPU."""

    def __init__(self, cfg: EmbedderConfig | None = None):
        from transformers import CLIPModel, CLIPTokenizerFast

        self.cfg = cfg = cfg or EmbedderConfig()
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)
        t0 = time.perf_counter()
        kw = {"revision": cfg.revision, "local_files_only": cfg.local_files_only}
        self.model = CLIPModel.from_pretrained(cfg.model_id, use_safetensors=True, dtype=self.dtype, **kw)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(cfg.model_id, **kw)
        vc = self.model.config.vision_config
        self.input_size = int(vc.image_size)
        from transformers import CLIPImageProcessor

        proc = CLIPImageProcessor.from_pretrained(cfg.model_id, **kw)
        self._mean = torch.tensor(proc.image_mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(proc.image_std, device=self.device).view(1, 3, 1, 1)
        self.dim = int(self.model.config.projection_dim)
        self.num_parameters = sum(p.numel() for p in self.model.vision_model.parameters()) + sum(
            p.numel() for p in self.model.visual_projection.parameters())
        self.load_seconds = time.perf_counter() - t0
        self.last_timings: dict[str, float] = {}
        self._lock = threading.Lock()
        log.info("Loaded %s (vision %.1fM params, dim %d, %s, %s) in %.1fs", cfg.model_id, self.num_parameters / 1e6,
                 self.dim, self.device, cfg.dtype, self.load_seconds)

    def describe(self) -> dict:
        return {"backend": self.cfg.backend, "model_id": self.cfg.model_id, "revision": self.cfg.revision,
                "vision_parameters_millions": round(self.num_parameters / 1e6, 2), "dim": self.dim,
                "input_size": self.input_size, "crop_mode": self.cfg.crop_mode, "device": str(self.device),
                "dtype": self.cfg.dtype, "load_seconds": round(self.load_seconds, 2)}

    # ----------------------------------------------------------------------------------------- images
    def _prepare(self, crop: np.ndarray) -> torch.Tensor:
        """H x W x 3 uint8 RGB -> 3 x S x S float in [0, 1] on the device, per ``crop_mode``."""
        crop = np.require(crop, requirements=["C", "W"])  # contiguous and writable (PIL arrays are read-only)
        x = torch.from_numpy(crop).to(self.device).permute(2, 0, 1).float().div_(255)
        s = self.input_size
        _, h, w = x.shape
        mode = self.cfg.crop_mode
        if mode == "letterbox":
            side = max(h, w)
            canvas = self._mean[0].expand(3, side, side).clone()
            top, left = (side - h) // 2, (side - w) // 2
            canvas[:, top:top + h, left:left + w] = x
            x = canvas
        elif mode == "center_crop":  # CLIP's own pre-processing: short side to S, then a centred S x S crop
            scale = s / min(h, w)
            x = F.interpolate(x[None], size=(max(s, round(h * scale)), max(s, round(w * scale))), mode="bicubic",
                              antialias=True, align_corners=False)[0]
            _, h, w = x.shape
            top, left = (h - s) // 2, (w - s) // 2
            return x[:, top:top + s, left:left + s]
        return F.interpolate(x[None], size=(s, s), mode="bicubic", antialias=True, align_corners=False)[0]

    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        with self._lock:
            return self._embed_images(crops)

    @torch.inference_mode()
    def _embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        t: dict[str, float] = {"pre": 0.0, "infer": 0.0}
        out = []
        for i in range(0, len(crops), self.cfg.batch_size):
            part: dict[str, float] = {}
            with timed(self.device, part, "pre"):
                x = torch.stack([self._prepare(c) for c in crops[i:i + self.cfg.batch_size]])
                x = ((x - self._mean) / self._std).to(self.dtype)
            with timed(self.device, part, "infer"):
                f = self.model.get_image_features(pixel_values=x)
                f = getattr(f, "pooler_output", f)  # transformers 5 may return a model output object
                f = F.normalize(f.float(), dim=-1)
            out.append(f.cpu().numpy())
            t["pre"] += part["pre"]
            t["infer"] += part["infer"]
        self.last_timings = t
        return np.concatenate(out)

    # ------------------------------------------------------------------------------------------ texts
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        with self._lock:
            return self._embed_texts(texts)

    @torch.inference_mode()
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        tok = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)
        f = self.model.get_text_features(**tok)
        f = getattr(f, "pooler_output", f)
        return F.normalize(f.float(), dim=-1).cpu().numpy()


def build_embedder(cfg: dict | EmbedderConfig | None):
    """CLIP (``hf_clip``, default) or a dedicated Re-ID model (``fastreid``); both have ``embed_images``."""
    cfg = cfg if isinstance(cfg, EmbedderConfig) else EmbedderConfig.from_dict(cfg)
    if cfg.backend == "fastreid":
        from lgep.reid.fastreid_embedder import FastReIDEmbedder

        return FastReIDEmbedder(cfg)
    return ClipEmbedder(cfg)
