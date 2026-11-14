"""Dedicated Re-ID embeddings with FastReID (He et al., 2020; JDAI-CV/fast-reid, Apache-2.0) — Milestone 9.

Same contract as ``ClipEmbedder.embed_images``: crops (H x W x 3 uint8 RGB) -> (N, D) L2-normalised float32.
FastReID models are trained for re-identification on one domain: the VeRi-776 model for vehicles and the
MSMT17 model for people (official model zoo, SBS R50-ibn). They cannot embed text, so search keeps CLIP.

Uses the official code from a checkout (``fastreid_root``, default ``third_party/fast-reid``; pinned commit in
docs/milestone9.md) for the model definitions, and needs ``yacs`` and ``termcolor``. Weights are loaded with
``torch.load(weights_only=True)`` (tensors only; a downloaded pickle never runs code).

Pre-processing as FastReID's demo predictor: resize the crop to the model's input size with bicubic
interpolation (``cv2.INTER_CUBIC``, no aspect-ratio padding), 0-255 RGB; the model normalises with its
pixel mean and std and returns the feature after the BN neck.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lgep.utils.gpu import resolve_device, resolve_dtype, timed

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]

# The v0.1.1 model-zoo checkpoints predate a rename of the BN neck in the head: ``heads.bnneck.*`` is now
# ``heads.bottleneck.0.*``. Without the mapping the neck stays at its initial values and the features are wrong.
KEY_RENAMES = (("heads.bnneck.", "heads.bottleneck.0."),)


def remap_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for old, new in KEY_RENAMES:
            if k.startswith(old):
                k = new + k[len(old):]
        out[k] = v
    return out


class FastReIDEmbedder:
    """``embed_images(crops) -> (N, D)`` L2-normalised float32. Thread-safe (one lock)."""

    def __init__(self, cfg):
        self.cfg = cfg
        root = Path(cfg.fastreid_root)
        root = root if root.is_absolute() else REPO_ROOT / root
        if not (root / "fastreid").is_dir():
            raise FileNotFoundError(f"FastReID checkout not found at {root}: git clone https://github.com/JDAI-CV/fast-reid "
                                    f"{root} (see docs/milestone9.md)")
        weights = Path(cfg.weights or "")
        weights = weights if weights.is_absolute() else REPO_ROOT / weights
        if not weights.is_file():
            raise FileNotFoundError(f"FastReID weights not found: {weights} (see docs/milestone9.md)")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from fastreid.config import get_cfg
        from fastreid.modeling import build_model

        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)
        t0 = time.perf_counter()
        fcfg = get_cfg()
        fcfg.merge_from_file(str(root / cfg.model_id))
        fcfg.MODEL.BACKBONE.PRETRAIN = False  # the checkpoint has every weight; no ImageNet download
        fcfg.MODEL.DEVICE = str(self.device)
        fcfg.freeze()
        model = build_model(fcfg)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = remap_state_dict(state.get("model", state))
        # pixel_mean / pixel_std: saved in the checkpoint, non-persistent buffers from the config here; must agree
        for name in ("pixel_mean", "pixel_std"):
            if name in state and not torch.allclose(state.pop(name).flatten().float(),
                                                    getattr(model, name).flatten().float().cpu()):
                raise ValueError(f"{weights.name}: {name} differs from the config's")
        missing, unexpected = model.load_state_dict(state, strict=False)
        # the identity classifier (training only) and BN step counters are the only acceptable gaps
        bad = [k for k in missing if not (k == "heads.weight" or k.endswith("num_batches_tracked"))]
        bad += [k for k in unexpected if not (k.startswith("heads.classifier") or k.endswith("num_batches_tracked"))]
        if bad:
            raise ValueError(f"{weights.name} does not match {cfg.model_id}: {bad[:6]}")
        self.model = model.to(self.device).eval().to(self.dtype)
        self.input_hw = tuple(int(v) for v in fcfg.INPUT.SIZE_TEST)  # (H, W)
        self.dim = int(fcfg.MODEL.HEADS.EMBEDDING_DIM) or 2048
        self.num_parameters = sum(p.numel() for p in model.parameters())
        self.load_seconds = time.perf_counter() - t0
        self.last_timings: dict[str, float] = {}
        self._lock = threading.Lock()
        self.weights_name = weights.name
        log.info("Loaded FastReID %s (%s, %.1fM params, dim %d, input %dx%d, %s, %s) in %.1fs", cfg.model_id,
                 weights.name, self.num_parameters / 1e6, self.dim, self.input_hw[0], self.input_hw[1], self.device,
                 cfg.dtype, self.load_seconds)

    def describe(self) -> dict:
        return {"backend": "fastreid", "display_name": f"FastReID {Path(self.weights_name).stem}",
                "model_id": self.cfg.model_id, "weights": self.weights_name,
                "parameters_millions": round(self.num_parameters / 1e6, 2), "dim": self.dim,
                "input_size": list(self.input_hw), "crop_mode": "stretch", "device": str(self.device),
                "dtype": self.cfg.dtype, "load_seconds": round(self.load_seconds, 2)}

    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        import cv2

        h, w = self.input_hw
        return cv2.resize(np.ascontiguousarray(crop), (w, h), interpolation=cv2.INTER_CUBIC)

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
                x = np.stack([self._prepare(c) for c in crops[i:i + self.cfg.batch_size]])
                x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).to(self.dtype)  # 0-255 RGB, NCHW
            with timed(self.device, part, "infer"):
                f = F.normalize(self.model(x).float(), dim=-1)
            out.append(f.cpu().numpy())
            t["pre"] += part["pre"]
            t["infer"] += part["infer"]
        self.last_timings = t
        return np.concatenate(out)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("FastReID embeds images only; use CLIP for text queries")
