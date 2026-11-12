"""Build a text-guided detector from a config block (``backend`` selects the implementation)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from lgep.perception.types import DetectionResult

BACKENDS = ("hf_transformers", "ultralytics_yolo_world")


class Detector(Protocol):
    device: torch.device
    num_parameters: int
    load_seconds: float

    def detect(self, image: Image.Image | np.ndarray, phrases: str | list[str]) -> DetectionResult: ...

    def describe(self) -> dict: ...


def build_detector(model_cfg: dict, device: str = "auto", base_dir: str | Path | None = None) -> Detector:
    """``base_dir`` resolves relative paths in the config (YOLO-World weights); defaults to the cwd."""
    backend = model_cfg.get("backend", "hf_transformers")
    cfg = {**model_cfg, "device": device}
    if backend == "hf_transformers":  # Grounding DINO
        from lgep.perception.grounding_dino import GroundingDinoConfig, GroundingDinoDetector

        return GroundingDinoDetector(GroundingDinoConfig.from_dict(cfg))
    if backend == "ultralytics_yolo_world":
        from lgep.perception.yolo_world import YoloWorldConfig, YoloWorldDetector

        return YoloWorldDetector(YoloWorldConfig.from_dict(cfg), base_dir=base_dir)
    raise ValueError(f"Unknown detector backend {backend!r}; choose from {BACKENDS}")
