"""YOLO-World open-vocabulary detector (Ultralytics backend; AGPL-3.0, optional ``yolo`` extra).

The pipeline is assembled from Ultralytics' own building blocks (``LetterBox``, ``WorldModel``,
``non_max_suppression``, ``scale_boxes``) instead of calling ``YOLOWorld.predict``, for three reasons:
  * the same per-stage timing (pre / forward / post, CUDA-synchronised) as the Grounding DINO backend;
  * ``multi_label`` NMS, which the official COCO validation uses but ``predict`` does not expose;
  * the CLIP text encoder is loaded once and kept; ``YOLOWorld.set_classes`` reloads it from disk on
    every call.
``tests/test_yolo_world.py`` checks that the output equals ``YOLOWorld.predict`` with the same settings.

Unlike Grounding DINO, text is encoded only when the phrase list changes ("prompt-then-detect"); the
per-image forward pass never sees the text encoder. Encoding time is recorded in ``text_encode_ms``.
The encoder runs on ``text_device`` (default CPU, FP32). That is what Ultralytics does in normal use
(``set_classes`` runs before the model moves to the GPU), and it keeps ~290 MiB of CLIP weights off the
GPU. On CUDA the ``clip`` package keeps FP16 weights, which shifts scores by ~1e-4.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lgep.perception import ultralytics_env
from lgep.perception.prompt import normalize_phrases
from lgep.perception.types import Detection, DetectionResult, StageTimings
from lgep.utils.download import sha256
from lgep.utils.gpu import resolve_device, resolve_dtype, timed

log = logging.getLogger(__name__)
SETUP_HINT = "Run: python scripts/setup_yolo_world.py"
# How the image is fitted into the network input (all keep the aspect ratio, pad with grey 114):
#   minimal  - long side = imgsz, pad to the next stride multiple (640x480 -> 640x480); YOLOWorld.predict()
#   val_rect - long side = imgsz, pad to a stride multiple + half a stride (640x480 -> 672x512); Ultralytics
#              model.val() (rect=True, pad=0.5), which produced the published COCO AP; batch 1 here
#   square   - fixed imgsz x imgsz (640x480 -> 640x640); a static-shape exported engine (ONNX/TensorRT)
LETTERBOX_MODES = ("minimal", "val_rect", "square")


def val_rect_shape(h: int, w: int, imgsz: int, stride: int, pad: float = 0.5) -> tuple[int, int]:
    """Input (height, width) that Ultralytics validation gives an h x w image at batch size 1."""
    ar = h / w
    shape = (ar, 1.0) if ar < 1 else (1.0, 1 / ar) if ar > 1 else (1.0, 1.0)
    return tuple(int(math.ceil(s * imgsz / stride + pad)) * stride for s in shape)


@dataclass
class YoloWorldConfig:
    weights: str = "models/yolo_world/yolov8l-worldv2.pt"
    device: str = "auto"
    dtype: str = "float32"
    imgsz: int = 640  # long side of the resized image; the padding depends on `letterbox`
    conf: float = 0.25  # Ultralytics predict default; its COCO validation uses 0.001
    iou: float = 0.5  # Ultralytics predict default; its COCO validation uses 0.7
    max_det: int = 300
    multi_label: bool = False  # True = one box may carry several classes (official COCO validation)
    agnostic_nms: bool = False
    letterbox: str = "minimal"  # see LETTERBOX_MODES
    text_device: str = "cpu"  # where CLIP encodes phrase lists: "cpu" (FP32) or "cuda" (FP16, uses VRAM)
    ultralytics_dir: str = "models/ultralytics"  # project-local Ultralytics settings + CLIP cache

    def __post_init__(self):
        if self.letterbox not in LETTERBOX_MODES:
            raise ValueError(f"letterbox must be one of {LETTERBOX_MODES}, got {self.letterbox!r}")

    @classmethod
    def from_dict(cls, d: dict) -> YoloWorldConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        unknown = set(d) - set(known) - {"backend"}
        if unknown:
            raise ValueError(f"Unknown YOLO-World config keys: {sorted(unknown)}")
        return cls(**known)


class YoloWorldDetector:
    """Load once, then call :meth:`detect` per image + phrase list."""

    def __init__(self, cfg: YoloWorldConfig, base_dir: str | Path | None = None):
        base = Path(base_dir) if base_dir else Path.cwd()
        self.cfg = cfg
        self.weights = (base / cfg.weights).resolve()
        config_dir = (base / cfg.ultralytics_dir).resolve()
        clip_file = ultralytics_env.weights_dir(config_dir) / "clip" / ultralytics_env.CLIP_FILE
        for f in (self.weights, clip_file):
            if not f.is_file():
                raise FileNotFoundError(f"{f} not found. {SETUP_HINT}")

        ultralytics_env.configure(config_dir)
        from ultralytics import YOLOWorld
        from ultralytics.data.augment import LetterBox
        from ultralytics.nn.text_model import build_text_model
        from ultralytics.utils import WEIGHTS_DIR, nms, ops

        if Path(WEIGHTS_DIR).resolve() != ultralytics_env.weights_dir(config_dir):
            raise RuntimeError(f"Ultralytics uses weights dir {WEIGHTS_DIR}, expected "
                               f"{ultralytics_env.weights_dir(config_dir)}. {SETUP_HINT}")

        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)
        if self.dtype not in (torch.float32, torch.float16):
            raise ValueError("YOLO-World supports float32 or float16")
        if self.device.type == "cpu" and self.dtype != torch.float32:
            log.warning("%s is not supported for CPU inference here; using float32", cfg.dtype)
            self.dtype = torch.float32

        t0 = time.perf_counter()
        self.weights_sha256 = sha256(self.weights)
        model = YOLOWorld(str(self.weights), verbose=False).model  # WorldModel (nn.Module)
        # Fuse Conv+BN on the CPU, then move: Ultralytics' PyTorch backend does the same. Fusing on the GPU
        # gives ~1e-5 weight differences that cuDNN's TF32 convolutions amplify to ~1 px in box outputs.
        model = model.fuse(verbose=False).to(self.device)
        model = model.half() if self.dtype == torch.float16 else model.float()
        self.model = model.eval()
        self.num_parameters = sum(p.numel() for p in self.model.parameters())  # before CLIP is attached
        self.stride = int(self.model.stride.max())
        self.end2end = bool(getattr(self.model, "end2end", False))
        self._LetterBox = LetterBox
        if cfg.letterbox == "minimal":  # identical to YOLOWorld.predict()
            self._letterbox = LetterBox(cfg.imgsz, auto=True, stride=self.stride)
        elif cfg.letterbox == "square":
            self._letterbox = LetterBox((cfg.imgsz, cfg.imgsz), auto=False, stride=self.stride)
        self._nms = nms.non_max_suppression
        self._scale_boxes = ops.scale_boxes

        # Load the CLIP text encoder once, here, so load time includes it and phrase changes measure
        # encoding only. The same text model WorldModel.get_text_pe builds.
        self.text_device = resolve_device(cfg.text_device)
        self._text_model = build_text_model("clip:ViT-B/32", device=self.text_device)
        self.text_encoder_parameters = sum(p.numel() for p in self._text_model.parameters())
        self._phrases: list[str] | None = None
        self.text_encode_ms: list[float] = []
        self.set_phrases(["object"])
        self.text_encode_ms.clear()
        self.load_seconds = time.perf_counter() - t0
        log.info("Loaded %s (%.1fM params + CLIP %.1fM, %s, %s) in %.1fs", self.weights.name,
                 self.num_parameters / 1e6, self.text_encoder_parameters / 1e6, self.device,
                 str(self.dtype).replace("torch.", ""), self.load_seconds)

    def describe(self) -> dict:
        return {"backend": "ultralytics_yolo_world", "model_id": self.weights.stem,
                "revision": f"sha256:{self.weights_sha256}",
                "parameters_millions": round(self.num_parameters / 1e6, 2),
                "text_encoder": f"CLIP ViT-B/32 on {self.text_device.type}",
                "text_encoder_parameters_millions": round(self.text_encoder_parameters / 1e6, 2),
                "load_seconds": round(self.load_seconds, 2), "device": str(self.device),
                "dtype": str(self.dtype).replace("torch.", ""), "imgsz": self.cfg.imgsz, "conf": self.cfg.conf,
                "iou": self.cfg.iou, "multi_label": self.cfg.multi_label, "max_det": self.cfg.max_det,
                "letterbox": self.cfg.letterbox}

    def _fit(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple | None]:
        """Letterbox an RGB image; returns it and the ``ratio_pad`` that maps boxes back (None = derive it)."""
        if self.cfg.letterbox != "val_rect":
            return self._letterbox(image=rgb), None
        # Ultralytics validation: BaseDataset.load_image resizes the long side to imgsz (ceil, INTER_LINEAR,
        # also upscales), then LetterBox(scaleup=False) pads to the rect batch shape.
        import cv2

        h0, w0 = rgb.shape[:2]
        imgsz = self.cfg.imgsz
        r = imgsz / max(h0, w0)
        if r != 1:
            size = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        shape = val_rect_shape(h0, w0, imgsz, self.stride)
        out = self._LetterBox(shape, auto=False, scaleup=False)(image=rgb)
        pad = (round((shape[1] - w) / 2 - 0.1), round((shape[0] - h) / 2 - 0.1))  # as LetterBox: left, top
        return out, ((h / h0, w / w0), pad)

    @torch.inference_mode()
    def set_phrases(self, phrases: list[str]) -> None:
        """Encode the vocabulary with CLIP; a no-op when it is unchanged."""
        if phrases == self._phrases:
            return
        t: dict[str, float] = {}
        with timed(self.device, t, "text"):  # mirrors WorldModel.set_classes / get_text_pe (batches of 80)
            tokens = self._text_model.tokenize(list(phrases))
            feats = torch.cat([self._text_model.encode_text(b) for b in tokens.split(80)])
            self.model.txt_feats = feats.reshape(1, len(phrases), -1).to(self.device)
            self.model.model[-1].nc = len(phrases)
        self.model.names = dict(enumerate(phrases))
        self._phrases = list(phrases)
        self.text_encode_ms.append(t["text"])

    @torch.inference_mode()
    def detect(self, image: Image.Image | np.ndarray, phrases: str | list[str]) -> DetectionResult:
        if isinstance(image, np.ndarray):  # assume RGB HxWx3 uint8
            image = Image.fromarray(image)
        image = image.convert("RGB")
        phrases = normalize_phrases(phrases)
        self.set_phrases(phrases)
        w, h = image.size
        cfg = self.cfg
        t: dict[str, float] = {}

        with timed(self.device, t, "pre"):
            # Letterboxing RGB directly equals Ultralytics' BGR letterbox followed by its BGR->RGB flip.
            lb, ratio_pad = self._fit(np.asarray(image))
            x = torch.from_numpy(lb).to(self.device).permute(2, 0, 1).unsqueeze(0).contiguous()
            x = (x.half() if self.dtype == torch.float16 else x.float()).div_(255)

        with timed(self.device, t, "infer"):
            preds = self.model(x)

        with timed(self.device, t, "post"):
            out = self._nms(preds, cfg.conf, cfg.iou, None, cfg.agnostic_nms, multi_label=cfg.multi_label,
                            max_det=cfg.max_det, nc=0, end2end=self.end2end)[0]
            if len(out):
                out[:, :4] = self._scale_boxes(x.shape[2:], out[:, :4], (h, w), ratio_pad=ratio_pad)
            out = out.float().cpu()

        detections = [Detection(box_xyxy=tuple(r[:4]), score=r[4], phrase_id=int(r[5]), phrase=phrases[int(r[5])])
                      for r in out[:, :6].tolist()]
        return DetectionResult(
            phrases=phrases, prompt=" . ".join(phrases), image_size=(w, h),
            input_size=(int(x.shape[3]), int(x.shape[2])), detections=detections,
            timings=StageTimings(preprocess_ms=t["pre"], inference_ms=t["infer"], postprocess_ms=t["post"]),
        )
