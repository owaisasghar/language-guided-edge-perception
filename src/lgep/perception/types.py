"""Detector-agnostic result types shared by every perception backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Detection:
    """One detected object in original-image pixel coordinates."""

    box_xyxy: tuple[float, float, float, float]
    score: float
    phrase_id: int  # index into DetectionResult.phrases
    phrase: str  # the query phrase this box was assigned to
    token_label: str = ""  # raw tokens above text_threshold (diagnostic, model-specific)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box_xyxy"] = [round(v, 2) for v in self.box_xyxy]
        d["score"] = round(self.score, 4)
        return d


@dataclass
class StageTimings:
    """Wall-clock milliseconds per stage. GPU stages are measured after cuda synchronize."""

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def to_dict(self) -> dict:
        return {
            "preprocess_ms": round(self.preprocess_ms, 3),
            "inference_ms": round(self.inference_ms, 3),
            "postprocess_ms": round(self.postprocess_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass
class DetectionResult:
    phrases: list[str]
    prompt: str  # exact text fed to the model
    image_size: tuple[int, int]  # (width, height) of the original image
    input_size: tuple[int, int]  # (width, height) of the tensor fed to the model
    detections: list[Detection] = field(default_factory=list)
    timings: StageTimings = field(default_factory=StageTimings)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "phrases": self.phrases,
            "image_size": list(self.image_size),
            "input_size": list(self.input_size),
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "timings": self.timings.to_dict(),
        }
