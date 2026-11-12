"""Text-guided object detection backends.

Heavy model modules are imported lazily so importing ``lgep.perception`` stays cheap.
"""

from lgep.perception.types import Detection, DetectionResult, StageTimings

__all__ = ["Detection", "DetectionResult", "StageTimings"]
