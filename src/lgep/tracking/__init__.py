"""Per-camera multi-object tracking (ByteTrack) with persistent per-camera local track IDs."""

from lgep.tracking.ids import IdAllocator
from lgep.tracking.tracker import CameraTracker, Track, TrackerConfig, TrackingResult

__all__ = ["CameraTracker", "IdAllocator", "Track", "TrackerConfig", "TrackingResult"]
