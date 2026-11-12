"""ByteTrack (Zhang et al., ECCV 2022), vendored from the official implementation.

Source: https://github.com/ifzhang/ByteTrack at d1bf0191adff59bc8fcfeaa0b33d3d1642552a99, MIT licence
(``LICENSE`` in this folder). Every change is marked ``lgep:`` in the code; ``byte_tracker.py`` lists them.
"""

from lgep.tracking.bytetrack.basetrack import TrackState
from lgep.tracking.bytetrack.byte_tracker import BYTETracker, STrack

__all__ = ["BYTETracker", "STrack", "TrackState"]
