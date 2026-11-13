"""Re-identification: appearance embeddings, track-level features, global identities across cameras.

Local track IDs (per camera, from lgep.tracking) are never changed; global identities are separate objects
that refer to local tracks, with an association record (evidence + score) per link.
"""

from lgep.reid.features import SamplingConfig, TrackFeatureBank, TrackletRecord
from lgep.reid.identity import (AssociationConfig, AssociationRecord, CameraTopology, GlobalIdentity, IdentityManager,
                                TransitWindow)

__all__ = ["AssociationConfig", "AssociationRecord", "CameraTopology", "GlobalIdentity", "IdentityManager",
           "SamplingConfig", "TrackFeatureBank", "TrackletRecord", "TransitWindow"]
