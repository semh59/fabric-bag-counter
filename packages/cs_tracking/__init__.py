"""Tracking and conveyor motion package (BeltMotionModel, ByteTrack, merge_detector)."""

# Source: Zhang et al., "ByteTrack" (ECCV 2022), arXiv:2110.06864
# Reference code: github.com/FoundationVision/ByteTrack (MIT)
# Novel part: BeltMotionModel, mask-IoU cost matrix, crossing_seq per track, merge_detector multi-signal

from packages.cs_tracking.matching import associate_detections_to_tracks
from packages.cs_tracking.merge_detector import MergeDetector, MergeHypothesis
from packages.cs_tracking.motion import BeltMotionModel
from packages.cs_tracking.tracker import BagTrack, ConveyorByteTracker

__all__ = [
    "BeltMotionModel",
    "BagTrack",
    "ConveyorByteTracker",
    "associate_detections_to_tracks",
    "MergeDetector",
    "MergeHypothesis",
]
