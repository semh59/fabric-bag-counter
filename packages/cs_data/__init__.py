"""Data pipeline package (extract, ssim, synth, split, cvat, mining)."""

from packages.cs_data.cvat_client import CvatClient
from packages.cs_data.extract_frames import extract_video_frames
from packages.cs_data.mining import HardFrameMiner, MiningCriterion
from packages.cs_data.split_dataset import DatasetSplitter, SplitResult
from packages.cs_data.ssim_filter import SSIMFilter
from packages.cs_data.synth import SyntheticBagGenerator

__all__ = [
    "extract_video_frames",
    "SSIMFilter",
    "SyntheticBagGenerator",
    "DatasetSplitter",
    "SplitResult",
    "CvatClient",
    "HardFrameMiner",
    "MiningCriterion",
]
