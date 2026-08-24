"""MergeDetector: Multi-signal bag occlusion and merge detection (§6.7)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from packages.cs_core.geometry import compute_polygon_area, mask_centroid


@dataclass
class MergeHypothesis:
    """Hypothesis that a single detected mask contains multiple physical bags."""
    is_merged: bool
    confidence: float
    signal_votes: list[str] = field(default_factory=list)
    estimated_object_count: int = 1
    centroid_seeds: list[tuple[float, float]] = field(default_factory=list)


class MergeDetector:
    """Multi-signal detector to identify two merged bags under single mask segmentation."""

    def __init__(
        self,
        mean_bag_gate_area_px: float | None = None,
        merge_area_ratio: float = 1.50,
        min_votes: int = 2,
        is_scale_calibrated: bool = False,
    ) -> None:
        self.mean_bag_gate_area_px = mean_bag_gate_area_px
        self.merge_area_ratio = merge_area_ratio
        self.min_votes = min_votes
        self.is_scale_calibrated = is_scale_calibrated

    def update_calibration(self, mean_bag_area_px: float | None, is_active: bool) -> None:
        """Update scale calibration parameters."""
        self.mean_bag_gate_area_px = mean_bag_area_px
        self.is_scale_calibrated = is_active and (mean_bag_area_px is not None and mean_bag_area_px > 0)

    def analyze_detection(
        self,
        mask: np.ndarray | None,
        box: list[float],
        print_marks: list[dict[str, Any]] | None = None,
        converging_tracks: list[Any] | None = None,
    ) -> MergeHypothesis:
        """Analyze a candidate mask with 4 independent signals.
        
        If active scale calibration is absent, safely bypasses detection (§5.3).
        """
        if not self.is_scale_calibrated or self.mean_bag_gate_area_px is None:
            # Safe fallback when scale calibration is inactive
            return MergeHypothesis(is_merged=False, confidence=0.0, signal_votes=["bypassed_no_calibration"])

        votes = []
        mask_area = float(np.sum(mask > 0)) if mask is not None else float((box[2] - box[0]) * (box[3] - box[1]))

        # -------------------------------------------------------------------
        # Signal 1: Area threshold
        # -------------------------------------------------------------------
        if mask_area >= (self.mean_bag_gate_area_px * self.merge_area_ratio):
            votes.append("signal_area_oversized")

        # -------------------------------------------------------------------
        # Signal 2: Shape convexity and aspect ratio deficit
        # -------------------------------------------------------------------
        box_w = max(1.0, box[2] - box[0])
        box_h = max(1.0, box[3] - box[1])
        box_area = box_w * box_h
        solidity = mask_area / box_area if box_area > 0 else 1.0

        # Normal single bag rectangular solidity is typically 0.70 - 0.90
        # Two shingled bags create an L-shape or stepped contour with lower solidity or elongated ratio
        aspect_ratio = max(box_w / box_h, box_h / box_w)
        if solidity < 0.55 or aspect_ratio > 2.8:
            votes.append("signal_shape_convexity_deficit")

        # -------------------------------------------------------------------
        # Signal 3: Temporal convergence
        # -------------------------------------------------------------------
        if converging_tracks and len(converging_tracks) >= 2:
            votes.append("signal_temporal_track_convergence")

        # -------------------------------------------------------------------
        # Signal 4: Print mark count within same mask
        # -------------------------------------------------------------------
        if print_marks and mask is not None:
            marks_inside = 0
            for pm in print_marks:
                pbox = pm.get("box", [0, 0, 0, 0])
                pcx = int((pbox[0] + pbox[2]) / 2.0)
                pcy = int((pbox[1] + pbox[3]) / 2.0)
                if 0 <= pcy < mask.shape[0] and 0 <= pcx < mask.shape[1]:
                    if mask[pcy, pcx]:
                        marks_inside += 1
            if marks_inside >= 2:
                votes.append("signal_multiple_print_marks")

        is_merged = len(votes) >= self.min_votes
        confidence = float(len(votes)) / 4.0

        # Derive centroid seeds for latent tracks or watershed separation
        seeds = []
        if is_merged:
            # Estimate 2 centers along the major axis
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            if box_w >= box_h:
                seeds = [(box[0] + box_w * 0.25, cy), (box[0] + box_w * 0.75, cy)]
            else:
                seeds = [(cx, box[1] + box_h * 0.25), (cx, box[1] + box_h * 0.75)]
        else:
            seeds = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)]

        return MergeHypothesis(
            is_merged=is_merged,
            confidence=confidence,
            signal_votes=votes,
            estimated_object_count=2 if is_merged else 1,
            centroid_seeds=seeds,
        )
