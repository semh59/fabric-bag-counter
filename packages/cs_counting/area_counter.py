"""AreaIntegralCounter: Independent area-based bag count estimator (§6.9)."""

from __future__ import annotations

from typing import Sequence
import numpy as np

# Normalizes belt_speed_px_per_frame (pixels/frame) down to a "per-100px-of-
# travel" unit before scaling accumulated mask area. This keeps the area
# integral's magnitude stable across different calibrated belt speeds --
# without it, a faster belt would inflate accumulated_area purely from
# larger belt_speed values, independent of how much bag area actually
# passed the gate. 100.0 is an arbitrary but fixed reference scale (not a
# physical unit), chosen so belt speeds in the tens-of-px/frame range used
# throughout this codebase produce a per-frame area contribution close to
# the raw frame_area itself.
SPEED_NORMALIZATION_DIVISOR = 100.0


class AreaIntegralCounter:
    """Independent second count estimator integrating mask areas over the gate region.
    
    Formula: count_estimate = total_mask_area_at_gate / mean_bag_gate_area_px
    """

    def __init__(
        self,
        mean_bag_gate_area_px: float | None = None,
        discrepancy_threshold: float = 0.08,  # 8% allowable delta
        is_scale_calibrated: bool = False,
    ) -> None:
        self.mean_bag_gate_area_px = mean_bag_gate_area_px
        self.discrepancy_threshold = discrepancy_threshold
        self.is_scale_calibrated = is_scale_calibrated
        self.accumulated_area: float = 0.0
        self.observed_frames_with_bag: int = 0

    def update_calibration(self, mean_bag_area_px: float | None, is_active: bool) -> None:
        """Update scale calibration status."""
        self.mean_bag_gate_area_px = mean_bag_area_px
        self.is_scale_calibrated = is_active and (mean_bag_area_px is not None and mean_bag_area_px > 0)

    def reset(self) -> None:
        """Reset accumulator for a new session."""
        self.accumulated_area = 0.0
        self.observed_frames_with_bag = 0

    def process_frame_masks(
        self,
        masks: Sequence[np.ndarray],
        belt_speed_px_per_frame: float = 5.0,
    ) -> float:
        """Accumulate mask area flux across conveyor cross-section.
        
        If active scale calibration is absent, bypasses computation (§5.3).
        """
        if not self.is_scale_calibrated or self.mean_bag_gate_area_px is None:
            return 0.0

        if not masks:
            return self.get_estimate()

        frame_area = sum(float(np.sum(m > 0)) for m in masks)
        if frame_area > 0:
            # Area flux = (instantaneous_area / nominal_bag_length_px) * belt_speed
            # Alternatively, area integral scaled by frame sampling speed
            speed = max(1.0, belt_speed_px_per_frame)
            # Area slice normalized per frame (see SPEED_NORMALIZATION_DIVISOR)
            self.accumulated_area += frame_area * (speed / SPEED_NORMALIZATION_DIVISOR)
            self.observed_frames_with_bag += 1

        return self.get_estimate()

    def get_estimate(self) -> float:
        """Calculate current area-based estimated bag count."""
        if not self.is_scale_calibrated or self.mean_bag_gate_area_px is None or self.mean_bag_gate_area_px <= 0:
            return 0.0
        return float(self.accumulated_area / self.mean_bag_gate_area_px)

    def check_discrepancy(self, ledger_count: int) -> tuple[bool, float]:
        """Compare ledger net count with area-integral estimate.
        
        Returns:
            (has_discrepancy, relative_delta)
        """
        if not self.is_scale_calibrated or ledger_count < 10:
            # Avoid noisy triggers at start of session
            return False, 0.0

        estimate = self.get_estimate()
        if estimate <= 0:
            return False, 0.0

        delta = abs(ledger_count - estimate)
        rel_diff = delta / max(1.0, float(ledger_count))
        has_discrepancy = rel_diff > self.discrepancy_threshold

        return has_discrepancy, rel_diff
