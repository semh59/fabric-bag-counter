"""BeltMotionModel: Optical-flow assisted conveyor velocity estimator and motion prior (§6.6)."""

# Source: Zhang et al., "ByteTrack" (ECCV 2022), arXiv:2110.06864
# Novel part: BeltMotionModel, real-time low-pass belt velocity prior

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


class BeltMotionModel:
    """Continuous conveyor belt velocity estimation and Kalman filter prior provider."""

    def __init__(
        self,
        default_speed_px: float = 0.0,
        default_direction: list[float] | tuple[float, float] = (1.0, 0.0),
        smoothing_alpha: float = 0.15,
    ) -> None:
        self.speed_px = float(default_speed_px)
        vx, vy = default_direction
        norm = math.sqrt(vx * vx + vy * vy)
        self.direction = (vx / norm, vy / norm) if norm > 0 else (1.0, 0.0)
        self.smoothing_alpha = smoothing_alpha
        self.velocity_vector = np.array([self.speed_px * self.direction[0], self.speed_px * self.direction[1]], dtype=np.float32)

    def update_from_calibration(self, speed_px: float | None, direction: list[float] | None) -> None:
        """Update motion model from line_calibration record."""
        if speed_px is not None:
            self.speed_px = float(speed_px)
        if direction is not None and len(direction) >= 2:
            vx, vy = direction[0], direction[1]
            norm = math.sqrt(vx * vx + vy * vy)
            self.direction = (vx / norm, vy / norm) if norm > 0 else (1.0, 0.0)
        self.velocity_vector = np.array([self.speed_px * self.direction[0], self.speed_px * self.direction[1]], dtype=np.float32)

    def update_sparse_optical_flow(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        status: np.ndarray | None = None,
    ) -> tuple[float, tuple[float, float]]:
        """Update belt velocity from tracked sparse keypoints between consecutive frames."""
        if status is not None:
            valid = status.flatten() == 1
            prev_points = prev_points[valid]
            curr_points = curr_points[valid]

        if len(prev_points) < 5:
            return self.speed_px, self.direction

        deltas = curr_points - prev_points  # shape (N, 2) = (dx, dy)
        median_dx = float(np.median(deltas[:, 0]))
        median_dy = float(np.median(deltas[:, 1]))

        measured_speed = math.sqrt(median_dx * median_dx + median_dy * median_dy)
        if measured_speed > 0.1:
            measured_dir = (median_dx / measured_speed, median_dy / measured_speed)
            # Low-pass filter
            self.speed_px = (1 - self.smoothing_alpha) * self.speed_px + self.smoothing_alpha * measured_speed
            smoothed_vx = (1 - self.smoothing_alpha) * self.direction[0] + self.smoothing_alpha * measured_dir[0]
            smoothed_vy = (1 - self.smoothing_alpha) * self.direction[1] + self.smoothing_alpha * measured_dir[1]
            norm = math.sqrt(smoothed_vx * smoothed_vx + smoothed_vy * smoothed_vy)
            self.direction = (smoothed_vx / norm, smoothed_vy / norm) if norm > 0 else (1.0, 0.0)
            self.velocity_vector = np.array([self.speed_px * self.direction[0], self.speed_px * self.direction[1]], dtype=np.float32)

        return self.speed_px, self.direction

    def get_velocity_prior(self) -> np.ndarray:
        """Return 2D velocity vector [vx, vy] in pixels per frame."""
        return self.velocity_vector.copy()
