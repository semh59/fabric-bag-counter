"""Brown-Conrady Lens Distortion Model & Progressive RANSAC Conveyor Rail Fitting (§6.1, §6.3).

Provides optical lens distortion removal (radial + tangential) and automatic conveyor
rail line estimation via robust RANSAC geometric consensus.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LensDistortionParams:
    """Camera intrinsic parameters and Brown-Conrady distortion coefficients."""

    fx: float = 800.0
    fy: float = 800.0
    cx: float = 320.0
    cy: float = 320.0
    k1: float = -0.05  # Radial distortion (barrel / pincushion)
    k2: float = 0.01
    p1: float = 0.0    # Tangential distortion (lens decentering)
    p2: float = 0.0
    k3: float = 0.0

    def get_camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    def get_dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)


class LensDistortionCorrector:
    """Removes radial/tangential optical distortion using Brown-Conrady model."""

    def __init__(self, params: LensDistortionParams | None = None) -> None:
        self.params = params or LensDistortionParams()
        self._mapx: np.ndarray | None = None
        self._mapy: np.ndarray | None = None
        self._cached_shape: tuple[int, int] | None = None

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Apply Brown-Conrady undistortion mapping to image frame."""
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        if self._cached_shape != (h, w) or self._mapx is None:
            K = self.params.get_camera_matrix()
            D = self.params.get_dist_coeffs()
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
            self._mapx, self._mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
            self._cached_shape = (h, w)

        return cv2.remap(image, self._mapx, self._mapy, interpolation=cv2.INTER_LINEAR)


class RansacConveyorRailDetector:
    """Progressive RANSAC parallel line detector for conveyor belt auto-alignment."""

    def __init__(self, max_iterations: int = 200, inlier_threshold_px: float = 3.5) -> None:
        self.max_iterations = max_iterations
        self.inlier_threshold = inlier_threshold_px

    def detect_rails(self, gray_image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Detect left and right conveyor rails and return (rail_1_pts, rail_2_pts, belt_angle_rad)."""
        if gray_image is None or gray_image.size == 0:
            return None, None, 0.0

        h, w = gray_image.shape[:2]
        # Canny edge detector
        edges = cv2.Canny(gray_image, 50, 150)
        y_indices, x_indices = np.nonzero(edges)

        if len(x_indices) < 50:
            return None, None, 0.0

        edge_points = np.column_stack((x_indices, y_indices))

        # RANSAC Line 1
        best_line1: tuple[float, float, float] | None = None
        best_inliers1: np.ndarray | None = None
        max_inliers1 = 0

        for _ in range(self.max_iterations):
            idx = np.random.choice(len(edge_points), 2, replace=False)
            p1, p2 = edge_points[idx[0]], edge_points[idx[1]]
            if p1[0] == p2[0] and p1[1] == p2[1]:
                continue

            # Line eq: Ax + By + C = 0
            A = float(p2[1] - p1[1])
            B = float(p1[0] - p2[0])
            norm = math.sqrt(A * A + B * B)
            if norm == 0:
                continue
            A, B = A / norm, B / norm
            C = -(A * p1[0] + B * p1[1])

            dist = np.abs(A * edge_points[:, 0] + B * edge_points[:, 1] + C)
            inliers = dist < self.inlier_threshold
            count = int(np.sum(inliers))

            if count > max_inliers1:
                max_inliers1 = count
                best_line1 = (A, B, C)
                best_inliers1 = inliers

        if best_line1 is None or max_inliers1 < 20:
            return None, None, 0.0

        A, B, _ = best_line1
        belt_angle = float(math.atan2(-A, B))
        line1_pts = edge_points[best_inliers1]

        # Remaining points for Rail 2
        remaining_points = edge_points[~best_inliers1]
        line2_pts = None
        if len(remaining_points) >= 20:
            best_inliers2: np.ndarray | None = None
            max_inliers2 = 0
            for _ in range(self.max_iterations // 2):
                idx = np.random.choice(len(remaining_points), 2, replace=False)
                p1, p2 = remaining_points[idx[0]], remaining_points[idx[1]]
                A2 = float(p2[1] - p1[1])
                B2 = float(p1[0] - p2[0])
                norm2 = math.sqrt(A2 * A2 + B2 * B2)
                if norm2 == 0:
                    continue
                A2, B2 = A2 / norm2, B2 / norm2
                # Parallel constraint (dot product with line 1 direction ~ 1.0)
                if abs(A * A2 + B * B2) < 0.90:
                    continue
                C2 = -(A2 * p1[0] + B2 * p1[1])
                dist2 = np.abs(A2 * remaining_points[:, 0] + B2 * remaining_points[:, 1] + C2)
                inliers2 = dist2 < self.inlier_threshold
                count2 = int(np.sum(inliers2))
                if count2 > max_inliers2:
                    max_inliers2 = count2
                    best_inliers2 = inliers2

            if best_inliers2 is not None and max_inliers2 >= 15:
                line2_pts = remaining_points[best_inliers2]

        return line1_pts, line2_pts, belt_angle
