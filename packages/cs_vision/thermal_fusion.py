"""Thermal & Multi-Spectral Vision Fusion Engine for Industrial Conveyors (§6.2, §6.10).

Processes radiometric infrared temperature matrices, aligns thermal and RGB frames via planar
homography, extracts bag surface temperature distributions, and detects critical thermal
anomalies (hot powder seam ruptures, moisture condensation, and conveyor roller overheating).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThermalColorMap(str, Enum):
    INFERNO = "inferno"
    JET = "jet"
    IRONBOW = "ironbow"
    HOT = "hot"


@dataclass
class ThermalAnomaly:
    """Detected thermal defect or safety hazard on conveyor or bag."""

    anomaly_type: str  # 'hot_leak', 'moisture_cold_spot', 'bearing_overheat'
    severity: str  # 'warning', 'critical'
    mean_temperature_c: float
    peak_temperature_c: float
    delta_t_c: float
    bbox: list[float]  # [x1, y1, x2, y2]
    centroid: tuple[float, float]
    description: str


@dataclass
class ThermalBagProfile:
    """Thermal signature of an individual bag on the conveyor."""

    track_id: int | None
    mean_temp_c: float
    max_temp_c: float
    min_temp_c: float
    std_temp_c: float
    is_normal: bool
    anomalies: list[ThermalAnomaly] = field(default_factory=list)


class MultiSpectralAligner:
    """Aligns low-resolution radiometric thermal frame to RGB camera coordinate space."""

    def __init__(
        self,
        homography_matrix: np.ndarray | None = None,
        target_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.target_size = target_size
        if homography_matrix is not None:
            self.H = np.asarray(homography_matrix, dtype=np.float32)
        else:
            # Identity scaling default
            self.H = np.eye(3, dtype=np.float32)

    def set_calibration_points(
        self,
        src_points: Sequence[tuple[float, float]],
        dst_points: Sequence[tuple[float, float]],
    ) -> None:
        """Compute homography matrix from corresponding fiducial points (min 4 pairs)."""
        pts_src = np.array(src_points, dtype=np.float32)
        pts_dst = np.array(dst_points, dtype=np.float32)
        if len(pts_src) >= 4 and len(pts_dst) >= 4:
            H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
            if H is not None:
                self.H = H

    def align(self, thermal_radiometric_c: np.ndarray) -> np.ndarray:
        """Warp and resize radiometric thermal matrix to match target RGB frame geometry."""
        h_t, w_t = self.target_size[1], self.target_size[0]
        aligned = cv2.warpPerspective(
            thermal_radiometric_c.astype(np.float32),
            self.H,
            (w_t, h_t),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned


class ThermalVisionAnalyzer:
    """Analyzes radiometric temperatures and detects hot powder leaks & temperature defects."""

    def __init__(
        self,
        normal_temp_range: tuple[float, float] = (45.0, 82.0),
        leak_gradient_threshold_c: float = 12.0,
        moisture_drop_threshold_c: float = 18.0,
        bearing_overheat_threshold_c: float = 85.0,
    ) -> None:
        self.normal_temp_min, self.normal_temp_max = normal_temp_range
        self.leak_gradient_threshold = leak_gradient_threshold_c
        self.moisture_drop_threshold = moisture_drop_threshold_c
        self.bearing_overheat_threshold = bearing_overheat_threshold_c

    def analyze_bag_temperature(
        self,
        thermal_c_map: np.ndarray,
        box: list[float],
        mask: np.ndarray | None = None,
        track_id: int | None = None,
    ) -> ThermalBagProfile:
        """Extract bag surface temperature distribution and detect burst leaks or cold moisture."""
        x1, y1, x2, y2 = [int(v) for v in box]
        h_map, w_map = thermal_c_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_map, x2), min(h_map, y2)

        if x2 <= x1 or y2 <= y1:
            return ThermalBagProfile(
                track_id=track_id,
                mean_temp_c=0.0,
                max_temp_c=0.0,
                min_temp_c=0.0,
                std_temp_c=0.0,
                is_normal=True,
            )

        crop_temp = thermal_c_map[y1:y2, x1:x2]

        if mask is not None:
            crop_mask = mask[y1:y2, x1:x2] > 0
            if np.any(crop_mask):
                valid_temps = crop_temp[crop_mask]
            else:
                valid_temps = crop_temp.flatten()
        else:
            valid_temps = crop_temp.flatten()

        mean_val = float(np.mean(valid_temps))
        max_val = float(np.max(valid_temps))
        min_val = float(np.min(valid_temps))
        std_val = float(np.std(valid_temps))

        anomalies: list[ThermalAnomaly] = []

        # 1. Hot cement powder rupture / seam leak detection
        # Hot escaping powder causes a sharp local hotspot exceeding mean by leak_gradient_threshold
        if (max_val - mean_val) >= self.leak_gradient_threshold and max_val > self.normal_temp_min:
            # Find hotspot location
            local_max_idx = np.unravel_index(np.argmax(crop_temp), crop_temp.shape)
            hx, hy = x1 + local_max_idx[1], y1 + local_max_idx[0]
            anomalies.append(
                ThermalAnomaly(
                    anomaly_type="hot_leak",
                    severity="critical" if (max_val - mean_val) > 20.0 else "warning",
                    mean_temperature_c=round(mean_val, 1),
                    peak_temperature_c=round(max_val, 1),
                    delta_t_c=round(max_val - mean_val, 1),
                    bbox=[hx - 15, hy - 15, hx + 15, hy + 15],
                    centroid=(float(hx), float(hy)),
                    description=f"Hot cement powder leak detected (+{max_val - mean_val:.1f}°C spike)",
                )
            )

        # 2. Moisture / water ingress anomaly
        if (mean_val - min_val) >= self.moisture_drop_threshold and min_val < 30.0:
            anomalies.append(
                ThermalAnomaly(
                    anomaly_type="moisture_cold_spot",
                    severity="warning",
                    mean_temperature_c=round(mean_val, 1),
                    peak_temperature_c=round(min_val, 1),
                    delta_t_c=round(mean_val - min_val, 1),
                    bbox=[x1, y1, x2, y2],
                    centroid=(float((x1 + x2) / 2), float((y1 + y2) / 2)),
                    description="Moisture condensation or wet package surface detected",
                )
            )

        is_normal = len(anomalies) == 0 and (self.normal_temp_min <= mean_val <= self.normal_temp_max)

        return ThermalBagProfile(
            track_id=track_id,
            mean_temp_c=round(mean_val, 1),
            max_temp_c=round(max_val, 1),
            min_temp_c=round(min_val, 1),
            std_temp_c=round(std_val, 1),
            is_normal=is_normal,
            anomalies=anomalies,
        )

    def generate_thermal_heatmap(
        self,
        thermal_c_map: np.ndarray,
        min_temp_c: float = 20.0,
        max_temp_c: float = 90.0,
        colormap: ThermalColorMap = ThermalColorMap.INFERNO,
    ) -> np.ndarray:
        """Convert float32 temperature matrix to false-color RGB image (BGR format)."""
        clipped = np.clip(thermal_c_map, min_temp_c, max_temp_c)
        normalized = ((clipped - min_temp_c) / (max_temp_c - min_temp_c) * 255.0).astype(np.uint8)

        if colormap == ThermalColorMap.JET:
            cv_map = cv2.COLORMAP_JET
        elif colormap == ThermalColorMap.HOT:
            cv_map = cv2.COLORMAP_HOT
        elif colormap == ThermalColorMap.IRONBOW or colormap == ThermalColorMap.INFERNO:
            cv_map = cv2.COLORMAP_INFERNO
        else:
            cv_map = cv2.COLORMAP_INFERNO

        heatmap_bgr = cv2.applyColorMap(normalized, cv_map)
        return heatmap_bgr

    def fuse_rgb_and_thermal(
        self,
        rgb_bgr: np.ndarray,
        thermal_c_map: np.ndarray,
        alpha: float = 0.55,
        min_temp_c: float = 20.0,
        max_temp_c: float = 90.0,
    ) -> np.ndarray:
        """Alpha-blend visible RGB camera frame with thermal false-color heatmap."""
        heatmap_bgr = self.generate_thermal_heatmap(thermal_c_map, min_temp_c, max_temp_c)
        if heatmap_bgr.shape[:2] != rgb_bgr.shape[:2]:
            heatmap_bgr = cv2.resize(heatmap_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))

        fused = cv2.addWeighted(rgb_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
        return fused

    @staticmethod
    def generate_synthetic_thermal_frame(
        canvas_size: tuple[int, int] = (640, 640),
        bag_boxes: Sequence[list[float]] | None = None,
        ambient_temp_c: float = 24.0,
        bag_temp_c: float = 68.0,
        inject_leak: bool = False,
    ) -> np.ndarray:
        """Synthesize a realistic radiometric temperature matrix (float32 °C) for tests/simulation."""
        w, h = canvas_size
        # Ambient temperature background with slight sensor noise
        thermal = np.full((h, w), ambient_temp_c, dtype=np.float32)
        noise = np.random.normal(0.0, 0.4, (h, w)).astype(np.float32)
        thermal += noise

        # Conveyor belt rubber friction warmth (~32°C)
        thermal[100:540, :] += 8.0

        if bag_boxes:
            for i, box in enumerate(bag_boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    # Warm cement filling
                    bw, bh = x2 - x1, y2 - y1
                    Y, X = np.ogrid[:bh, :bw]
                    dist_from_center = np.sqrt(((X - bw / 2) / (bw / 2)) ** 2 + ((Y - bh / 2) / (bh / 2)) ** 2)
                    bag_heat = bag_temp_c - (dist_from_center * 10.0)
                    bag_mask = dist_from_center <= 1.0
                    thermal[y1:y2, x1:x2][bag_mask] = bag_heat[bag_mask]

                    # Optional simulated hot leak plume on first bag
                    if inject_leak and i == 0:
                        leak_x, leak_y = (x1 + x2) // 2, (y1 + y2) // 2
                        thermal[leak_y - 8 : leak_y + 8, leak_x - 8 : leak_x + 8] = bag_temp_c + 18.5

        return thermal
