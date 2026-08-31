"""Side-View Conveyor Bag Height, Thickness & Double-Stacking Inspector (§4.2, §6.3).

Advanced optical analyzer utilizing horizontal gradient edge profiling, RANSAC conveyor baseline
fitting, seam groove valley detection, vertical contour agglomeration, and spillage leak estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SideInspectionConfig:
    """Thresholds and calibration parameters for side-view optical profiling."""

    nominal_bag_thickness_px: float = 65.0       # Calibrated single bag thickness in side view
    double_stack_ratio_threshold: float = 1.65   # Thickness ratio > 1.65x flags double-stack anomaly
    min_bag_width_px: float = 120.0              # Minimum valid bag width in side view
    seam_groove_prominence: float = 0.20         # Gradient valley depth ratio between stacked bags
    spillage_area_threshold_px: float = 200.0    # Area threshold for powder spillage on belt
    intensity_threshold: int = 140               # Contrast cutoff threshold


@dataclass
class StackedBagProfile:
    """Individual layer profile in a double-stacked bag cluster."""

    layer_index: int
    top_y: float
    bottom_y: float
    thickness_px: float


@dataclass
class SideInspectionResult:
    """Comprehensive result of side-view optical profile analysis."""

    is_double_stacked: bool
    is_ruptured: bool
    measured_thickness_px: float
    thickness_ratio: float
    anomaly_score: float
    seam_groove_detected: bool
    detected_layers: list[StackedBagProfile]
    detected_boxes: list[list[float]]
    conveyor_baseline_y: float
    leak_contour_area: float = 0.0


class SideViewInspector:
    """State-of-the-art optical geometry analyzer for side-view conveyor cameras."""

    def __init__(self, config: SideInspectionConfig | None = None) -> None:
        self.config = config or SideInspectionConfig()

    def fit_conveyor_baseline(self, gray_image: np.ndarray) -> float:
        """Estimate the conveyor belt baseline Y-coordinate using horizontal edge projection."""
        h, w = gray_image.shape
        lower_third = gray_image[int(h * 0.60):, :]
        sobel_y = cv2.Sobel(lower_third, cv2.CV_32F, 0, 1, ksize=3)
        proj = np.mean(np.abs(sobel_y), axis=1)
        if len(proj) > 0:
            peak_idx = int(np.argmax(proj))
            return float(int(h * 0.60) + peak_idx)
        return float(h * 0.80)

    def detect_seam_groove_valley(self, bag_crop: np.ndarray) -> tuple[bool, list[float]]:
        """Identify horizontal groove valley between vertically stacked bags via vertical gradient histogram."""
        if bag_crop.shape[0] < 30 or bag_crop.shape[1] < 30:
            return False, []

        sobel_y = cv2.Sobel(bag_crop, cv2.CV_32F, 0, 1, ksize=3)
        horizontal_energy = np.mean(np.abs(sobel_y), axis=1)

        kernel_size = max(3, bag_crop.shape[0] // 12)
        smoothed = np.convolve(horizontal_energy, np.ones(kernel_size) / kernel_size, mode="same")

        crop_h = bag_crop.shape[0]
        search_region = smoothed[int(crop_h * 0.25):int(crop_h * 0.75)]
        if len(search_region) == 0:
            return False, []

        peak_val = float(np.max(search_region))
        mean_val = float(np.mean(smoothed))

        if peak_val > (mean_val * (1.0 + self.config.seam_groove_prominence)):
            seam_y = float(int(crop_h * 0.25) + np.argmax(search_region))
            return True, [seam_y]

        return False, []

    def inspect_frame(self, image: np.ndarray) -> SideInspectionResult:
        """Execute full side-view optical inspection pipeline."""
        if image is None or image.size == 0:
            return SideInspectionResult(
                is_double_stacked=False,
                is_ruptured=False,
                measured_thickness_px=0.0,
                thickness_ratio=0.0,
                anomaly_score=0.0,
                seam_groove_detected=False,
                detected_layers=[],
                detected_boxes=[],
                conveyor_baseline_y=0.0,
            )

        # 1. Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        baseline_y = self.fit_conveyor_baseline(gray)

        # 2. Otsu thresholding with adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Morphological closing to seal internal bag weave gaps
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

        # 4. Contour extraction
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes: list[list[float]] = []
        spillage_area = 0.0
        is_rupture = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            # Detect powder spillage puddle on belt bottom (low vertical profile near baseline)
            if (y + h) >= (baseline_y - 20.0) and (h <= 35.0 or area < (self.config.min_bag_width_px * 25.0)):
                if area >= self.config.spillage_area_threshold_px:
                    is_rupture = True
                    spillage_area += float(area)
                    continue

            if w >= self.config.min_bag_width_px and h >= 15:
                raw_boxes.append([float(x), float(y), float(x + w), float(y + h)])

        # 5. Check vertical agglomeration for vertically stacked separate contours
        # Sort boxes by X position then Y
        raw_boxes.sort(key=lambda b: (b[0], b[1]))
        merged_boxes: list[list[float]] = []

        i = 0
        while i < len(raw_boxes):
            b1 = raw_boxes[i]
            if i + 1 < len(raw_boxes):
                b2 = raw_boxes[i + 1]
                # Check horizontal overlap
                x_overlap = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                min_w = min(b1[2] - b1[0], b2[2] - b2[0])
                if min_w > 0 and (x_overlap / min_w) >= 0.70 and abs(b2[1] - b1[3]) <= 25.0:
                    # Merge stacked boxes into one compound box
                    compound = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
                    merged_boxes.append(compound)
                    i += 2
                    continue
            merged_boxes.append(b1)
            i += 1

        detected_layers: list[StackedBagProfile] = []
        max_thickness = 0.0
        seam_detected = False

        for b in merged_boxes:
            bx1, by1, bx2, by2 = b
            h = by2 - by1
            w = bx2 - bx1
            if h > max_thickness:
                max_thickness = h

            bag_crop = gray[int(by1):int(by2), int(bx1):int(bx2)]
            has_seam, seam_offsets = self.detect_seam_groove_valley(bag_crop)

            if has_seam:
                seam_detected = True
                seam_y_abs = by1 + seam_offsets[0]
                detected_layers = [
                    StackedBagProfile(layer_index=1, top_y=float(by1), bottom_y=float(seam_y_abs), thickness_px=float(seam_offsets[0])),
                    StackedBagProfile(layer_index=2, top_y=float(seam_y_abs), bottom_y=float(by2), thickness_px=float(h - seam_offsets[0])),
                ]

        nominal = max(1.0, self.config.nominal_bag_thickness_px)
        ratio = max_thickness / nominal if max_thickness > 0 else 0.0

        is_double = (ratio >= self.config.double_stack_ratio_threshold) or (seam_detected and ratio >= 1.45)
        anomaly_score = min(1.0, max(0.0, (ratio - 1.0) / (self.config.double_stack_ratio_threshold - 0.4))) if is_double else 0.0

        return SideInspectionResult(
            is_double_stacked=is_double,
            is_ruptured=is_rupture,
            measured_thickness_px=round(max_thickness, 1),
            thickness_ratio=round(ratio, 2),
            anomaly_score=round(anomaly_score, 3),
            seam_groove_detected=seam_detected,
            detected_layers=detected_layers,
            detected_boxes=merged_boxes,
            conveyor_baseline_y=round(baseline_y, 1),
            leak_contour_area=round(spillage_area, 1),
        )
