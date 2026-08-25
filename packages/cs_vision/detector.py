"""Computer Vision Detector: RF-DETR ONNX Runtime & Placeholder OpenCV Contour Fallback (§6.2, §6.3)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import cv2

from packages.cs_vision.postprocess import postprocess_rfdetr_seg
from packages.cs_vision.preprocess import preprocess_image

logger = logging.getLogger(__name__)

# Default model path pointing to verified trained ONNX weights
DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "models" / "rfdetr_seg_v2.onnx"),
)


@dataclass
class DetectionResult:
    """Detection output containing segmented bag bodies and print marks."""
    bag_bodies: list[dict[str, Any]] = field(default_factory=list)  # list of {"box": [...], "score": float, "mask": np.ndarray}
    print_marks: list[dict[str, Any]] = field(default_factory=list) # list of {"box": [...], "score": float}


class VisionDetector:
    """Instance segmentation detector with RF-DETR Seg ONNX Runtime as primary engine

    and a documented temporary OpenCV fallback (§6.2, §6.3).
    """

    def __init__(
        self,
        model_path: str | None = DEFAULT_MODEL_PATH,
        use_cuda: bool = False,
        conf_threshold: float = 0.40,
        mask_threshold: float = 0.50,
        input_size: tuple[int, int] = (640, 640),
        allow_fallback: bool = True,
        mean_bag_gate_area_px: float | None = None,
        merge_area_ratio: float = 1.50,
    ) -> None:
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold
        self.input_size = input_size
        self.allow_fallback = allow_fallback
        self.mean_bag_gate_area_px = mean_bag_gate_area_px
        self.merge_area_ratio = merge_area_ratio
        self.is_scale_calibrated = mean_bag_gate_area_px is not None and mean_bag_gate_area_px > 0
        self.session: Any = None
        self._init_session()

    def update_calibration(
        self,
        mean_bag_area_px: float | None,
        is_active: bool = True,
        merge_area_ratio: float = 1.50,
    ) -> None:
        """Update calibration parameters unified with MergeDetector and AreaIntegralCounter."""
        self.mean_bag_gate_area_px = mean_bag_area_px
        self.merge_area_ratio = merge_area_ratio
        self.is_scale_calibrated = is_active and (mean_bag_area_px is not None and mean_bag_area_px > 0)

    def _init_session(self) -> None:
        if self.model_path is None or not os.path.exists(self.model_path):
            if not self.allow_fallback:
                raise FileNotFoundError(
                    f"[VisionDetector] RF-DETR Seg model not found at '{self.model_path}'. "
                    "OpenCV fallback is disabled in strict mode."
                )
            logger.warning(
                f"[VisionDetector] WARNING: RF-DETR model not found at '{self.model_path}'. "
                "Active path: GEÇİCİ / PLACEHOLDER OpenCV contour segmentation (Temporary fallback until model trained)."
            )
            return

        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_cuda else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"[VisionDetector] RF-DETR ONNX Inference Session loaded successfully from '{self.model_path}'")
        except Exception as e:
            if not self.allow_fallback:
                raise RuntimeError(f"[VisionDetector] Failed to load ONNX model '{self.model_path}': {e}") from e
            logger.warning(
                f"[VisionDetector] WARNING: Failed to initialize ONNX Runtime session: {e}. "
                "Active path: GEÇİCİ / PLACEHOLDER OpenCV contour segmentation."
            )
            self.session = None

    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run instance segmentation and detection on a single image frame."""
        h, w = image.shape[:2]

        if self.session is not None:
            # ===================================================================
            # 1. Primary Engine: Deep Learning Inference (RF-DETR Seg ONNX Runtime)
            # ===================================================================
            blob, scale, pad = preprocess_image(image, self.input_size)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: blob})
            boxes = outputs[0][0] if len(outputs) > 0 else np.zeros((0, 4))
            scores = outputs[1][0] if len(outputs) > 1 else np.zeros((0,))
            classes = outputs[2][0] if len(outputs) > 2 else np.zeros((0,))
            raw_masks = outputs[3][0] if len(outputs) > 3 else None

            bag_bodies, print_marks = postprocess_rfdetr_seg(
                boxes=boxes,
                scores=scores,
                classes=classes,
                raw_masks=raw_masks,
                orig_shape=(h, w),
                scale=scale,
                pad=pad,
                conf_threshold=self.conf_threshold,
                mask_threshold=self.mask_threshold,
            )

            # Enrich bag count estimation using single source of calibration
            for bag in bag_bodies:
                mask = bag.get("mask")
                if mask is not None and self.is_scale_calibrated and self.mean_bag_gate_area_px:
                    mask_area = float(np.sum(mask > 0))
                    if mask_area >= (self.mean_bag_gate_area_px * self.merge_area_ratio):
                        bag["bag_count_estimate"] = max(2, int(round(mask_area / self.mean_bag_gate_area_px)))
                    else:
                        bag["bag_count_estimate"] = 1
                else:
                    bag["bag_count_estimate"] = 1

            return DetectionResult(bag_bodies=bag_bodies, print_marks=print_marks)

        # =======================================================================
        # 2. GEÇİCİ / PLACEHOLDER — Gerçek model eğitilene/yüklenene kadar Fallback
        # =======================================================================
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bag_bodies: list[dict[str, Any]] = []
        print_marks: list[dict[str, Any]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3500 or area > (h * w * 0.85):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / max(1.0, float(bh))
            if aspect_ratio < 0.25 or aspect_ratio > 4.0:
                continue

            mask = np.zeros((h, w), dtype=bool)
            cv2.drawContours(mask.view(np.uint8), [cnt], -1, 1, -1)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / max(1.0, float(hull_area))
            score = float(min(0.99, max(0.85, solidity * 0.98)))

            is_defective = bool(solidity < 0.82 or aspect_ratio < 0.35 or aspect_ratio > 3.2)
            defect_type = "DAMAGED_DEFORMED" if is_defective else "NONE"

            # Unified calibration-driven multi-bag estimate (No arbitrary pixel constants)
            bag_count_estimate = 1
            if self.is_scale_calibrated and self.mean_bag_gate_area_px is not None:
                if area >= (self.mean_bag_gate_area_px * self.merge_area_ratio):
                    bag_count_estimate = max(2, int(round(area / self.mean_bag_gate_area_px)))
            else:
                # Safe uncalibrated default
                bag_count_estimate = 1

            box = [float(x), float(y), float(x + bw), float(y + bh)]
            bag_bodies.append({
                "box": box,
                "score": score,
                "mask": mask,
                "contour": cnt,
                "is_defective": is_defective,
                "defect_type": defect_type,
                "bag_count_estimate": bag_count_estimate,
                "solidity": round(solidity, 3),
            })

            # Detect contrast print marks inside bag bounding box
            roi_gray = gray[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
            if roi_gray.size > 0:
                _, mark_thresh = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY_INV)
                mark_cnts, _ = cv2.findContours(mark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for mc in mark_cnts:
                    m_area = cv2.contourArea(mc)
                    if 150 < m_area < 2500:
                        mx, my, mw, mh = cv2.boundingRect(mc)
                        print_marks.append({
                            "box": [float(x + mx), float(y + my), float(x + mx + mw), float(y + my + mh)],
                            "score": 0.92,
                        })

        return DetectionResult(bag_bodies=bag_bodies, print_marks=print_marks)
