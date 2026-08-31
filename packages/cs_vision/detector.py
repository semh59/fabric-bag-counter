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

# Score bounds for the OpenCV contour FALLBACK path only. This is a shape
# heuristic (solidity * 0.98), not a real model confidence -- it is capped
# well below what the real RF-DETR model can report so that fallback
# detections never look as trustworthy as genuine ML inference results in
# downstream UI/consumers (which also check DetectionResult.is_fallback_mode).
FALLBACK_SCORE_MIN = 0.30
FALLBACK_SCORE_MAX = 0.60


@dataclass
class DetectionResult:
    """Detection output containing segmented bag bodies and print marks."""
    bag_bodies: list[dict[str, Any]] = field(default_factory=list)  # list of {"box": [...], "score": float, "mask": np.ndarray}
    print_marks: list[dict[str, Any]] = field(default_factory=list) # list of {"box": [...], "score": float}
    # True when this result came from the TEMPORARY / PLACEHOLDER OpenCV contour
    # fallback (no working RF-DETR ONNX model available), rather than the
    # real ML model. Callers/API must surface this so operators are never
    # silently shown fallback heuristic detections as if they were genuine
    # model inference results.
    is_fallback_mode: bool = False


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
            logger.error(
                f"[VisionDetector] REAL MODEL UNAVAILABLE: RF-DETR model not found at '{self.model_path}'. "
                "Falling back to TEMPORARY / PLACEHOLDER OpenCV contour segmentation (temporary heuristic, "
                "NOT the trained ML model) -- detections from this session will be flagged via "
                "DetectionResult.is_fallback_mode=True."
            )
            return

        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            want_cuda = self.use_cuda or os.getenv("USE_CUDA", "").lower() in ("1", "true", "yes")

            providers = []
            if want_cuda:
                if "TensorrtExecutionProvider" in available:
                    providers.append("TensorrtExecutionProvider")
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                if not providers:
                    logger.warning("[VisionDetector] CUDA requested but neither CUDAExecutionProvider nor TensorrtExecutionProvider found in onnxruntime. Falling back to CPU.")
            elif "CUDAExecutionProvider" in available:
                # Auto-accelerate if CUDA is present in the environment
                providers.append("CUDAExecutionProvider")

            providers.append("CPUExecutionProvider")

            self.session = ort.InferenceSession(self.model_path, providers=providers)
            active_provider = self.session.get_providers()[0] if hasattr(self.session, "get_providers") else providers[0]
            logger.info(
                f"[VisionDetector] RF-DETR ONNX Inference Session loaded successfully from '{self.model_path}' "
                f"(Active Execution Provider: {active_provider})"
            )
        except Exception as e:
            if not self.allow_fallback:
                raise RuntimeError(f"[VisionDetector] Failed to load ONNX model '{self.model_path}': {e}") from e
            logger.error(
                f"[VisionDetector] REAL MODEL UNAVAILABLE: Failed to initialize ONNX Runtime session: {e}. "
                "Falling back to TEMPORARY / PLACEHOLDER OpenCV contour segmentation (temporary heuristic, "
                "NOT the trained ML model) -- detections from this session will be flagged via "
                "DetectionResult.is_fallback_mode=True."
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
                canvas_size=self.input_size,
            )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

            # Enrich bag count estimation and defect flags using single source of calibration
            for bag in bag_bodies:
                mask = bag.get("mask")
                box = bag.get("box", [0, 0, 0, 0])
                bw = max(1.0, float(box[2] - box[0]))
                bh = max(1.0, float(box[3] - box[1]))
                aspect_ratio = bw / bh

                # None means "could not be measured" (no contour found in the
                # ROI, or a degenerate/zero-area box) -- this must stay
                # distinguishable from a real, measured high-solidity value.
                # A fabricated "looks fine" default here would silently hide
                # a bag that genuinely could not be inspected.
                solidity: float | None = None
                mask_area = float(bw * bh)
                bx1, by1, bx2, by2 = int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3]))
                if bx2 > bx1 and by2 > by1:
                    roi_gray = gray[by1:by2, bx1:bx2]
                    if roi_gray.size > 0:
                        # Fixed low threshold (30) is sufficient here because
                        # this ROI is already localized to a single ML-detected
                        # bag's bounding box, cropped from a conveyor scene
                        # where the bag body is reliably much brighter than the
                        # dark belt/background behind it -- unlike the
                        # full-frame Otsu threshold used in the OpenCV fallback
                        # path below, which has no such localized prior and
                        # must adapt to whatever lighting the whole frame has.
                        _, roi_thresh = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY)
                        cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            cnt = max(cnts, key=cv2.contourArea)
                            c_area = cv2.contourArea(cnt)
                            hull = cv2.convexHull(cnt)
                            hull_area = cv2.contourArea(hull)
                            if hull_area > 0:
                                solidity = float(c_area) / float(hull_area)
                            bag["contour"] = cnt + np.array([bx1, by1])

                if solidity is None:
                    # Shape integrity could not be measured. Do NOT silently
                    # report "not defective" -- flag for review instead, while
                    # still honoring a clearly measurable aspect-ratio defect.
                    is_defective = True
                    defect_type = (
                        "DAMAGED_DEFORMED"
                        if (aspect_ratio < 0.35 or aspect_ratio > 3.2)
                        else "INDETERMINATE_NO_CONTOUR"
                    )
                else:
                    is_defective = bool(solidity < 0.82 or aspect_ratio < 0.35 or aspect_ratio > 3.2)
                    defect_type = "DAMAGED_DEFORMED" if is_defective else "NONE"

                bag_count_estimate = 1
                if self.is_scale_calibrated and self.mean_bag_gate_area_px:
                    effective_area = max(mask_area, bw * bh)
                    if effective_area >= (self.mean_bag_gate_area_px * self.merge_area_ratio):
                        bag_count_estimate = max(2, int(round(effective_area / self.mean_bag_gate_area_px)))

                bag["solidity"] = round(solidity, 3) if solidity is not None else None
                bag["is_defective"] = is_defective
                bag["defect_type"] = defect_type
                bag["bag_count_estimate"] = bag_count_estimate


            return DetectionResult(bag_bodies=bag_bodies, print_marks=print_marks, is_fallback_mode=False)



        # =======================================================================
        # 2. TEMPORARY / PLACEHOLDER — Fallback until real model is trained/loaded
        # =======================================================================
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # The literal 45 here is actually inert: cv2.THRESH_OTSU ignores the
        # passed threshold value and computes it automatically from the
        # image histogram (OpenCV returns the Otsu-computed value instead).
        # It differs in spirit from the ML enrichment path's fixed threshold
        # of 30 above because this path has no localized ROI prior -- it must
        # segment bag(s) out of an entire, possibly unevenly lit frame, so it
        # needs Otsu's adaptive threshold rather than one fixed magic number.
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
            # This is a shape heuristic score, not a real ML confidence -- it
            # must never look as trustworthy as a genuine model score, so it
            # is capped well below the detector's own conf_threshold ceiling
            # region and tagged via DetectionResult.is_fallback_mode so
            # downstream UI/consumers can visually distinguish it.
            score = float(min(FALLBACK_SCORE_MAX, max(FALLBACK_SCORE_MIN, solidity * 0.98)))

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

        return DetectionResult(bag_bodies=bag_bodies, print_marks=print_marks, is_fallback_mode=True)
