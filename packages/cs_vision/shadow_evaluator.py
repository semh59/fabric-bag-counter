"""Live Shadow Model A/B Parallel Evaluation Engine (§6.4, §10).

Production-grade A/B evaluation engine utilizing Hungarian Bipartite Matching (scipy linear_sum_assignment),
streaming Welford online variance updates, and Sequential Probability Ratio Testing (SPRT)
to mathematically certify neural network model upgrades on live conveyor video streams.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class WelfordStats:
    """Online streaming mean and variance accumulator (Welford algorithm)."""

    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return (self.M2 / (self.count - 1)) if self.count > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class ShadowComparisonMetrics:
    """Comprehensive statistical A/B metrics comparing active vs. shadow models."""

    active_model_id: int
    shadow_model_id: int
    frames_compared: int
    agreement_rate: float
    mean_iou: float
    iou_stddev: float
    active_latency_ms: float
    shadow_latency_ms: float
    latency_delta_ms: float
    active_fps: float
    shadow_fps: float
    shadow_count_diff: int
    true_positive_matches: int
    false_positives: int
    false_negatives: int
    is_ready_for_promotion: bool
    sprt_log_likelihood_ratio: float = 0.0


class ShadowModelEvaluator:
    """Parallel evaluator comparing active vs shadow ONNX models using Hungarian matching."""

    def __init__(
        self,
        active_model_id: int,
        shadow_model_id: int,
        active_detector: Any = None,
        shadow_detector: Any = None,
        window_size: int = 300,
        promotion_iou_threshold: float = 0.85,
        promotion_agreement_threshold: float = 0.95,
        match_iou_threshold: float = 0.50,
    ) -> None:
        self.active_model_id = active_model_id
        self.shadow_model_id = shadow_model_id
        self.active_detector = active_detector
        self.shadow_detector = shadow_detector
        self.window_size = window_size
        self.promotion_iou_threshold = promotion_iou_threshold
        self.promotion_agreement_threshold = promotion_agreement_threshold
        self.match_iou_threshold = match_iou_threshold

        self._ious = deque(maxlen=window_size)
        self._agreements = deque(maxlen=window_size)
        self._active_times = deque(maxlen=window_size)
        self._shadow_times = deque(maxlen=window_size)
        self._count_deltas = deque(maxlen=window_size)

        self._iou_welford = WelfordStats()
        self._act_latency_welford = WelfordStats()
        self._shd_latency_welford = WelfordStats()

        self._tp_count = 0
        self._fp_count = 0
        self._fn_count = 0
        self._sprt_llr = 0.0  # Sequential Probability Ratio Test Log-Likelihood

    @staticmethod
    def _compute_iou(boxA: list[float], boxB: list[float]) -> float:
        """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        union = areaA + areaB - inter

        return (inter / union) if union > 0 else 0.0

    def evaluate_frame(self, image: np.ndarray) -> dict[str, Any]:
        """Run optimal bipartite Hungarian matching between active and shadow inferences."""
        if self.active_detector is None or self.shadow_detector is None:
            return {"status": "inactive"}

        # 1. Active Model Inference
        t0 = time.perf_counter()
        active_res = self.active_detector.predict(image)
        t_active = time.perf_counter() - t0
        self._active_times.append(t_active)
        self._act_latency_welford.update(t_active * 1000)

        # 2. Shadow Model Inference
        t1 = time.perf_counter()
        shadow_res = self.shadow_detector.predict(image)
        t_shadow = time.perf_counter() - t1
        self._shadow_times.append(t_shadow)
        self._shd_latency_welford.update(t_shadow * 1000)

        active_boxes = [b["box"] for b in active_res.bag_bodies]
        shadow_boxes = [b["box"] for b in shadow_res.bag_bodies]

        nA = len(active_boxes)
        nS = len(shadow_boxes)

        count_match = 1 if nA == nS else 0
        self._agreements.append(count_match)
        self._count_deltas.append(nS - nA)

        # 3. Hungarian Maximum Bipartite Matching on pairwise IoU Cost Matrix
        matched_ious: list[float] = []

        if nA > 0 and nS > 0:
            cost_matrix = np.zeros((nA, nS), dtype=np.float32)
            for i, a_box in enumerate(active_boxes):
                for j, s_box in enumerate(shadow_boxes):
                    cost_matrix[i, j] = 1.0 - self._compute_iou(a_box, s_box)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                iou = 1.0 - float(cost_matrix[r, c])
                if iou >= self.match_iou_threshold:
                    matched_ious.append(iou)
                    self._tp_count += 1
                else:
                    self._fn_count += 1
                    self._fp_count += 1

            # Unmatched predictions
            self._fn_count += max(0, nA - len(row_ind))
            self._fp_count += max(0, nS - len(col_ind))

        elif nA > 0 and nS == 0:
            self._fn_count += nA
        elif nA == 0 and nS > 0:
            self._fp_count += nS

        mean_frame_iou = float(np.mean(matched_ious)) if matched_ious else (1.0 if (nA == 0 and nS == 0) else 0.0)
        self._ious.append(mean_frame_iou)
        self._iou_welford.update(mean_frame_iou)

        # 4. Wald's SPRT (Sequential Probability Ratio Test) update
        # H0: Agreement Rate p0 = 0.90 vs H1: Agreement Rate p1 = 0.98
        p0, p1 = 0.90, 0.98
        if count_match == 1:
            self._sprt_llr += math.log(p1 / p0)
        else:
            self._sprt_llr += math.log((1.0 - p1) / (1.0 - p0))

        return {
            "active_count": nA,
            "shadow_count": nS,
            "matched_pairs": len(matched_ious),
            "mean_iou": round(mean_frame_iou, 3),
            "active_ms": round(t_active * 1000, 2),
            "shadow_ms": round(t_shadow * 1000, 2),
            "sprt_llr": round(self._sprt_llr, 2),
        }

    def get_comparison_summary(self) -> ShadowComparisonMetrics:
        """Return statistically rigorous A/B comparison metrics."""
        frames = len(self._agreements)
        if frames == 0:
            return ShadowComparisonMetrics(
                active_model_id=self.active_model_id,
                shadow_model_id=self.shadow_model_id,
                frames_compared=0,
                agreement_rate=1.0,
                mean_iou=1.0,
                iou_stddev=0.0,
                active_latency_ms=0.0,
                shadow_latency_ms=0.0,
                latency_delta_ms=0.0,
                active_fps=0.0,
                shadow_fps=0.0,
                shadow_count_diff=0,
                true_positive_matches=0,
                false_positives=0,
                false_negatives=0,
                is_ready_for_promotion=False,
                sprt_log_likelihood_ratio=0.0,
            )

        agreement = float(np.mean(self._agreements))
        mean_iou = float(np.mean(self._ious)) if self._ious else 0.0
        avg_act_t = float(np.mean(self._active_times)) if self._active_times else 0.001
        avg_shd_t = float(np.mean(self._shadow_times)) if self._shadow_times else 0.001

        # Promotion criteria: min 30 frames, statistical agreement, mean IoU, and SPRT acceptance
        is_ready = (
            frames >= 30
            and agreement >= self.promotion_agreement_threshold
            and mean_iou >= self.promotion_iou_threshold
            and self._sprt_llr >= 2.94  # log((1-beta)/alpha) for alpha=0.05, beta=0.05
        )

        return ShadowComparisonMetrics(
            active_model_id=self.active_model_id,
            shadow_model_id=self.shadow_model_id,
            frames_compared=frames,
            agreement_rate=round(agreement, 3),
            mean_iou=round(mean_iou, 3),
            iou_stddev=round(self._iou_welford.stddev, 3),
            active_latency_ms=round(avg_act_t * 1000, 2),
            shadow_latency_ms=round(avg_shd_t * 1000, 2),
            latency_delta_ms=round((avg_shd_t - avg_act_t) * 1000, 2),
            active_fps=round(1.0 / avg_act_t, 1) if avg_act_t > 0 else 0.0,
            shadow_fps=round(1.0 / avg_shd_t, 1) if avg_shd_t > 0 else 0.0,
            shadow_count_diff=int(np.sum(self._count_deltas)),
            true_positive_matches=self._tp_count,
            false_positives=self._fp_count,
            false_negatives=self._fn_count,
            is_ready_for_promotion=is_ready,
            sprt_log_likelihood_ratio=round(self._sprt_llr, 2),
        )
