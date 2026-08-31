"""Unit tests for Live Shadow Model A/B Parallel Evaluator (§6.4, §10)."""

from types import SimpleNamespace
import numpy as np
import pytest
from packages.cs_vision.shadow_evaluator import ShadowModelEvaluator


class DummyDetector:
    """Mock detector returning predefined boxes for A/B testing."""

    def __init__(self, boxes: list[list[float]]) -> None:
        self.boxes = boxes

    def predict(self, image: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(
            bag_bodies=[{"box": b, "score": 0.95} for b in self.boxes],
            print_marks=[],
        )


def test_shadow_evaluator_hungarian_perfect_agreement():
    boxes = [[50.0, 50.0, 200.0, 150.0], [250.0, 50.0, 400.0, 150.0]]
    active_det = DummyDetector(boxes)
    shadow_det = DummyDetector(boxes)

    evaluator = ShadowModelEvaluator(
        active_model_id=1,
        shadow_model_id=2,
        active_detector=active_det,
        shadow_detector=shadow_det,
        window_size=50,
    )

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(40):
        evaluator.evaluate_frame(img)

    summary = evaluator.get_comparison_summary()
    assert summary.frames_compared == 40
    assert summary.agreement_rate == 1.0
    assert summary.mean_iou == pytest.approx(1.0, rel=1e-3)
    assert summary.true_positive_matches == 80  # 40 frames * 2 boxes
    assert summary.false_positives == 0
    assert summary.false_negatives == 0
    assert summary.is_ready_for_promotion is True
    assert summary.sprt_log_likelihood_ratio > 2.94


def test_shadow_evaluator_hungarian_mismatch_blocks_promotion():
    active_boxes = [[50.0, 50.0, 200.0, 150.0]]
    shadow_boxes = [[50.0, 50.0, 200.0, 150.0], [300.0, 50.0, 450.0, 150.0]]

    active_det = DummyDetector(active_boxes)
    shadow_det = DummyDetector(shadow_boxes)

    evaluator = ShadowModelEvaluator(
        active_model_id=1,
        shadow_model_id=2,
        active_detector=active_det,
        shadow_detector=shadow_det,
        window_size=50,
    )

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(35):
        evaluator.evaluate_frame(img)

    summary = evaluator.get_comparison_summary()
    assert summary.frames_compared == 35
    assert summary.agreement_rate == 0.0
    assert summary.false_positives == 35
    assert summary.is_ready_for_promotion is False
