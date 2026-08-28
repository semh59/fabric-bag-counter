"""Tests for CountingEngine.configure() (§5.4) -- the real wiring between
the config-versioning system (ConfigRepository/ConfigVersionORM) and the
live detection pipeline.

Before this, CountingEngine() was constructed with zero arguments in both
real pipelines (LiveStreamRenderer, InferenceWorker) and always ran on
hardcoded defaults -- an activated bundle's confidence_threshold/
merge_area_ratio/roi_polygon/etc. never reached a running engine.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from packages.cs_counting.engine import CANVAS_SIZE, CountingEngine, _denormalize_roi
from packages.cs_vision.detector import DetectionResult


def test_configure_applies_all_wireable_thresholds():
    engine = CountingEngine()
    engine.configure({
        "confidence_threshold": 0.72,
        "merge_area_ratio": 2.1,
        "discrepancy_threshold": 0.22,
        "merge_signals": {"min_votes": 3},
        "mask_iou_threshold": 0.61,
    })

    assert engine.detector.conf_threshold == 0.72
    assert engine.detector.merge_area_ratio == 2.1
    assert engine.merge_detector.merge_area_ratio == 2.1
    assert engine.area_counter.discrepancy_threshold == 0.22
    assert engine.merge_detector.min_votes == 3
    assert engine.tracker.match_cost_threshold == 0.61


def test_configure_partial_payload_only_touches_given_keys():
    engine = CountingEngine()
    original_ratio = engine.detector.merge_area_ratio
    engine.configure({"confidence_threshold": 0.9})

    assert engine.detector.conf_threshold == 0.9
    assert engine.detector.merge_area_ratio == original_ratio  # untouched


def test_denormalize_roi_full_frame_default_is_none():
    full_frame = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert _denormalize_roi(full_frame) is None
    assert _denormalize_roi(None) is None
    assert _denormalize_roi([]) is None


def test_denormalize_roi_custom_polygon_scales_to_canvas():
    poly = _denormalize_roi([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    w, h = CANVAS_SIZE
    assert poly == [(0.1 * w, 0.1 * h), (0.9 * w, 0.1 * h), (0.9 * w, 0.9 * h), (0.1 * w, 0.9 * h)]


def test_configure_no_roi_leaves_engine_unfiltered():
    engine = CountingEngine()
    assert engine._roi_polygon_px is None
    engine.configure({})
    assert engine._roi_polygon_px is None


def test_process_frame_roi_filters_detections_outside_polygon(monkeypatch):
    """Real end-to-end proof: a detection whose box centroid is outside the
    configured ROI never reaches the tracker/ledger, using the real
    point_in_polygon math (packages/cs_core/geometry.py), not a mock of the
    filtering itself."""
    engine = CountingEngine()
    # Restrict counting to the left half of the canvas.
    engine.configure({"roi_polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]})

    inside_box = [50.0, 50.0, 150.0, 150.0]   # centroid (100, 100) -- inside left half
    outside_box = [500.0, 50.0, 600.0, 150.0]  # centroid (550, 100) -- outside left half

    def fake_predict(image):
        return DetectionResult(
            bag_bodies=[
                {"box": inside_box, "score": 0.95, "mask": None},
                {"box": outside_box, "score": 0.95, "mask": None},
            ],
            print_marks=[],
        )

    monkeypatch.setattr(engine.detector, "predict", fake_predict)

    out = engine.process_frame(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        frame_index=1,
        monotonic_ns=1,
        wall_clock=datetime.now(UTC),
    )

    assert len(out.detections.bag_bodies) == 1
    assert out.detections.bag_bodies[0]["box"] == inside_box


def test_configure_confidence_threshold_reaches_real_postprocessing_call(monkeypatch):
    """Proves the wiring goes all the way to the real consumer, not just
    that an attribute got set: detector.predict() must pass the
    *current* self.conf_threshold into postprocess_rfdetr_seg on every
    call, not a value captured once at construction time. Patches
    postprocess_rfdetr_seg itself (not predict()) so this only requires a
    real ONNX session, not a controllable/predictable real confidence
    output on a synthetic image."""
    import packages.cs_vision.detector as detector_module

    engine = CountingEngine()
    if engine.detector.session is None:
        pytest.skip("No real ONNX session available in this environment (fallback mode)")

    engine.configure({"confidence_threshold": 0.83})

    captured = {}
    real_postprocess = detector_module.postprocess_rfdetr_seg

    def spy_postprocess(*args, **kwargs):
        captured["conf_threshold"] = kwargs.get("conf_threshold")
        return real_postprocess(*args, **kwargs)

    monkeypatch.setattr(detector_module, "postprocess_rfdetr_seg", spy_postprocess)

    engine.detector.predict(np.zeros((640, 640, 3), dtype=np.uint8))

    assert captured["conf_threshold"] == 0.83


def test_process_frame_without_roi_keeps_all_detections(monkeypatch):
    engine = CountingEngine()
    boxes = [[50.0, 50.0, 150.0, 150.0], [500.0, 50.0, 600.0, 150.0]]

    def fake_predict(image):
        return DetectionResult(
            bag_bodies=[{"box": b, "score": 0.95, "mask": None} for b in boxes],
            print_marks=[],
        )

    monkeypatch.setattr(engine.detector, "predict", fake_predict)

    out = engine.process_frame(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        frame_index=1,
        monotonic_ns=1,
        wall_clock=datetime.now(UTC),
    )

    assert len(out.detections.bag_bodies) == 2
