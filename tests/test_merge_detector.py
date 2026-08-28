"""Unit tests for MergeDetector and safe fallback without scale calibration (§6.7, §5.3)."""

import numpy as np

from packages.cs_tracking.merge_detector import MergeDetector


def test_merge_detector_safe_fallback_without_calibration():
    # When is_scale_calibrated is False, merge detector MUST safely disable itself
    detector = MergeDetector(mean_bag_gate_area_px=1000.0, is_scale_calibrated=False)
    mask = np.ones((50, 50), dtype=bool)  # area = 2500, normally exceeds 1.5 ratio
    hyp = detector.analyze_detection(mask=mask, box=[0, 0, 50, 50])
    assert hyp.is_merged is False
    assert "bypassed_no_calibration" in hyp.signal_votes


def test_merge_detector_with_calibration():
    # Active calibration with mean single bag area = 1000 px
    detector = MergeDetector(mean_bag_gate_area_px=1000.0, is_scale_calibrated=True, min_votes=2)

    # 1. Normal single bag
    single_mask = np.zeros((100, 100), dtype=bool)
    single_mask[10:40, 10:40] = True  # area = 900
    hyp_single = detector.analyze_detection(mask=single_mask, box=[10, 10, 40, 40])
    assert hyp_single.is_merged is False

    # 2. Two merged shingled bags (area 2200 px > 1500 px, plus elongated aspect ratio)
    merged_mask = np.zeros((100, 200), dtype=bool)
    merged_mask[20:50, 10:110] = True  # area = 3000
    hyp_merged = detector.analyze_detection(
        mask=merged_mask,
        box=[10, 20, 110, 50],
        print_marks=[
            {"box": [20, 25, 40, 45]},
            {"box": [80, 25, 100, 45]},
        ],
    )
    assert hyp_merged.is_merged is True
    assert hyp_merged.estimated_object_count == 2
    assert len(hyp_merged.centroid_seeds) == 2
    assert "signal_area_oversized" in hyp_merged.signal_votes
    assert "signal_multiple_print_marks" in hyp_merged.signal_votes


def test_signal_enable_flags_have_real_effect_on_vote_outcome():
    """merge_signals.{area,shape,temporal,print_mark}_enabled (see
    CountingEngine.configure()) must actually change whether a signal casts
    a vote, not just exist as an attribute -- reuses the exact scenario
    from test_merge_detector_with_calibration that trips both the area and
    print_mark signals (2 votes, >= min_votes=2 -> is_merged=True), and
    proves disabling both signals removes their votes and flips the
    outcome to not-merged."""
    detector = MergeDetector(
        mean_bag_gate_area_px=1000.0, is_scale_calibrated=True, min_votes=2,
        area_enabled=False, print_mark_enabled=False,
    )
    merged_mask = np.zeros((100, 200), dtype=bool)
    merged_mask[20:50, 10:110] = True
    hyp = detector.analyze_detection(
        mask=merged_mask,
        box=[10, 20, 110, 50],
        print_marks=[
            {"box": [20, 25, 40, 45]},
            {"box": [80, 25, 100, 45]},
        ],
    )
    assert "signal_area_oversized" not in hyp.signal_votes
    assert "signal_multiple_print_marks" not in hyp.signal_votes
    assert hyp.is_merged is False  # only signal_shape_convexity_deficit could still fire; below min_votes=2
