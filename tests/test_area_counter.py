"""Unit tests for AreaIntegralCounter (§6.9, §5.3)."""

import numpy as np
from packages.cs_counting.area_counter import AreaIntegralCounter


def test_area_counter_safe_fallback_without_calibration():
    counter = AreaIntegralCounter(mean_bag_gate_area_px=1000.0, is_scale_calibrated=False)
    masks = [np.ones((40, 40), dtype=bool)]
    est = counter.process_frame_masks(masks)
    assert est == 0.0
    has_disc, _ = counter.check_discrepancy(ledger_count=50)
    assert has_disc is False


def test_area_counter_accumulation_and_discrepancy():
    counter = AreaIntegralCounter(
        mean_bag_gate_area_px=1000.0,
        discrepancy_threshold=0.08,
        is_scale_calibrated=True,
    )

    # Accumulate area from 100 frames with 1 bag each (area = 1000 px per frame)
    bag_mask = np.ones((25, 40), dtype=bool)  # 1000 px
    for _ in range(200):
        counter.process_frame_masks([bag_mask], belt_speed_px_per_frame=5.0)

    est = counter.get_estimate()
    assert est > 0.0

    # 1. Matching ledger count (within 8%): no discrepancy
    has_disc_match, delta_match = counter.check_discrepancy(ledger_count=int(round(est)))
    assert has_disc_match is False

    # 2. Severely mismatched ledger count (e.g. 50% delta): triggers discrepancy
    has_disc_mismatch, delta_mismatch = counter.check_discrepancy(ledger_count=int(round(est * 1.5)))
    assert has_disc_mismatch is True
    assert delta_mismatch > 0.08
