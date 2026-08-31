"""Unit tests for Side-View Conveyor Bag Inspector (§4.2, §6.3)."""

import numpy as np
import pytest
from packages.cs_vision.side_inspector import SideInspectionConfig, SideViewInspector


def test_side_inspector_nominal_single_bag():
    cfg = SideInspectionConfig(nominal_bag_thickness_px=60.0, double_stack_ratio_threshold=1.65, min_bag_width_px=100.0)
    inspector = SideViewInspector(config=cfg)

    img = np.zeros((300, 400), dtype=np.uint8)
    img[150:205, 100:300] = 220

    res = inspector.inspect_frame(img)
    assert res.is_double_stacked is False
    assert res.measured_thickness_px == pytest.approx(55.0, abs=4.0)
    assert len(res.detected_boxes) == 1


def test_side_inspector_detects_vertical_double_stacking_and_seam():
    cfg = SideInspectionConfig(nominal_bag_thickness_px=60.0, double_stack_ratio_threshold=1.65, min_bag_width_px=100.0)
    inspector = SideViewInspector(config=cfg)

    # 2 vertically stacked bags with horizontal seam line in the middle
    img = np.zeros((300, 400), dtype=np.uint8)
    img[80:140, 100:300] = 230   # Bag 1 (top)
    img[140:145, 100:300] = 30   # Dark seam groove valley
    img[145:205, 100:300] = 230  # Bag 2 (bottom)

    res = inspector.inspect_frame(img)
    assert res.is_double_stacked is True
    assert res.measured_thickness_px >= 115.0
    assert res.thickness_ratio >= 1.65
    assert res.anomaly_score > 0.0


def test_side_inspector_detects_powder_spillage_rupture():
    cfg = SideInspectionConfig(nominal_bag_thickness_px=60.0, spillage_area_threshold_px=200.0)
    inspector = SideViewInspector(config=cfg)

    img = np.zeros((300, 400), dtype=np.uint8)
    img[100:160, 100:300] = 220  # Normal bag
    img[240:270, 120:250] = 200  # Powder spillage leak puddle on belt bottom

    res = inspector.inspect_frame(img)
    assert res.is_ruptured is True
    assert res.leak_contour_area > 0.0
