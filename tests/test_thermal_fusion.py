"""Unit and integration tests for Radiometric Thermal IR Vision & Anomaly Detection (§6.2, §6.10)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.cs_counting.stream_renderer import LiveStreamRenderer
from packages.cs_vision.thermal_fusion import (
    MultiSpectralAligner,
    ThermalColorMap,
    ThermalVisionAnalyzer,
)
from services.api.main import app

client = TestClient(app)


def test_multispectral_aligner():
    aligner = MultiSpectralAligner(target_size=(640, 480))
    # Dummy low-res thermal sensor frame (160x120)
    low_res = np.full((120, 160), 35.0, dtype=np.float32)
    aligned = aligner.align(low_res)

    assert aligned.shape == (480, 640)
    assert np.isclose(aligned[240, 320], 35.0, atol=1.0)

    # Set 4 fiducial homography points
    src_pts = [(0, 0), (160, 0), (160, 120), (0, 120)]
    dst_pts = [(50, 50), (590, 50), (590, 430), (50, 430)]
    aligner.set_calibration_points(src_pts, dst_pts)
    warped = aligner.align(low_res)
    assert warped.shape == (480, 640)


def test_thermal_bag_analysis_normal():
    analyzer = ThermalVisionAnalyzer(normal_temp_range=(40.0, 80.0))
    box = [100.0, 100.0, 250.0, 300.0]
    # Create thermal frame with normal warm cement temperature (62°C)
    thermal_frame = analyzer.generate_synthetic_thermal_frame(
        canvas_size=(640, 640),
        bag_boxes=[box],
        bag_temp_c=62.0,
        inject_leak=False,
    )

    profile = analyzer.analyze_bag_temperature(thermal_frame, box, track_id=1)
    assert profile.track_id == 1
    assert 50.0 <= profile.mean_temp_c <= 70.0
    assert profile.is_normal is True
    assert len(profile.anomalies) == 0


def test_thermal_bag_analysis_hot_powder_leak():
    analyzer = ThermalVisionAnalyzer(normal_temp_range=(40.0, 80.0), leak_gradient_threshold_c=12.0)
    box = [100.0, 100.0, 250.0, 300.0]
    # Injected leak produces a localized temperature plume
    thermal_frame = analyzer.generate_synthetic_thermal_frame(
        canvas_size=(640, 640),
        bag_boxes=[box],
        bag_temp_c=65.0,
        inject_leak=True,
    )

    profile = analyzer.analyze_bag_temperature(thermal_frame, box, track_id=2)
    assert profile.is_normal is False
    assert len(profile.anomalies) >= 1
    leak = profile.anomalies[0]
    assert leak.anomaly_type == "hot_leak"
    assert leak.peak_temperature_c > profile.mean_temp_c


def test_thermal_heatmap_and_fusion():
    analyzer = ThermalVisionAnalyzer()
    thermal = np.full((300, 400), 55.0, dtype=np.float32)
    rgb = np.zeros((300, 400, 3), dtype=np.uint8)

    for cmap in (ThermalColorMap.INFERNO, ThermalColorMap.JET, ThermalColorMap.HOT):
        heatmap = analyzer.generate_thermal_heatmap(thermal, colormap=cmap)
        assert heatmap.shape == (300, 400, 3)
        assert heatmap.dtype == np.uint8

    fused = analyzer.fuse_rgb_and_thermal(rgb, thermal, alpha=0.5)
    assert fused.shape == (300, 400, 3)
    assert fused.dtype == np.uint8


def test_live_stream_renderer_thermal_toggle():
    renderer = LiveStreamRenderer(line_id=10)
    assert renderer.thermal_mode_enabled is False

    # Toggle ON
    state = renderer.toggle_thermal_mode(True)
    assert state is True
    assert renderer.thermal_mode_enabled is True

    frame = renderer.get_next_annotated_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape[0] > 0
    assert len(renderer.latest_thermal_profiles) > 0

    # Toggle OFF
    state_off = renderer.toggle_thermal_mode(False)
    assert state_off is False


def test_thermal_api_endpoints():
    # 1. Toggle endpoint
    res_toggle = client.post("/api/live/lines/1/thermal/toggle")
    assert res_toggle.status_code == 200
    assert "thermal_mode_enabled" in res_toggle.json()

    # 2. Stats endpoint
    res_stats = client.get("/api/live/lines/1/thermal/stats")
    assert res_stats.status_code == 200
    data = res_stats.json()
    assert "active_profiles" in data
    assert "thermal_mode_enabled" in data
