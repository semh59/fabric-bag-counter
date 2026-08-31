"""Unit tests for Brown-Conrady lens distortion and RANSAC rail detection."""

import numpy as np
import pytest
from packages.cs_vision.lens_distortion import (
    LensDistortionCorrector,
    LensDistortionParams,
    RansacConveyorRailDetector,
)


def test_lens_distortion_corrector_preserves_shape():
    params = LensDistortionParams(k1=-0.05, k2=0.01)
    corrector = LensDistortionCorrector(params)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:380, 100:540] = 200

    undistorted = corrector.undistort_image(img)
    assert undistorted.shape == (480, 640, 3)
    assert undistorted.dtype == np.uint8


def test_ransac_conveyor_rail_detector():
    detector = RansacConveyorRailDetector(max_iterations=100)

    # Synthetic image with 2 horizontal parallel lines representing conveyor rails
    img = np.zeros((480, 640), dtype=np.uint8)
    img[120, 50:590] = 255  # Upper rail
    img[360, 50:590] = 255  # Lower rail

    line1_pts, line2_pts, angle = detector.detect_rails(img)
    assert line1_pts is not None
    assert len(line1_pts) >= 20
