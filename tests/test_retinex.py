"""Unit tests for Multi-Scale Retinex with Color Restoration (MSRCR)."""

import numpy as np
import pytest
from packages.cs_vision.retinex import MultiScaleRetinex


def test_multiscale_retinex_grayscale():
    retinex = MultiScaleRetinex(scales=(10, 30, 80))

    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 180

    enhanced = retinex.enhance(img)
    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8
    assert np.max(enhanced) > 0


def test_multiscale_retinex_rgb_shadow_removal():
    retinex = MultiScaleRetinex(scales=(10, 30, 80))

    # RGB image with dark shadow region on the right
    img = np.ones((120, 120, 3), dtype=np.uint8) * 150
    img[:, 60:, :] = 40  # Shadow region

    enhanced = retinex.enhance(img)
    assert enhanced.shape == (120, 120, 3)
    assert enhanced.dtype == np.uint8
    # Retinex brings the dark shadow region closer to normalized illumination
    assert np.mean(enhanced[:, 60:, :]) > 40
