"""Multi-Scale Retinex with Color Restoration (MSRCR) for Industrial Lighting (§6.1).

Eliminates harsh shadows, LED flicker, and ambient sunlight shifts across conveyor belts
via logarithmic illumination-reflectance decomposition and color constancy restoration.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MultiScaleRetinex:
    """Industrial Multi-Scale Retinex lighting normalizer."""

    def __init__(
        self,
        scales: Sequence[int] = (15, 80, 250),
        dynamic_clip_percent: float = 1.0,
        alpha: float = 125.0,
        beta: float = 46.0,
    ) -> None:
        self.scales = list(scales)
        self.weights = [1.0 / len(self.scales)] * len(self.scales)
        self.dynamic_clip_percent = dynamic_clip_percent
        self.alpha = alpha
        self.beta = beta

    def process_channel(self, channel: np.ndarray) -> np.ndarray:
        """Apply Multi-Scale Retinex decomposition to a single 2D float channel."""
        img_float = np.float32(channel) + 1.0
        log_img = np.log(img_float)

        retinex = np.zeros_like(log_img)
        for scale, weight in zip(self.scales, self.weights):
            ksize = int(scale * 3) | 1
            blur = cv2.GaussianBlur(img_float, (ksize, ksize), scale)
            retinex += weight * (log_img - np.log(blur + 1.0))

        return retinex

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance RGB or Grayscale image using MSR with color restoration."""
        if image is None or image.size == 0:
            return image

        is_gray = len(image.shape) == 2 or image.shape[2] == 1
        if is_gray:
            gray = image if len(image.shape) == 2 else image[:, :, 0]
            ret = self.process_channel(gray)
            return self._normalize_and_clip(ret)

        # Multi-channel color restoration
        img_f = np.float32(image) + 1.0
        sum_channels = np.sum(img_f, axis=2, keepdims=True)
        color_restoration = self.beta * (np.log(self.alpha * img_f) - np.log(sum_channels + 1.0))

        enhanced_channels = []
        for i in range(3):
            msr = self.process_channel(image[:, :, i])
            msrcr = color_restoration[:, :, i] * msr
            norm = self._normalize_and_clip(msrcr)
            enhanced_channels.append(norm)

        return np.stack(enhanced_channels, axis=2)

    def _normalize_and_clip(self, retinex: np.ndarray) -> np.ndarray:
        """Linear contrast stretching using percentiles."""
        low = np.percentile(retinex, self.dynamic_clip_percent)
        high = np.percentile(retinex, 100.0 - self.dynamic_clip_percent)

        if high > low:
            stretched = (retinex - low) / (high - low) * 255.0
        else:
            stretched = retinex * 255.0

        clipped = np.clip(stretched, 0, 255)
        return np.uint8(clipped)
