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
            orig_c = image[:, :, i]
            msr = self.process_channel(orig_c)
            msrcr = color_restoration[:, :, i] * msr
            norm = self._normalize_and_clip(msrcr, fallback_channel=orig_c)
            # Blend original + shadow-compensated Retinex to preserve model color space
            blended = cv2.addWeighted(orig_c, 0.7, norm, 0.3, 0)
            enhanced_channels.append(blended)

        return np.stack(enhanced_channels, axis=2)

    def _normalize_and_clip(self, retinex: np.ndarray, fallback_channel: np.ndarray | None = None) -> np.ndarray:
        """Statistical mean/std normalization (Jobson MSRCR standard)."""
        mean = float(np.mean(retinex))
        std = float(np.std(retinex))

        # If flat illumination / synthetic uniform background with no shadows
        if std < 0.5:
            return fallback_channel if fallback_channel is not None else np.uint8(np.clip(retinex, 0, 255))

        min_val = mean - 2.0 * std
        max_val = mean + 2.0 * std

        if max_val > min_val:
            stretched = (retinex - min_val) / (max_val - min_val) * 255.0
        else:
            stretched = retinex

        clipped = np.clip(stretched, 0, 255)
        return np.uint8(clipped)
