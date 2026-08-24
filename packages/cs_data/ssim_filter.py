"""SSIM-based frame deduplication to remove redundant conveyor idle frames (§11 M1)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity (SSIM) between two grayscale images."""
    if img1.shape != img2.shape:
        return 0.0

    # Convert to float64
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x_sq = np.var(x)
    sigma_y_sq = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)

    return float(numerator / denominator)


class SSIMFilter:
    """Filters video frames based on structural similarity against previous frame."""

    def __init__(self, ssim_threshold: float = 0.95, resize_dim: tuple[int, int] = (160, 120)) -> None:
        self.ssim_threshold = ssim_threshold
        self.resize_dim = resize_dim
        self.last_thumbnail: np.ndarray | None = None

    def _to_gray_thumb(self, image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(image).convert("L")
        resized = pil_img.resize(self.resize_dim, Image.Resampling.BILINEAR)
        return np.array(resized)

    def is_redundant(self, image: np.ndarray) -> bool:
        """Check if frame is redundant (SSIM >= threshold). Updates internal reference if unique."""
        thumb = self._to_gray_thumb(image)
        if self.last_thumbnail is None:
            self.last_thumbnail = thumb
            return False

        ssim_val = compute_ssim_simple(self.last_thumbnail, thumb)
        if ssim_val >= self.ssim_threshold:
            return True  # Redundant / conveyor stationary

        self.last_thumbnail = thumb
        return False
