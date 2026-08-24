"""Preprocessing and image normalization for RF-DETR Seg inference."""

from __future__ import annotations

import numpy as np
from PIL import Image


def letterbox_image(
    image: np.ndarray,
    target_shape: tuple[int, int] = (640, 640),
    fill_value: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image to target shape maintaining aspect ratio with padding.
    
    Returns:
        (padded_image, scale_factor, (pad_w, pad_h))
    """
    h, w = image.shape[:2]
    target_h, target_w = target_shape

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # Resize using PIL
    pil_img = Image.fromarray(image)
    resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil)

    # Pad
    if len(image.shape) == 3:
        padded = np.full((target_h, target_w, image.shape[2]), fill_value, dtype=image.dtype)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w, :] = resized_arr
    else:
        padded = np.full((target_h, target_w), fill_value, dtype=image.dtype)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_arr

    return padded, scale, (pad_w, pad_h)


def preprocess_image(
    image: np.ndarray,
    target_shape: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Standardize image to NCHW float32 tensor normalized to [0, 1]."""
    padded, scale, (pad_w, pad_h) = letterbox_image(image, target_shape)
    
    # HWC -> CHW, normalized [0, 1]
    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)  # (1, C, H, W)
    return blob, scale, (pad_w, pad_h)
