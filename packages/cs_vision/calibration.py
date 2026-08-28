"""Real perspective (ROI-warp) camera calibration (§6.2, §5.3 Stage 3).

anchor_grid() in train_rfdetr.py hard-codes where the model expects the
belt's region of interest to sit in a 640x640 canonical frame -- a real,
documented limitation: a camera mounted at a different height/angle/framing
than the one the model was trained against will silently point every
anchor at the wrong part of the image.

This module computes and applies a real OpenCV perspective transform
(cv2.getPerspectiveTransform / cv2.warpPerspective) so a differently-framed
camera's raw belt ROI gets warped into that same canonical view before
detection -- the detector never has to know the physical camera framing
changed. It does not remove the need to recalibrate per camera; it replaces
"hand-edit anchor_grid() and retrain" with "mark 4 points once per camera."
"""

from __future__ import annotations

import numpy as np
import cv2

# Deliberately NOT imported from train_rfdetr.py: that module pulls in
# torch/onnx at module level (training/GPU-only deps, see the `vision`
# extra in pyproject.toml), and this module is imported by
# stream_renderer.py -- part of cs-api's live-detection path, which only
# has the lightweight `api-inference` extra (plain onnxruntime) installed.
# Importing train_rfdetr here would crash cs-api at import time. Must stay
# equal to train_rfdetr.CANVAS_SIZE (also a fixed (640, 640) literal, not
# computed) and to VisionDetector's default input_size.
CANVAS_SIZE = (640, 640)

# Canonical destination rectangle the 4 marked source points get warped
# onto -- the full canvas the model/anchor_grid() were trained against.
_DST_POINTS = np.array(
    [[0.0, 0.0], [CANVAS_SIZE[0], 0.0], [CANVAS_SIZE[0], CANVAS_SIZE[1]], [0.0, CANVAS_SIZE[1]]],
    dtype=np.float32,
)


def compute_homography(src_points: list[list[float]]) -> list[list[float]]:
    """Compute a real 3x3 perspective transform from 4 source points (any
    order the operator clicked them in, as long as it's consistently
    top-left/top-right/bottom-right/bottom-left) to the canonical canvas.

    Raises ValueError for degenerate input (not exactly 4 points, or 3+
    collinear points that OpenCV cannot invert) rather than silently
    returning an unusable matrix.
    """
    if len(src_points) != 4:
        raise ValueError(f"Perspective calibration needs exactly 4 points, got {len(src_points)}")

    src = np.array(src_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, _DST_POINTS)
    # cv2.getPerspectiveTransform solves a linear system that stays finite
    # even for collinear/duplicate points -- it just returns a near-singular
    # matrix (determinant ~0) that maps everything to a line or a point
    # instead of a real quadrilateral. np.isfinite alone does not catch
    # this; a near-zero determinant does.
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-6:
        raise ValueError("Computed homography is degenerate (points are likely collinear or duplicated)")
    return matrix.tolist()


def apply_perspective_warp(frame: np.ndarray, homography_matrix: list[list[float]]) -> np.ndarray:
    """Warp a raw camera frame into the canonical belt-ROI view."""
    matrix = np.array(homography_matrix, dtype=np.float32)
    return cv2.warpPerspective(frame, matrix, CANVAS_SIZE)
