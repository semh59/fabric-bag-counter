"""Geometry, polygon, ROI, gate projection, and mask IoU utilities."""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]] | Sequence[list[float]]) -> bool:
    """Ray-casting algorithm to test if a 2D point is inside a polygon."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def polygon_centroid(polygon: Sequence[tuple[float, float]] | Sequence[list[float]]) -> tuple[float, float]:
    """Calculate the area-weighted (shoelace) centroid of a polygon.

    This is the correct geometric centroid of the polygon's interior area --
    NOT the average of its vertices. The two only coincide for regular
    polygons (e.g. a square/rectangle); for an irregular polygon (e.g. one
    with a cluster of extra vertices along one edge) a plain vertex average
    is pulled toward the dense side and does not represent the centroid of
    mass, which is wrong for ROI/gate geometry that should be centered on
    the actual shape. See test_geometry.py for an asymmetric polygon where
    the two formulas clearly disagree.
    """
    n = len(polygon)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(polygon[0][0]), float(polygon[0][1])
    if n == 2:
        # Degenerate (zero-area) polygon: fall back to the midpoint.
        return (float(polygon[0][0]) + float(polygon[1][0])) / 2.0, (float(polygon[0][1]) + float(polygon[1][1])) / 2.0

    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    signed_area *= 0.5
    if signed_area == 0.0:
        # Degenerate (collinear/zero-area) polygon: fall back to vertex average.
        cx = sum(p[0] for p in polygon) / n
        cy = sum(p[1] for p in polygon) / n
        return float(cx), float(cy)

    cx /= 6.0 * signed_area
    cy /= 6.0 * signed_area
    return float(cx), float(cy)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Calculate center of mass (centroid) of a binary or boolean mask."""
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=bool)

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return 0.0, 0.0

    # coords gives (row, col) = (y, x)
    mean_y = float(np.mean(coords[:, 0]))
    mean_x = float(np.mean(coords[:, 1]))
    return mean_x, mean_y


def project_point_on_axis(
    point: tuple[float, float],
    axis_origin: tuple[float, float],
    axis_vector: tuple[float, float],
) -> float:
    """Project a point onto the 1D belt motion axis.
    
    Returns scalar distance along the axis direction from axis_origin.
    """
    vx, vy = axis_vector
    norm = math.sqrt(vx * vx + vy * vy)
    if norm == 0:
        raise ValueError(
            "axis_vector must be non-zero to project a point onto the belt motion axis "
            f"(got axis_vector={axis_vector!r})"
        )
    ux, uy = vx / norm, vy / norm

    dx = point[0] - axis_origin[0]
    dy = point[1] - axis_origin[1]
    return dx * ux + dy * uy


def _resize_mask_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize a 2D mask to `target_shape` (height, width).

    Mirrors the nearest-neighbor resampling used for segmentation masks in
    packages/cs_vision/postprocess.py (PIL `Image.Resampling.NEAREST`), but
    implemented with plain numpy indexing so cs_core does not need to
    depend on PIL/cv2 for this small utility.
    """
    src_h, src_w = mask.shape[:2]
    dst_h, dst_w = target_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return mask

    row_idx = np.clip((np.arange(dst_h) * src_h) // max(dst_h, 1), 0, src_h - 1)
    col_idx = np.clip((np.arange(dst_w) * src_w) // max(dst_w, 1), 0, src_w - 1)
    return mask[row_idx[:, None], col_idx]


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks.

    Masks of differing resolutions are reconciled by nearest-neighbor
    resizing `mask2` onto `mask1`'s shape before comparing -- this can
    legitimately happen when comparing masks decoded at different model
    input resolutions, and previously silently returned 0.0 (always "no
    overlap"), which is wrong and masks real matches.
    """
    if mask1.shape != mask2.shape:
        mask2 = _resize_mask_nearest(mask2, (mask1.shape[0], mask1.shape[1]))

    m1 = mask1 > 0
    m2 = mask2 > 0

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def compute_mask_iou_matrix(masks1: list[np.ndarray], masks2: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise mask IoU matrix of shape (len(masks1), len(masks2))."""
    n = len(masks1)
    m = len(masks2)
    iou_matrix = np.zeros((n, m), dtype=np.float32)

    if n == 0 or m == 0:
        return iou_matrix

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_mask_iou(masks1[i], masks2[j])

    return iou_matrix


def compute_box_iou(
    box1: Sequence[float] | np.ndarray,
    box2: Sequence[float] | np.ndarray,
) -> float:
    """Compute intersection-over-union between two axis-aligned boxes
    [x1, y1, x2, y2]."""
    xa = max(float(box1[0]), float(box2[0]))
    ya = max(float(box1[1]), float(box2[1]))
    xb = min(float(box1[2]), float(box2[2]))
    yb = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1])))
    area2 = max(0.0, (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1])))

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_polygon_area(polygon: Sequence[tuple[float, float]] | Sequence[list[float]]) -> float:
    """Compute polygon area using Shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2.0
