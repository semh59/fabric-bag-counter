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
    """Calculate the 2D centroid of a polygon."""
    n = len(polygon)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(polygon[0][0]), float(polygon[0][1])

    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
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
        norm = 1.0
    ux, uy = vx / norm, vy / norm

    dx = point[0] - axis_origin[0]
    dy = point[1] - axis_origin[1]
    return dx * ux + dy * uy


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute exact intersection-over-union between two binary masks."""
    if mask1.shape != mask2.shape:
        # Resize or pad if necessary
        return 0.0

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
