"""Unit tests for geometric primitives, projection on belt axis, and mask IoU."""

import numpy as np
from packages.cs_core.geometry import (
    compute_mask_iou,
    compute_mask_iou_matrix,
    compute_polygon_area,
    mask_centroid,
    point_in_polygon,
    polygon_centroid,
    project_point_on_axis,
)


def test_point_in_polygon():
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert point_in_polygon((5.0, 5.0), polygon) is True
    assert point_in_polygon((15.0, 5.0), polygon) is False
    assert point_in_polygon((-1.0, -1.0), polygon) is False


def test_polygon_centroid_and_area():
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    cx, cy = polygon_centroid(polygon)
    assert cx == 5.0 and cy == 5.0
    area = compute_polygon_area(polygon)
    assert area == 100.0


def test_mask_centroid():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 30:50] = True
    cx, cy = mask_centroid(mask)
    # col mean is 39.5, row mean is 29.5
    assert abs(cx - 39.5) < 0.1
    assert abs(cy - 29.5) < 0.1


def test_project_point_on_axis():
    # Horizontal belt axis (along +x)
    origin = (0.0, 100.0)
    vector = (1.0, 0.0)
    
    pos1 = project_point_on_axis((50.0, 100.0), origin, vector)
    pos2 = project_point_on_axis((120.0, 100.0), origin, vector)
    assert pos1 == 50.0
    assert pos2 == 120.0
    assert pos2 > pos1


def test_compute_mask_iou():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)

    m1[10:30, 10:30] = True
    m2[10:30, 10:30] = True
    assert compute_mask_iou(m1, m2) == 1.0

    m2 = np.zeros((50, 50), dtype=bool)
    m2[20:40, 10:30] = True
    # intersection: 10x20 = 200. union: 30x20 = 600. IoU = 200/600 = 1/3
    iou = compute_mask_iou(m1, m2)
    assert abs(iou - (1.0 / 3.0)) < 0.01
