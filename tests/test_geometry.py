"""Unit tests for geometric primitives, projection on belt axis, and mask IoU."""

import numpy as np
import pytest
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


def test_polygon_centroid_asymmetric_area_weighted_not_vertex_average():
    """An L-shaped (asymmetric) polygon where the naive vertex average and the
    correct area-weighted (shoelace) centroid clearly disagree -- proves
    polygon_centroid computes the real centroid of the shape's area rather
    than just averaging vertex coordinates.

    The L-shape is a 10x10 square with a 5x5 square notch cut out of its
    top-right corner:

        (0,10)---(10,10)
          |    L-shape |
          |      (5,10)+---+(10,10)  <- notch removed, vertices below trace it
        (0,0)-----------(10,0)

    Concretely: [(0,0), (10,0), (10,5), (5,5), (5,10), (0,10)].
    """
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0), (5.0, 10.0), (0.0, 10.0)]

    naive_avg_x = sum(p[0] for p in polygon) / len(polygon)
    naive_avg_y = sum(p[1] for p in polygon) / len(polygon)

    cx, cy = polygon_centroid(polygon)

    # Correct area-weighted centroid of this L-shape (75 sq units of area:
    # a 10x10 square minus a 5x5 notch), computed independently via the
    # standard composite-shape formula (big square centroid (5,5) weight 100,
    # minus removed 5x5 notch centroid (7.5, 7.5) weight 25):
    #   cx = (100*5 - 25*7.5) / 75 = 4.1666...
    #   cy = (100*5 - 25*7.5) / 75 = 4.1666...
    expected_cx = (100.0 * 5.0 - 25.0 * 7.5) / 75.0
    expected_cy = (100.0 * 5.0 - 25.0 * 7.5) / 75.0

    assert abs(cx - expected_cx) < 1e-6
    assert abs(cy - expected_cy) < 1e-6

    # And this must clearly differ from the naive vertex average -- proving
    # the implementation isn't just averaging vertex coordinates.
    assert abs(cx - naive_avg_x) > 0.5
    assert abs(cy - naive_avg_y) > 0.5


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


def test_project_point_on_axis_degenerate_vector_raises():
    """A zero-length axis vector cannot define a projection axis -- must raise
    ValueError instead of silently substituting norm=1.0 (which would
    previously divide by "1.0" and return nonsense un-normalized components).
    """
    with pytest.raises(ValueError):
        project_point_on_axis((5.0, 5.0), axis_origin=(0.0, 0.0), axis_vector=(0.0, 0.0))


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


def test_compute_mask_iou_mismatched_shapes_resizes_instead_of_zero():
    """Masks decoded at different resolutions must be reconciled by resizing
    (nearest-neighbor) before comparing, not treated as automatically 0
    overlap -- a full-frame mask resized down/up should still report ~1.0
    IoU against an identical full-frame mask at a different resolution."""
    m1 = np.ones((50, 50), dtype=bool)
    m2 = np.ones((100, 100), dtype=bool)  # same full-coverage mask, different resolution
    assert compute_mask_iou(m1, m2) == 1.0

    # A mismatched-shape mask with genuinely different (non-overlapping)
    # content must NOT be forced to 0.0 outright -- it should reflect the
    # real overlap after resizing.
    m3 = np.zeros((50, 50), dtype=bool)
    m3[0:25, :] = True  # top half at 50x50 resolution
    m4 = np.zeros((100, 100), dtype=bool)
    m4[0:50, :] = True  # top half at 100x100 resolution (same region, different res)
    iou = compute_mask_iou(m3, m4)
    assert abs(iou - 1.0) < 0.05
