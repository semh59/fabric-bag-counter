"""Unit tests for DIoU and GIoU tracking metrics."""

import numpy as np
import pytest
from packages.cs_tracking.diou import compute_diou, compute_giou, compute_pairwise_diou_matrix


def test_compute_diou_identical_boxes():
    box1 = [100.0, 100.0, 200.0, 200.0]
    box2 = [100.0, 100.0, 200.0, 200.0]
    # IoU=1.0, distance=0.0 -> DIoU=1.0
    assert compute_diou(box1, box2) == pytest.approx(1.0, rel=1e-4)
    assert compute_giou(box1, box2) == pytest.approx(1.0, rel=1e-4)


def test_compute_diou_non_overlapping_boxes():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 0.0, 30.0, 10.0]
    # IoU=0.0, centroids at (5,5) and (25,5), distance=20, enc_box=[0,0,30,10] (diag=sqrt(900+100)=sqrt(1000))
    # DIoU = 0 - 400/1000 = -0.4
    diou = compute_diou(box1, box2)
    assert diou < 0.0
    assert diou == pytest.approx(-0.4, rel=1e-4)


def test_compute_pairwise_diou_matrix():
    boxes_a = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    boxes_b = np.array([[0.0, 0.0, 10.0, 10.0]])

    matrix = compute_pairwise_diou_matrix(boxes_a, boxes_b)
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == pytest.approx(1.0, rel=1e-4)
    assert matrix[1, 0] < 0.0
