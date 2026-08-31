"""Distance-IoU (DIoU) and Generalized-IoU (GIoU) Tracking Metrics (§6.2, §7.1).

Provides mathematically exact spatial overlap and centroid distance metrics for robust
data association in conveyor object tracking, eliminating ID switches across high-speed belts.
"""

from __future__ import annotations

import math
import numpy as np


def compute_diou(box1: list[float] | tuple[float, float, float, float] | np.ndarray,
                 box2: list[float] | tuple[float, float, float, float] | np.ndarray) -> float:
    """Compute Distance-IoU between two bounding boxes [x1, y1, x2, y2].

    DIoU = IoU - (d^2 / c^2)
    where:
    - d is the Euclidean distance between box centroids.
    - c is the diagonal length of the smallest enclosing box covering both boxes.
    """
    x1_min, y1_min, x1_max, y1_max = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
    x2_min, y2_min, x2_max, y2_max = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])

    # 1. Intersection Area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # 2. Union Area
    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    iou = (inter_area / union_area) if union_area > 0 else 0.0

    # 3. Centroid Euclidean Distance (d^2)
    c1_x = (x1_min + x1_max) / 2.0
    c1_y = (y1_min + y1_max) / 2.0
    c2_x = (x2_min + x2_max) / 2.0
    c2_y = (y2_min + y2_max) / 2.0
    d_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2

    # 4. Smallest Enclosing Box Diagonal (c^2)
    enc_xmin = min(x1_min, x2_min)
    enc_ymin = min(y1_min, y2_min)
    enc_xmax = max(x1_max, x2_max)
    enc_ymax = max(y1_max, y2_max)
    enc_w = enc_xmax - enc_xmin
    enc_h = enc_ymax - enc_ymin
    c_sq = (enc_w ** 2) + (enc_h ** 2)

    if c_sq <= 1e-6:
        return iou

    diou = iou - (d_sq / c_sq)
    return float(max(-1.0, min(1.0, diou)))


def compute_giou(box1: list[float] | tuple[float, float, float, float] | np.ndarray,
                 box2: list[float] | tuple[float, float, float, float] | np.ndarray) -> float:
    """Compute Generalized-IoU (GIoU) between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
    x2_min, y2_min, x2_max, y2_max = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    iou = (inter_area / union_area) if union_area > 0 else 0.0

    enc_xmin = min(x1_min, x2_min)
    enc_ymin = min(y1_min, y2_min)
    enc_xmax = max(x1_max, x2_max)
    enc_ymax = max(y1_max, y2_max)
    enc_area = max(0.0, enc_xmax - enc_xmin) * max(0.0, enc_ymax - enc_ymin)

    if enc_area <= 1e-6:
        return iou

    giou = iou - ((enc_area - union_area) / enc_area)
    return float(max(-1.0, min(1.0, giou)))


def compute_pairwise_diou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise DIoU matrix of shape (len(boxes_a), len(boxes_b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    matrix = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            matrix[i, j] = compute_diou(a, b)

    return matrix
