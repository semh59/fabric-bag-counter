"""Cost matrix computation and Hungarian bipartite matching (§6.6)."""

# Source: Zhang et al., "ByteTrack" (ECCV 2022), arXiv:2110.06864
# Reference code: github.com/FoundationVision/ByteTrack (MIT)
# Novel part: Exact Mask-IoU and physical centroid distance matching replacing bbox-IoU

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from packages.cs_core.geometry import compute_mask_iou
from packages.cs_tracking.diou import compute_diou


def compute_cost_matrix(
    tracks: list[Any],
    detections: list[dict[str, Any]],
    w_mask: float = 0.70,
    w_dist: float = 0.30,
    max_distance_px: float = 150.0,
) -> np.ndarray:
    """Compute association cost matrix between existing tracks and new candidate detections.

    Cost combines Mask IoU distance (1 - IoU) or DIoU spatial centroid distance with Euclidean proximity.
    """
    n_tracks = len(tracks)
    n_dets = len(detections)
    # Default to maximum cost (1.0 = "no match") for every (track, detection)
    # pair. This is the safe starting assumption before any pair is actually
    # scored below, and it is also what a missing mask degrades to (see
    # mask_cost below) rather than an unfilled/undefined value.
    cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

    if n_tracks == 0 or n_dets == 0:
        return cost_matrix

    for i, track in enumerate(tracks):
        track_mask = track.mask
        track_box = track.box
        track_centroid = track.centroid  # (cx, cy)

        for j, det in enumerate(detections):
            det_mask = det.get("mask")
            det_box = det.get("box", [0, 0, 0, 0])
            det_centroid = (
                (det_box[0] + det_box[2]) / 2.0,
                (det_box[1] + det_box[3]) / 2.0,
            )

            # 1. Mask IoU or DIoU cost
            if track_mask is not None and det_mask is not None:
                mask_iou = compute_mask_iou(track_mask, det_mask)
                spatial_cost = 1.0 - mask_iou
            else:
                diou = compute_diou(track_box, det_box)
                # Map DIoU [-1, 1] to cost [0, 1]
                spatial_cost = (1.0 - diou) / 2.0

            # 2. Normalized Euclidean distance cost
            dx = track_centroid[0] - det_centroid[0]
            dy = track_centroid[1] - det_centroid[1]
            dist = math.sqrt(dx * dx + dy * dy)
            dist_cost = min(1.0, dist / max_distance_px)

            cost_matrix[i, j] = w_mask * spatial_cost + w_dist * dist_cost

    return cost_matrix


def associate_detections_to_tracks(
    tracks: list[Any],
    detections: list[dict[str, Any]],
    cost_threshold: float = 0.70,
    w_mask: float = 0.70,
    w_dist: float = 0.30,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate detections to active tracks using Hungarian linear sum assignment.

    w_mask/w_dist were previously silently ignored -- compute_cost_matrix()
    has always accepted them, but every caller (this function included)
    called it with no arguments, so the cost weighting was permanently
    stuck at its hardcoded defaults regardless of config
    (tracking_cost_weights.mask_iou/centroid_distance, see
    CountingEngine.configure()).

    Returns:
        (matches, unmatched_tracks, unmatched_detections)
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))
    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    cost_matrix = compute_cost_matrix(tracks, detections, w_mask=w_mask, w_dist=w_dist)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= cost_threshold:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_detections.discard(c)

    return matches, sorted(unmatched_tracks), sorted(unmatched_detections)
