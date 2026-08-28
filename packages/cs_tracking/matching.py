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


def compute_cost_matrix(
    tracks: list[Any],
    detections: list[dict[str, Any]],
    w_mask: float = 0.70,
    w_dist: float = 0.30,
    max_distance_px: float = 150.0,
) -> np.ndarray:
    """Compute association cost matrix between existing tracks and new candidate detections.
    
    Cost is a weighted linear combination of Mask IoU distance (1 - IoU) and Euclidean distance.
    Bounding box IoU is explicitly avoided to prevent false associations during shingling.
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
        track_centroid = track.centroid  # (cx, cy)

        for j, det in enumerate(detections):
            det_mask = det.get("mask")
            det_box = det.get("box", [0, 0, 0, 0])
            det_centroid = (
                (det_box[0] + det_box[2]) / 2.0,
                (det_box[1] + det_box[3]) / 2.0,
            )

            # 1. Mask IoU cost
            #
            # When either side has no mask (e.g. the fallback bbox-only
            # detector path, or a brand-new track not yet assigned a mask),
            # we cannot measure IoU at all -- there is no "unknown" value in
            # a cost matrix, so this deliberately assumes the worst case
            # (iou=0.0 -> mask_cost=1.0) rather than silently guessing a
            # good match. Since w_mask=0.70 dominates the weighting below,
            # association in that situation is effectively driven almost
            # entirely by centroid distance alone (the w_dist=0.30 term),
            # not truly disabled -- just heavily penalized versus a real
            # mask-IoU match.
            if track_mask is not None and det_mask is not None:
                iou = compute_mask_iou(track_mask, det_mask)
            else:
                iou = 0.0
            mask_cost = 1.0 - iou

            # 2. Centroid distance cost
            dx = track_centroid[0] - det_centroid[0]
            dy = track_centroid[1] - det_centroid[1]
            dist = math.sqrt(dx * dx + dy * dy)
            norm_dist = min(1.0, dist / max_distance_px)

            cost = (w_mask * mask_cost) + (w_dist * norm_dist)
            cost_matrix[i, j] = cost

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
