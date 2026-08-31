"""Temporal Amodal Mask Reconstruction & Trajectory Prior Completion (§6.3, §6.6).

Reconstructs occluded / touching bag mask boundaries by projecting unoccluded prior shape
observations along the conveyor motion trajectory.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackMaskObservation:
    """Historical unoccluded mask observation of a bag on the conveyor."""

    frame_index: int
    centroid: tuple[float, float]
    box: list[float]
    area_px: float
    mask: np.ndarray


class TemporalAmodalReconstructor:
    """Reconstructs full amodal bag masks during contact/occlusion using motion priors."""

    def __init__(self, max_history_frames: int = 15) -> None:
        self.max_history = max_history_frames
        self._history: dict[int, deque[TrackMaskObservation]] = {}

    def record_observation(
        self,
        track_id: int,
        frame_index: int,
        box: list[float],
        mask: np.ndarray | None,
        is_isolated: bool = True,
    ) -> None:
        """Record an unoccluded mask observation for a track."""
        if mask is None or not is_isolated:
            return

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.max_history)

        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        area = float(np.sum(mask))

        obs = TrackMaskObservation(
            frame_index=frame_index,
            centroid=(cx, cy),
            box=list(box),
            area_px=area,
            mask=mask.copy(),
        )
        self._history[track_id].append(obs)

    def reconstruct_amodal_mask(
        self,
        track_id: int,
        current_box: list[float],
        current_visible_mask: np.ndarray | None,
        canvas_shape: tuple[int, int] = (640, 640),
    ) -> np.ndarray:
        """Reconstruct the full amodal mask using past unoccluded shape projected to current centroid."""
        if track_id not in self._history or len(self._history[track_id]) == 0:
            if current_visible_mask is not None:
                return current_visible_mask
            # Create synthetic bounding rectangle mask
            m = np.zeros(canvas_shape, dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in current_box]
            m[max(0, y1):min(canvas_shape[0], y2), max(0, x1):min(canvas_shape[1], x2)] = True
            return m

        # Retrieve best unoccluded reference observation
        best_obs = self._history[track_id][-1]
        ref_mask = best_obs.mask
        ref_cx, ref_cy = best_obs.centroid

        cur_cx = (current_box[0] + current_box[2]) / 2.0
        cur_cy = (current_box[1] + current_box[3]) / 2.0

        dx = cur_cx - ref_cx
        dy = cur_cy - ref_cy

        # Affine translation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = canvas_shape
        projected = cv2.warpAffine(np.uint8(ref_mask) * 255, M, (w, h), flags=cv2.INTER_NEAREST)

        # Merge projected prior with current visible mask if available
        if current_visible_mask is not None:
            combined = np.logical_or(projected > 127, current_visible_mask)
            return combined

        return projected > 127

    def cleanup_track(self, track_id: int) -> None:
        """Remove history for deleted track."""
        self._history.pop(track_id, None)
