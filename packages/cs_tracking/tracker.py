"""ConveyorByteTracker: Specialized multi-object tracker with conveyor motion prior and crossing_seq (§6.6)."""

# Source: Zhang et al., "ByteTrack" (ECCV 2022), arXiv:2110.06864
# Reference code: github.com/FoundationVision/ByteTrack (MIT)
# Novel part: Conveyor motion prior Kalman, mask-IoU cost matrix, crossing_seq per track, latent track splitting

from __future__ import annotations

import math
from typing import Any
import numpy as np
from packages.cs_tracking.matching import associate_detections_to_tracks
from packages.cs_tracking.motion import BeltMotionModel


class BagTrack:
    """Represents a single continuous bag track across video frames."""

    _count = 0

    def __init__(
        self,
        box: list[float],
        score: float,
        mask: np.ndarray | None = None,
        is_latent: bool = False,
    ) -> None:
        BagTrack._count += 1
        self.track_id: int = BagTrack._count
        self.box = list(box)
        self.score = float(score)
        self.mask = mask
        self.is_latent = is_latent
        self.merge_flag: bool = is_latent

        self.centroid: tuple[float, float] = (
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
        )
        self.velocity = np.zeros(2, dtype=np.float32)  # [vx, vy]

        self.state: str = "tentative"  # tentative | confirmed | lost | deleted
        self.age: int = 1
        self.hits: int = 1
        self.time_since_update: int = 0
        self.history: list[tuple[float, float]] = [self.centroid]

        # Monotonically increasing crossing counter per track (§5.5)
        self.crossing_seq: int = 0

    def predict(self, belt_velocity_prior: np.ndarray | None = None) -> None:
        """Predict next position using velocity and conveyor prior."""
        self.age += 1
        self.time_since_update += 1

        vx, vy = self.velocity[0], self.velocity[1]
        if belt_velocity_prior is not None and (abs(belt_velocity_prior[0]) > 0.01 or abs(belt_velocity_prior[1]) > 0.01):
            # Blend velocity with conveyor prior
            vx = 0.5 * vx + 0.5 * belt_velocity_prior[0]
            vy = 0.5 * vy + 0.5 * belt_velocity_prior[1]

        new_cx = self.centroid[0] + vx
        new_cy = self.centroid[1] + vy
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]

        self.centroid = (new_cx, new_cy)
        self.box = [new_cx - w / 2, new_cy - h / 2, new_cx + w / 2, new_cy + h / 2]

    def update(self, box: list[float], score: float, mask: np.ndarray | None = None) -> None:
        """Update track with newly associated measurement."""
        new_cx = (box[0] + box[2]) / 2.0
        new_cy = (box[1] + box[3]) / 2.0

        # Estimate observed velocity
        self.velocity[0] = 0.7 * (new_cx - self.centroid[0]) + 0.3 * self.velocity[0]
        self.velocity[1] = 0.7 * (new_cy - self.centroid[1]) + 0.3 * self.velocity[1]

        self.centroid = (new_cx, new_cy)
        self.box = list(box)
        self.score = float(score)
        if mask is not None:
            self.mask = mask

        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.centroid)

        if self.state == "tentative" and self.hits >= 2:
            self.state = "confirmed"
        elif self.state == "lost":
            self.state = "confirmed"

    def mark_lost(self) -> None:
        self.state = "lost"

    def mark_deleted(self) -> None:
        self.state = "deleted"

    @classmethod
    def reset_counter(cls) -> None:
        cls._count = 0


class ConveyorByteTracker:
    """Specialized ByteTrack implementation for shingled bags on conveyors."""

    def __init__(
        self,
        high_score_threshold: float = 0.45,
        low_score_threshold: float = 0.15,
        match_cost_threshold: float = 0.70,
        max_time_lost: int = 30,
        belt_motion: BeltMotionModel | None = None,
    ) -> None:
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.match_cost_threshold = match_cost_threshold
        self.max_time_lost = max_time_lost
        self.belt_motion = belt_motion or BeltMotionModel()

        self.tracked_tracks: list[BagTrack] = []
        self.lost_tracks: list[BagTrack] = []
        self.frame_count = 0

    def update(
        self,
        detections: list[dict[str, Any]],  # list of {"box": [...], "score": float, "mask": np.ndarray}
    ) -> list[BagTrack]:
        """Update tracker with detections for the current frame."""
        self.frame_count += 1
        belt_prior = self.belt_motion.get_velocity_prior()

        # Step 1: Predict positions of existing active tracks
        for t in self.tracked_tracks:
            t.predict(belt_prior)
        for t in self.lost_tracks:
            t.predict(belt_prior)

        # Step 2: Split detections into high score and low score sets
        high_dets = [d for d in detections if d.get("score", 0.0) >= self.high_score_threshold]
        low_dets = [d for d in detections if self.low_score_threshold <= d.get("score", 0.0) < self.high_score_threshold]

        # Step 3: First association — Match active tracks with high score detections
        matches_a, u_track_a, u_det_high = associate_detections_to_tracks(
            self.tracked_tracks, high_dets, cost_threshold=self.match_cost_threshold
        )

        for track_idx, det_idx in matches_a:
            det = high_dets[det_idx]
            self.tracked_tracks[track_idx].update(
                box=det["box"], score=det["score"], mask=det.get("mask")
            )

        # Step 4: Second association — Match remaining active tracks with low score detections
        remaining_active_tracks = [self.tracked_tracks[i] for i in u_track_a]
        matches_b, u_track_b, _ = associate_detections_to_tracks(
            remaining_active_tracks, low_dets, cost_threshold=self.match_cost_threshold
        )

        for track_sub_idx, det_idx in matches_b:
            det = low_dets[det_idx]
            remaining_active_tracks[track_sub_idx].update(
                box=det["box"], score=det["score"], mask=det.get("mask")
            )

        # Step 5: Match remaining high score detections with lost tracks
        unmatched_high_dets = [high_dets[i] for i in u_det_high]
        matches_c, u_lost_c, u_det_final = associate_detections_to_tracks(
            self.lost_tracks, unmatched_high_dets, cost_threshold=self.match_cost_threshold
        )

        for lost_idx, det_idx in matches_c:
            det = unmatched_high_dets[det_idx]
            matched_track = self.lost_tracks[lost_idx]
            matched_track.update(box=det["box"], score=det["score"], mask=det.get("mask"))
            self.tracked_tracks.append(matched_track)

        # Remove resurrected tracks from lost list
        resurrected = {self.lost_tracks[lost_idx].track_id for lost_idx, _ in matches_c}
        self.lost_tracks = [t for t in self.lost_tracks if t.track_id not in resurrected]

        # Step 6: Transition unmatched tracks to lost
        unmatched_active_tracks = [remaining_active_tracks[i] for i in u_track_b]
        for t in unmatched_active_tracks:
            t.mark_lost()
            self.lost_tracks.append(t)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state in ["tentative", "confirmed"]]

        # Step 7: Create new tracks from remaining unmatched high detections
        for det_idx in u_det_final:
            det = unmatched_high_dets[det_idx]
            new_track = BagTrack(
                box=det["box"],
                score=det["score"],
                mask=det.get("mask"),
                is_latent=det.get("is_latent", False),
            )
            self.tracked_tracks.append(new_track)

        # Step 8: Prune lost tracks older than max_time_lost
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.max_time_lost]

        return [t for t in self.tracked_tracks if t.state in ["tentative", "confirmed"]]
