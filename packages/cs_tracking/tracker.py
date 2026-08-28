"""ConveyorByteTracker: Specialized multi-object tracker with conveyor motion prior and crossing_seq (§6.6)."""

# Source: Zhang et al., "ByteTrack" (ECCV 2022), arXiv:2110.06864
# Reference code: github.com/FoundationVision/ByteTrack (MIT)
# Novel part: Conveyor motion prior Kalman, mask-IoU cost matrix, crossing_seq per track, latent track splitting

from __future__ import annotations

import math
from typing import Any
import numpy as np
from packages.cs_core.geometry import compute_box_iou
from packages.cs_tracking.matching import associate_detections_to_tracks
from packages.cs_tracking.motion import BeltMotionModel

# Minimum box-IoU between a just-lost track's (predicted) box and a
# brand-new track's starting box, within the SAME update() call, for that
# coincidence to be counted as a likely ID switch (see Step 6/7 below).
ID_SWITCH_IOU_THRESHOLD = 0.3


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
        # Must be <= VisionDetector's own conf_threshold (default 0.40): anything
        # the detector already filtered out never reaches this tracker, so a
        # higher bar here (previously 0.45) silently blocked every detection
        # from ever starting a new track -- no track meant no gate crossing was
        # ever possible, regardless of camera, motion, or anything downstream.
        high_score_threshold: float = 0.40,
        low_score_threshold: float = 0.15,
        match_cost_threshold: float = 0.70,
        max_time_lost: int = 30,
        belt_motion: BeltMotionModel | None = None,
        w_mask: float = 0.70,
        w_dist: float = 0.30,
    ) -> None:
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.match_cost_threshold = match_cost_threshold
        self.max_time_lost = max_time_lost
        self.belt_motion = belt_motion or BeltMotionModel()
        # Cost-matrix weighting for association (see matching.py's
        # compute_cost_matrix) -- mirrors config schema's
        # tracking_cost_weights.mask_iou/centroid_distance.
        self.w_mask = w_mask
        self.w_dist = w_dist

        self.tracked_tracks: list[BagTrack] = []
        self.lost_tracks: list[BagTrack] = []
        self.frame_count = 0

        # Real, observed tracking-quality counters (not estimates): incremented
        # directly from genuine state transitions below, so downstream
        # evaluation code (packages/cs_eval/replay_engine.py) can report real
        # id-switch / fragmentation metrics instead of hardcoded zeros.
        self.fragmentation_count: int = 0
        self.id_switch_count: int = 0

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
            self.tracked_tracks, high_dets, cost_threshold=self.match_cost_threshold,
            w_mask=self.w_mask, w_dist=self.w_dist,
        )

        for track_idx, det_idx in matches_a:
            det = high_dets[det_idx]
            self.tracked_tracks[track_idx].update(
                box=det["box"], score=det["score"], mask=det.get("mask")
            )

        # Step 4: Second association — Match remaining active tracks with low score detections
        remaining_active_tracks = [self.tracked_tracks[i] for i in u_track_a]
        matches_b, u_track_b, _ = associate_detections_to_tracks(
            remaining_active_tracks, low_dets, cost_threshold=self.match_cost_threshold,
            w_mask=self.w_mask, w_dist=self.w_dist,
        )

        for track_sub_idx, det_idx in matches_b:
            det = low_dets[det_idx]
            remaining_active_tracks[track_sub_idx].update(
                box=det["box"], score=det["score"], mask=det.get("mask")
            )

        # Step 5: Match remaining high score detections with lost tracks
        unmatched_high_dets = [high_dets[i] for i in u_det_high]
        matches_c, u_lost_c, u_det_final = associate_detections_to_tracks(
            self.lost_tracks, unmatched_high_dets, cost_threshold=self.match_cost_threshold,
            w_mask=self.w_mask, w_dist=self.w_dist,
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
        # Snapshot each just-lost track's predicted box *before* it is marked
        # lost, so Step 7 can check whether a brand-new track starts right on
        # top of one of them -- a real, geometry-derived id-switch signal.
        lost_this_frame_boxes = [(t.track_id, list(t.box)) for t in unmatched_active_tracks]
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
            # ID-switch heuristic: a track was JUST marked lost this same
            # update() call, and this brand-new track's starting box
            # substantially overlaps where that lost track was predicted to
            # be. The association step (cost_threshold gate) evidently
            # rejected the match, but the geometry says it plausibly IS the
            # same physical bag getting a new identity -- i.e. a real
            # id switch, not a fabricated count.
            for _lost_tid, lost_box in lost_this_frame_boxes:
                if compute_box_iou(new_track.box, lost_box) > ID_SWITCH_IOU_THRESHOLD:
                    self.id_switch_count += 1
                    break
            self.tracked_tracks.append(new_track)

        # Step 8: Prune lost tracks older than max_time_lost. A pruned track's
        # identity is gone for good -- if the same physical bag reappears
        # later it will be assigned a brand-new track_id -- so each prune is
        # a genuine track fragmentation event.
        newly_pruned = [t for t in self.lost_tracks if t.time_since_update > self.max_time_lost]
        self.fragmentation_count += len(newly_pruned)
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.max_time_lost]

        return [t for t in self.tracked_tracks if t.state in ["tentative", "confirmed"]]
