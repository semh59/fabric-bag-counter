"""Hard frame mining and active learning (§6.10)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
from packages.cs_core.geometry import compute_box_iou

# IoU threshold below which a primary/secondary detection pair is considered
# spatially unmatched (i.e. a genuine spatial disagreement, not just the same
# bag detected at a slightly different box).
SPATIAL_DISAGREEMENT_IOU_THRESHOLD = 0.5


def _count_unmatched(
    dets_a: list[dict[str, Any]],
    dets_b: list[dict[str, Any]],
    iou_threshold: float,
) -> int:
    """Count how many detections in `dets_a` have no best-IoU match in `dets_b`
    at or above `iou_threshold` (greedy nearest-box matching, one-directional).
    """
    unmatched = 0
    for da in dets_a:
        box_a = da.get("box")
        if box_a is None:
            unmatched += 1
            continue
        best_iou = 0.0
        for db in dets_b:
            box_b = db.get("box")
            if box_b is None:
                continue
            iou = compute_box_iou(box_a, box_b)
            if iou > best_iou:
                best_iou = iou
        if best_iou < iou_threshold:
            unmatched += 1
    return unmatched


class MiningCriterion(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    MODEL_DISAGREEMENT = "model_disagreement"
    LEDGER_AREA_MISMATCH = "ledger_area_mismatch"
    MERGE_FLAG_ACTIVE = "merge_flag_active"
    TRACK_FRAGMENTATION = "track_fragmentation"
    HUMAN_CORRECTION = "human_correction"


@dataclass
class HardFrameCandidate:
    frame_index: int
    camera_id: int
    session_id: int
    criterion: MiningCriterion
    score: float
    metadata: dict[str, Any]


class HardFrameMiner:
    """Collects high-value, ambiguous, or error-prone frames for CVAT relabeling."""

    def __init__(
        self,
        low_conf_threshold: float = 0.50,
        max_samples_per_session: int = 200,
    ) -> None:
        self.low_conf_threshold = low_conf_threshold
        self.max_samples = max_samples_per_session
        # Running count of candidates already mined per session, so
        # max_samples_per_session is enforced for real across repeated calls
        # to evaluate_frame() on this miner instance (e.g. when a caller
        # iterates evaluate_frame() over every frame of a session).
        self._session_sample_counts: dict[Any, int] = {}

    def samples_mined_for_session(self, session_id: Any) -> int:
        """Number of hard-frame candidates already mined for `session_id` so far."""
        return self._session_sample_counts.get(session_id, 0)

    def evaluate_frame(
        self,
        frame_index: int,
        camera_id: int,
        session_id: int,
        detections: list[dict[str, Any]],
        secondary_model_detections: list[dict[str, Any]] | None = None,
        has_merge_flag: bool = False,
        area_mismatch: bool = False,
    ) -> list[HardFrameCandidate]:
        """Evaluate a processed frame against hard frame mining criteria."""
        # Enforce max_samples_per_session: once this session has already had
        # `max_samples` candidates mined (across prior calls to this same
        # miner instance), stop mining more frames for it.
        if self._session_sample_counts.get(session_id, 0) >= self.max_samples:
            return []

        candidates = []

        # 1. Check for low confidence detections -- consider ALL detections in
        # the frame, not just the first one found, so every qualifying box is
        # surfaced as its own candidate.
        for d in detections:
            score = d.get("score", 1.0)
            if score < self.low_conf_threshold:
                candidates.append(
                    HardFrameCandidate(
                        frame_index=frame_index,
                        camera_id=camera_id,
                        session_id=session_id,
                        criterion=MiningCriterion.LOW_CONFIDENCE,
                        score=score,
                        metadata={"box": d.get("box")},
                    )
                )

        # 2. Check for multi-model disagreement: both a COUNT mismatch and a
        # genuine SPATIAL mismatch (same count, but boxes don't line up, e.g.
        # one model's detection has no reasonable-IoU counterpart in the
        # other's) count as disagreement. A count-only check misses the
        # common case where both models report N bags but disagree on where
        # they are.
        if secondary_model_detections is not None:
            count1 = len(detections)
            count2 = len(secondary_model_detections)
            count_mismatch = count1 != count2

            unmatched_primary = _count_unmatched(
                detections, secondary_model_detections, SPATIAL_DISAGREEMENT_IOU_THRESHOLD
            )
            unmatched_secondary = _count_unmatched(
                secondary_model_detections, detections, SPATIAL_DISAGREEMENT_IOU_THRESHOLD
            )
            spatial_mismatch = unmatched_primary > 0 or unmatched_secondary > 0

            if count_mismatch or spatial_mismatch:
                candidates.append(
                    HardFrameCandidate(
                        frame_index=frame_index,
                        camera_id=camera_id,
                        session_id=session_id,
                        criterion=MiningCriterion.MODEL_DISAGREEMENT,
                        score=1.0,
                        metadata={
                            "primary_count": count1,
                            "secondary_count": count2,
                            "count_mismatch": count_mismatch,
                            "spatial_mismatch": spatial_mismatch,
                            "unmatched_primary": unmatched_primary,
                            "unmatched_secondary": unmatched_secondary,
                            "iou_threshold": SPATIAL_DISAGREEMENT_IOU_THRESHOLD,
                        },
                    )
                )

        # 3. Merge event active
        if has_merge_flag:
            candidates.append(
                HardFrameCandidate(
                    frame_index=frame_index,
                    camera_id=camera_id,
                    session_id=session_id,
                    criterion=MiningCriterion.MERGE_FLAG_ACTIVE,
                    score=0.9,
                    metadata={"reason": "merge_hypothesis_triggered"},
                )
            )

        # 4. Ledger vs Area Mismatch window
        if area_mismatch:
            candidates.append(
                HardFrameCandidate(
                    frame_index=frame_index,
                    camera_id=camera_id,
                    session_id=session_id,
                    criterion=MiningCriterion.LEDGER_AREA_MISMATCH,
                    score=0.95,
                    metadata={"reason": "area_integral_deviation"},
                )
            )

        # Apply the per-session cap for real: trim (rather than reject
        # outright) so a frame that would only partially exceed the
        # remaining quota still contributes up to the limit, then record how
        # many candidates this session has now accumulated in total.
        already_mined = self._session_sample_counts.get(session_id, 0)
        remaining_quota = max(0, self.max_samples - already_mined)
        if len(candidates) > remaining_quota:
            candidates = candidates[:remaining_quota]
        self._session_sample_counts[session_id] = already_mined + len(candidates)

        return candidates
