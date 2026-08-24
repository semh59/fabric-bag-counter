"""Hard frame mining and active learning (§6.10)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np


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
        candidates = []

        # 1. Check for low confidence detections
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
                break

        # 2. Check for multi-model disagreement
        if secondary_model_detections is not None:
            count1 = len(detections)
            count2 = len(secondary_model_detections)
            if count1 != count2:
                candidates.append(
                    HardFrameCandidate(
                        frame_index=frame_index,
                        camera_id=camera_id,
                        session_id=session_id,
                        criterion=MiningCriterion.MODEL_DISAGREEMENT,
                        score=1.0,
                        metadata={"primary_count": count1, "secondary_count": count2},
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

        return candidates
