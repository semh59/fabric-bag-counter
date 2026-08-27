"""CountingEngine: Complete synchronized vision, tracking, and counting pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import numpy as np
from packages.cs_core.frame import Frame
from packages.cs_counting.area_counter import AreaIntegralCounter
from packages.cs_counting.gate import GateCrossingEvent, GateStateMachine
from packages.cs_tracking.merge_detector import MergeDetector
from packages.cs_tracking.motion import BeltMotionModel
from packages.cs_tracking.tracker import BagTrack, ConveyorByteTracker
from packages.cs_vision.detector import DetectionResult, VisionDetector


@dataclass
class FrameProcessingOutput:
    """Output for a single evaluated video frame."""
    frame_index: int
    monotonic_ns: int
    wall_clock: datetime
    detections: DetectionResult
    active_tracks: list[BagTrack]
    gate_crossings: list[GateCrossingEvent]
    running_net_count: int
    area_estimate: float
    discrepancy_flag: bool = False
    merge_events_in_frame: int = 0
    merge_extra_bags_in_frame: int = 0


class CountingEngine:
    """End-to-end counting engine running offline or within live inference worker."""

    def __init__(
        self,
        detector: VisionDetector | None = None,
        tracker: ConveyorByteTracker | None = None,
        merge_detector: MergeDetector | None = None,
        gate_state_machine: GateStateMachine | None = None,
        area_counter: AreaIntegralCounter | None = None,
        belt_motion: BeltMotionModel | None = None,
    ) -> None:
        self.belt_motion = belt_motion or BeltMotionModel()
        self.detector = detector or VisionDetector()
        self.tracker = tracker or ConveyorByteTracker(belt_motion=self.belt_motion)
        self.merge_detector = merge_detector or MergeDetector()
        self.gate_state_machine = gate_state_machine or GateStateMachine()
        self.area_counter = area_counter or AreaIntegralCounter()

        self.running_net_count = 0
        self.total_forward_crossings = 0
        self.total_backward_crossings = 0

    def reset_session(self) -> None:
        """Reset internal states for a fresh session."""
        self.running_net_count = 0
        self.total_forward_crossings = 0
        self.total_backward_crossings = 0
        self.area_counter.reset()
        BagTrack.reset_counter()

    def process_frame(
        self,
        image: np.ndarray,
        frame_index: int,
        monotonic_ns: int,
        wall_clock: datetime,
    ) -> FrameProcessingOutput:
        """Process a single image frame through the full vision-tracking-counting pipeline."""
        # 1. Vision Detection (RF-DETR Seg)
        detection_result = self.detector.predict(image)

        # 2. Merge Detection & Hypothesis analysis
        enriched_detections: list[dict[str, Any]] = []
        merge_count = 0
        merge_extra_bags = 0

        for bag in detection_result.bag_bodies:
            box = bag["box"]
            mask = bag.get("mask")
            hypothesis = self.merge_detector.analyze_detection(
                mask=mask,
                box=box,
                print_marks=detection_result.print_marks,
            )
            if hypothesis.is_merged:
                merge_count += 1
                # If two bags merged, register latent tracks
                cnt = max(2, hypothesis.estimated_object_count)
                for seed in hypothesis.centroid_seeds:
                    w = (box[2] - box[0]) / float(cnt)
                    h = (box[3] - box[1]) / float(cnt)
                    latent_box = [seed[0] - w / 2, seed[1] - h / 2, seed[0] + w / 2, seed[1] + h / 2]
                    enriched_detections.append({
                        "box": latent_box,
                        "score": bag["score"],
                        "mask": mask,
                        "is_latent": True,
                    })
                # This single detection just became len(centroid_seeds) latent
                # detections; every seed beyond the first is a bag that would
                # have been silently absorbed into one blob without merge
                # splitting. Track that real count so evaluation code (see
                # packages/cs_eval/replay_engine.py) can report a genuine
                # merge-caused-undercount signal instead of always 0.
                merge_extra_bags += max(0, len(hypothesis.centroid_seeds) - 1)
            else:
                enriched_detections.append(bag)

        # 3. Conveyor Multi-Object Tracking
        active_tracks = self.tracker.update(enriched_detections)

        # 4. Gate State Machine (PRE -> GATE -> POST)
        gate_crossings = self.gate_state_machine.process_tracks(
            tracks=active_tracks,
            frame_index=frame_index,
            monotonic_ns=monotonic_ns,
            wall_clock=wall_clock,
        )

        for event in gate_crossings:
            self.running_net_count += event.direction
            if event.direction > 0:
                self.total_forward_crossings += 1
            else:
                self.total_backward_crossings += 1

        # 5. Independent Area-Integral Estimator
        frame_masks = [b["mask"] for b in detection_result.bag_bodies if b.get("mask") is not None]
        area_estimate = self.area_counter.process_frame_masks(
            masks=frame_masks,
            belt_speed_px_per_frame=self.belt_motion.speed_px,
        )
        has_discrepancy, _ = self.area_counter.check_discrepancy(self.running_net_count)

        return FrameProcessingOutput(
            frame_index=frame_index,
            monotonic_ns=monotonic_ns,
            wall_clock=wall_clock,
            detections=detection_result,
            active_tracks=active_tracks,
            gate_crossings=gate_crossings,
            running_net_count=self.running_net_count,
            area_estimate=area_estimate,
            discrepancy_flag=has_discrepancy,
            merge_events_in_frame=merge_count,
            merge_extra_bags_in_frame=merge_extra_bags,
        )
