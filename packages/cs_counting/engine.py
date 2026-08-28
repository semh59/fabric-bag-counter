"""CountingEngine: Complete synchronized vision, tracking, and counting pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from packages.cs_core.geometry import point_in_polygon
from packages.cs_counting.area_counter import AreaIntegralCounter
from packages.cs_counting.gate import GateCrossingEvent, GateStateMachine
from packages.cs_tracking.merge_detector import MergeDetector
from packages.cs_tracking.motion import BeltMotionModel
from packages.cs_tracking.tracker import BagTrack, ConveyorByteTracker
from packages.cs_vision.detector import DetectionResult, VisionDetector

# The detector always runs on this canonical canvas size in the real
# pipeline (letterboxed/perspective-warped upstream in
# LiveStreamRenderer/InferenceWorker before process_frame() is ever
# called) -- matches packages/cs_vision/calibration.py's own CANVAS_SIZE,
# duplicated rather than imported for the same reason documented there:
# avoiding a heavier import chain into this module.
CANVAS_SIZE = (640, 640)


# The schema's own full-frame default (packages/cs_core/config_defaults.py's
# SCHEMA_V1_DEFAULTS["roi_polygon"]) -- an unmodified config genuinely means
# "no ROI restriction", so it's the one shape allowed to short-circuit to
# None. A bounding-box check ("does this polygon's extent touch all four
# canvas edges") would be unsound here -- a non-rectangular polygon (e.g. a
# diamond whose points touch each edge's midpoint) can satisfy that while
# covering only a fraction of the canvas -- so this compares the actual
# normalized points instead.
_FULL_FRAME_ROI_NORM = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def _denormalize_roi(polygon_norm: list[list[float]] | None) -> list[tuple[float, float]] | None:
    """Convert a [0,1]-normalized ROI polygon (packages/cs_core/config_defaults.py's
    schema) into pixel coordinates on CANVAS_SIZE. Returns None (meaning "no
    filtering") for an empty/missing polygon or one that exactly matches the
    schema's full-frame default, so the common no-custom-ROI case costs
    nothing extra per frame.
    """
    if not polygon_norm or len(polygon_norm) < 3:
        return None
    normalized = [[round(float(x), 6), round(float(y), 6)] for x, y in polygon_norm]
    if normalized == _FULL_FRAME_ROI_NORM:
        return None
    w, h = CANVAS_SIZE
    return [(float(x) * w, float(y) * h) for x, y in polygon_norm]


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
        # Set via configure() -- None means no ROI restriction (the default,
        # matching an engine nobody has ever called configure() on).
        self._roi_polygon_px: list[tuple[float, float]] | None = None

    def configure(self, payload: dict[str, Any]) -> None:
        """Apply an effective line config payload (see
        packages/cs_core/config_defaults.py, ConfigRepository.
        get_effective_config_payload()) to this engine's already-constructed
        sub-components.

        Only the schema keys that have a real consumer somewhere in this
        pipeline are applied here -- confidence_threshold, merge_area_ratio
        (two copies: the detector's own merge-count sizing, and
        MergeDetector's Signal 1), discrepancy_threshold, merge_signals.
        min_votes, and mask_iou_threshold (mapped onto
        ConveyorByteTracker.match_cost_threshold, the closest real match for
        that name -- both describe the matching-cost cutoff for track
        association). The remaining schema keys (gate_line, pre_gate_zone,
        post_gate_zone, tracking_cost_weights, latent_track_grace_frames,
        latency_p95_pause_threshold_ms, merge_signals.*_enabled,
        area_integral.min_confidence/smoothing_window) have no consumer
        anywhere in packages/ or services/ today -- wiring those would mean
        inventing new behavior in classes that don't support it yet, not
        connecting an existing one, so they're deliberately left alone here.
        """
        if "confidence_threshold" in payload:
            self.detector.conf_threshold = float(payload["confidence_threshold"])
        if "merge_area_ratio" in payload:
            ratio = float(payload["merge_area_ratio"])
            self.detector.merge_area_ratio = ratio
            self.merge_detector.merge_area_ratio = ratio
        if "discrepancy_threshold" in payload:
            self.area_counter.discrepancy_threshold = float(payload["discrepancy_threshold"])
        min_votes = payload.get("merge_signals", {}).get("min_votes")
        if min_votes is not None:
            self.merge_detector.min_votes = int(min_votes)
        if "mask_iou_threshold" in payload:
            self.tracker.match_cost_threshold = float(payload["mask_iou_threshold"])
        self._roi_polygon_px = _denormalize_roi(payload.get("roi_polygon"))

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

        # 1b. Real counting-area ROI filter (set via configure(), see
        # _denormalize_roi): a detection whose box centroid falls outside
        # the configured polygon is dropped before it can reach tracking,
        # merge analysis, or the area estimator -- an operator-restricted
        # counting area actually restricts counting, not just the display.
        if self._roi_polygon_px is not None:
            detection_result.bag_bodies = [
                b for b in detection_result.bag_bodies
                if point_in_polygon(
                    ((b["box"][0] + b["box"][2]) / 2.0, (b["box"][1] + b["box"][3]) / 2.0),
                    self._roi_polygon_px,
                )
            ]

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
