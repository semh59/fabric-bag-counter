"""Replay engine running scenarios through CountingEngine and generating metrics (§11 M4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence
import numpy as np
from packages.cs_counting.engine import CountingEngine
from packages.cs_eval.metrics import EvaluationMetrics, compute_counting_metrics

# How many leading frames of a scenario to sample when deriving a real
# AreaIntegralCounter calibration (see ReplayEngine._calibrate_area_counter).
AREA_CALIBRATION_SAMPLE_FRAMES = 30


@dataclass
class ReplayScenario:
    """Pre-recorded or synthetic scenario definition with gold-standard counts."""
    name: str
    scenario_type: str  # heavy_shingling | sparse_flow | start_stop | backward_slip | light_change
    ground_truth_count: int
    frames: list[np.ndarray] = field(default_factory=list)
    fps: float = 25.0


@dataclass
class ScenarioRunResult:
    """Everything measured while replaying a single ReplayScenario."""
    predicted_count: int
    area_estimate: float | None  # None when area calibration genuinely wasn't available
    events: list[Any]
    dropped_frames: int
    merge_extra_bags: int
    id_switches: int
    track_fragmentations: int


class ReplayEngine:
    """Evaluates counting models against a standard library of test scenarios."""

    def __init__(self, engine: CountingEngine | None = None) -> None:
        self.engine = engine or CountingEngine()

    def _calibrate_area_counter(self, scenario: ReplayScenario) -> bool:
        """Derive a real mean_bag_gate_area_px calibration for the engine's
        AreaIntegralCounter from this scenario's own frames, by sampling
        actual detected bag mask areas.

        Without this, AreaIntegralCounter.is_scale_calibrated stays False
        forever (it defaults False and nothing in ReplayEngine ever called
        update_calibration), so area_estimate silently reports 0.0 for every
        session -- not because there's genuinely no area signal, but because
        the estimator was never told the pixel scale to convert accumulated
        mask area into a bag count.

        Returns True if calibration succeeded (at least one bag mask sample
        was found in the scenario's leading frames); False means area
        estimation genuinely isn't available for this scenario (e.g. no
        detections at all), and callers should treat area_estimate as
        unmeasured (None) rather than a misleading 0.0.
        """
        sample_areas: list[float] = []
        for frame in scenario.frames[:AREA_CALIBRATION_SAMPLE_FRAMES]:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                continue
            detection_result = self.engine.detector.predict(frame)
            for bag in detection_result.bag_bodies:
                mask = bag.get("mask")
                if mask is not None:
                    area = float(np.sum(mask > 0))
                    if area > 0:
                        sample_areas.append(area)

        if not sample_areas:
            self.engine.area_counter.update_calibration(mean_bag_area_px=None, is_active=False)
            return False

        mean_bag_area_px = float(np.mean(sample_areas))
        self.engine.area_counter.update_calibration(mean_bag_area_px=mean_bag_area_px, is_active=True)
        return True

    def run_scenario(self, scenario: ReplayScenario) -> ScenarioRunResult:
        """Run counting engine through a single scenario."""
        self.engine.reset_session()
        base_time = datetime(2026, 8, 24, 12, 0, 0)
        dt = timedelta(seconds=1.0 / scenario.fps)

        all_events: list[Any] = []
        last_area = 0.0
        dropped_frames = 0
        merge_extra_bags = 0

        # Real per-scenario tracking-quality counters: ConveyorByteTracker
        # accumulates fragmentation_count / id_switch_count as running totals
        # from genuine state transitions (see packages/cs_tracking/tracker.py).
        # Snapshotting before/after isolates the delta caused by THIS
        # scenario, even though the tracker instance is shared across a
        # run_suite() call.
        frag_before = self.engine.tracker.fragmentation_count
        switches_before = self.engine.tracker.id_switch_count

        area_calibrated = self._calibrate_area_counter(scenario)

        for frame_idx, frame in enumerate(scenario.frames):
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                # A malformed/missing frame -- e.g. a camera glitch or an
                # upstream decode failure -- is a genuine dropped frame, not
                # something to silently process or ignore.
                dropped_frames += 1
                continue

            wall_clock = base_time + (dt * frame_idx)
            mono_ns = int(frame_idx * (1e9 / scenario.fps))
            output = self.engine.process_frame(
                image=frame,
                frame_index=frame_idx,
                monotonic_ns=mono_ns,
                wall_clock=wall_clock,
            )
            all_events.extend(output.gate_crossings)
            last_area = output.area_estimate
            merge_extra_bags += output.merge_extra_bags_in_frame

        frag_after = self.engine.tracker.fragmentation_count
        switches_after = self.engine.tracker.id_switch_count

        return ScenarioRunResult(
            predicted_count=self.engine.running_net_count,
            area_estimate=last_area if area_calibrated else None,
            events=all_events,
            dropped_frames=dropped_frames,
            merge_extra_bags=merge_extra_bags,
            id_switches=switches_after - switches_before,
            track_fragmentations=frag_after - frag_before,
        )

    def run_suite(self, scenarios: Sequence[ReplayScenario]) -> EvaluationMetrics:
        """Run entire suite of scenarios and compute aggregated evaluation metrics."""
        gt_counts = []
        pred_counts = []
        area_estimates: list[float | None] = []
        merge_caused_fn_counts = []
        id_switches_list = []
        track_frags_list = []

        total_frames = 0
        total_dropped_frames = 0
        for sc in scenarios:
            result = self.run_scenario(sc)
            gt_counts.append(sc.ground_truth_count)
            pred_counts.append(result.predicted_count)
            area_estimates.append(result.area_estimate)
            merge_caused_fn_counts.append(result.merge_extra_bags)
            id_switches_list.append(result.id_switches)
            track_frags_list.append(result.track_fragmentations)
            total_frames += len(sc.frames)
            total_dropped_frames += result.dropped_frames

        return compute_counting_metrics(
            ground_truth_counts=gt_counts,
            predicted_counts=pred_counts,
            area_estimates=area_estimates,
            merge_caused_fn_counts=merge_caused_fn_counts,
            id_switches_list=id_switches_list,
            track_frags_list=track_frags_list,
            total_frames=total_frames,
            dropped_frames=total_dropped_frames,
        )
