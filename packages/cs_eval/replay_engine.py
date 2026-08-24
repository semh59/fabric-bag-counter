"""Replay engine running scenarios through CountingEngine and generating metrics (§11 M4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence
import numpy as np
from packages.cs_counting.engine import CountingEngine
from packages.cs_eval.metrics import EvaluationMetrics, compute_counting_metrics


@dataclass
class ReplayScenario:
    """Pre-recorded or synthetic scenario definition with gold-standard counts."""
    name: str
    scenario_type: str  # heavy_shingling | sparse_flow | start_stop | backward_slip | light_change
    ground_truth_count: int
    frames: list[np.ndarray] = field(default_factory=list)
    fps: float = 25.0


class ReplayEngine:
    """Evaluates counting models against a standard library of test scenarios."""

    def __init__(self, engine: CountingEngine | None = None) -> None:
        self.engine = engine or CountingEngine()

    def run_scenario(self, scenario: ReplayScenario) -> tuple[int, float, list[Any]]:
        """Run counting engine through a single scenario."""
        self.engine.reset_session()
        base_time = datetime(2026, 8, 24, 12, 0, 0)
        dt = timedelta(seconds=1.0 / scenario.fps)

        all_events = []
        last_area = 0.0

        for frame_idx, frame in enumerate(scenario.frames):
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

        return self.engine.running_net_count, last_area, all_events

    def run_suite(self, scenarios: Sequence[ReplayScenario]) -> EvaluationMetrics:
        """Run entire suite of scenarios and compute aggregated evaluation metrics."""
        gt_counts = []
        pred_counts = []
        area_estimates = []

        total_frames = 0
        for sc in scenarios:
            pred_count, area_est, _ = self.run_scenario(sc)
            gt_counts.append(sc.ground_truth_count)
            pred_counts.append(pred_count)
            area_estimates.append(area_est)
            total_frames += len(sc.frames)

        return compute_counting_metrics(
            ground_truth_counts=gt_counts,
            predicted_counts=pred_counts,
            area_estimates=area_estimates,
            total_frames=total_frames,
            dropped_frames=0,
        )
