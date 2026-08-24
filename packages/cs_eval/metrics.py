"""Counting and tracking evaluation metrics (§7.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for counting sessions and replay test runs."""
    total_ground_truth: int
    total_predicted: int
    net_absolute_error: int
    exact_count_sessions_ratio: float
    fp_per_1000_bags: float
    fn_per_1000_bags: float
    merge_caused_undercount_rate: float
    id_switches: int
    track_fragmentations: int
    systematic_bias: float  # (predicted - gt) / gt
    ledger_area_mean_delta: float
    dropped_frame_rate: float
    session_level_details: list[dict[str, Any]] = field(default_factory=list)


def compute_counting_metrics(
    ground_truth_counts: list[int],
    predicted_counts: list[int],
    area_estimates: list[float] | None = None,
    merge_caused_fn_counts: list[int] | None = None,
    id_switches_list: list[int] | None = None,
    track_frags_list: list[int] | None = None,
    total_frames: int = 1000,
    dropped_frames: int = 0,
) -> EvaluationMetrics:
    """Compute complete benchmark evaluation metrics across multiple test sessions."""
    n_sessions = len(ground_truth_counts)
    if n_sessions == 0:
        return EvaluationMetrics(
            total_ground_truth=0,
            total_predicted=0,
            net_absolute_error=0,
            exact_count_sessions_ratio=1.0,
            fp_per_1000_bags=0.0,
            fn_per_1000_bags=0.0,
            merge_caused_undercount_rate=0.0,
            id_switches=0,
            track_fragmentations=0,
            systematic_bias=0.0,
            ledger_area_mean_delta=0.0,
            dropped_frame_rate=0.0,
        )

    tot_gt = sum(ground_truth_counts)
    tot_pred = sum(predicted_counts)
    net_abs_err = abs(tot_pred - tot_gt)

    exact_sessions = sum(1 for gt, pred in zip(ground_truth_counts, predicted_counts) if gt == pred)
    exact_ratio = float(exact_sessions) / float(n_sessions)

    # False positives and False negatives calculation
    fps = sum(max(0, pred - gt) for gt, pred in zip(ground_truth_counts, predicted_counts))
    fns = sum(max(0, gt - pred) for gt, pred in zip(ground_truth_counts, predicted_counts))

    normalizer = max(1.0, float(tot_gt)) / 1000.0
    fp_per_1000 = float(fps) / normalizer
    fn_per_1000 = float(fns) / normalizer

    # Merge caused undercounts
    merge_fns = sum(merge_caused_fn_counts) if merge_caused_fn_counts else 0
    merge_undercount_rate = (float(merge_fns) / float(tot_gt)) if tot_gt > 0 else 0.0

    id_switches = sum(id_switches_list) if id_switches_list else 0
    track_frags = sum(track_frags_list) if track_frags_list else 0

    systematic_bias = float(tot_pred - tot_gt) / float(tot_gt) if tot_gt > 0 else 0.0

    # Ledger vs Area delta
    if area_estimates:
        deltas = [abs(pred - area) for pred, area in zip(predicted_counts, area_estimates)]
        ledger_area_delta = float(sum(deltas)) / float(n_sessions)
    else:
        ledger_area_delta = 0.0

    drop_rate = float(dropped_frames) / float(total_frames) if total_frames > 0 else 0.0

    session_details = []
    for i in range(n_sessions):
        gt = ground_truth_counts[i]
        pred = predicted_counts[i]
        area = area_estimates[i] if area_estimates else 0.0
        session_details.append({
            "session_index": i,
            "ground_truth": gt,
            "predicted": pred,
            "error": pred - gt,
            "area_estimate": area,
        })

    return EvaluationMetrics(
        total_ground_truth=tot_gt,
        total_predicted=tot_pred,
        net_absolute_error=net_abs_err,
        exact_count_sessions_ratio=exact_ratio,
        fp_per_1000_bags=fp_per_1000,
        fn_per_1000_bags=fn_per_1000,
        merge_caused_undercount_rate=merge_undercount_rate,
        id_switches=id_switches,
        track_fragmentations=track_frags,
        systematic_bias=systematic_bias,
        ledger_area_mean_delta=ledger_area_delta,
        dropped_frame_rate=drop_rate,
        session_level_details=session_details,
    )
