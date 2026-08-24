"""Scoreboard formatter for replay harness regression runs (§7.4, §11 M4)."""

from __future__ import annotations

from packages.cs_eval.metrics import EvaluationMetrics


def generate_scoreboard_text(metrics: EvaluationMetrics, title: str = "Replay Evaluation Scoreboard") -> str:
    """Generate a clean ASCII scoreboard table for CI regression gates."""
    lines = [
        f"==================================================================",
        f"                  {title.upper()}",
        f"==================================================================",
        f"  Total Ground Truth Bags  : {metrics.total_ground_truth}",
        f"  Total Predicted Bags     : {metrics.total_predicted}",
        f"  Net Absolute Error       : {metrics.net_absolute_error} ({metrics.systematic_bias:+.2%})",
        f"  Exact Session Accuracy   : {metrics.exact_count_sessions_ratio:.1%}",
        f"------------------------------------------------------------------",
        f"  FP per 1,000 Bags        : {metrics.fp_per_1000_bags:.2f}",
        f"  FN per 1,000 Bags        : {metrics.fn_per_1000_bags:.2f}",
        f"  Merge Undercount Rate    : {metrics.merge_caused_undercount_rate:.2%}",
        f"  ID Switches              : {metrics.id_switches}",
        f"  Track Fragmentations     : {metrics.track_fragmentations}",
        f"  Ledger vs Area Delta     : {metrics.ledger_area_mean_delta:.2f} bags",
        f"  Dropped Frame Rate       : {metrics.dropped_frame_rate:.3%}",
        f"==================================================================",
    ]
    return "\n".join(lines)
