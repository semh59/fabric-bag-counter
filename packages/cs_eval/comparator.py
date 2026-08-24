"""Side-by-side model comparator for regression analysis (§11 M4)."""

from __future__ import annotations

from typing import Any
from packages.cs_eval.metrics import EvaluationMetrics


class ModelComparator:
    """Compares evaluation scorecards between candidate model version and base model version."""

    @staticmethod
    def compare(
        model_a_name: str,
        metrics_a: EvaluationMetrics,
        model_b_name: str,
        metrics_b: EvaluationMetrics,
    ) -> dict[str, Any]:
        """Produce structured comparison delta dictionary."""
        return {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "net_error_diff": metrics_b.net_absolute_error - metrics_a.net_absolute_error,
            "accuracy_diff": metrics_b.exact_count_sessions_ratio - metrics_a.exact_count_sessions_ratio,
            "fp_diff": metrics_b.fp_per_1000_bags - metrics_a.fp_per_1000_bags,
            "fn_diff": metrics_b.fn_per_1000_bags - metrics_a.fn_per_1000_bags,
            "merge_undercount_diff": metrics_b.merge_caused_undercount_rate - metrics_a.merge_caused_undercount_rate,
            "is_improved": (metrics_b.net_absolute_error < metrics_a.net_absolute_error) and (metrics_b.exact_count_sessions_ratio >= metrics_a.exact_count_sessions_ratio),
        }
