"""Evaluation, metrics calculation, scoreboard generation, and replay harness (§7.4, §11 M4)."""

from packages.cs_eval.comparator import ModelComparator
from packages.cs_eval.metrics import EvaluationMetrics, compute_counting_metrics
from packages.cs_eval.replay_engine import ReplayEngine, ReplayScenario
from packages.cs_eval.scoreboard import generate_scoreboard_text

__all__ = [
    "compute_counting_metrics",
    "EvaluationMetrics",
    "generate_scoreboard_text",
    "ReplayEngine",
    "ReplayScenario",
    "ModelComparator",
]
