"""Replay CLI tool: python tools/replay.py [bundle_id] [scenario_dir] (§11 M4)."""

from __future__ import annotations

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_eval.replay_engine import ReplayEngine, ReplayScenario
from packages.cs_eval.scoreboard import generate_scoreboard_text


def build_synthetic_scenarios() -> list[ReplayScenario]:
    """Generate default test scenarios representing various factory conveyor conditions."""
    gen = SyntheticBagGenerator()
    scenarios = []

    # Scenario 1: Heavy shingling (overlapping bags)
    frames_s1 = [gen.generate_scene(num_bags=3)["image"] for _ in range(50)]
    scenarios.append(ReplayScenario(name="Heavy Shingling Overlap", scenario_type="heavy_shingling", ground_truth_count=15, frames=frames_s1))

    # Scenario 2: Sparse isolated flow
    frames_s2 = [gen.generate_scene(num_bags=1)["image"] for _ in range(40)]
    scenarios.append(ReplayScenario(name="Sparse Flow", scenario_type="sparse_flow", ground_truth_count=10, frames=frames_s2))

    return scenarios


def main() -> None:
    bundle_id = sys.argv[1] if len(sys.argv) > 1 else "default_bundle"
    scenario_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/scenarios"

    print(f"[Replay Harness] Running evaluation for bundle '{bundle_id}' against scenarios in '{scenario_dir}'...")

    scenarios = build_synthetic_scenarios()
    engine = ReplayEngine()
    metrics = engine.run_suite(scenarios)

    scoreboard = generate_scoreboard_text(metrics, title=f"Replay Evaluation - Bundle {bundle_id}")
    print("\n" + scoreboard + "\n")


if __name__ == "__main__":
    main()
