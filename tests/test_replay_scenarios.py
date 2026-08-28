"""Tests for the real replay evaluation scenario generator (§11 M4).

Before this fix, the jobrunner's "replay" job kind hardcoded two
ReplayScenario objects with frames=[] (empty) and arbitrary
ground_truth_count values (20, 15) that had no real scenario behind them --
ReplayEngine.run_scenario() iterates scenario.frames, so with zero frames
it never ran CountingEngine.process_frame() at all. The job always
returned the same meaningless "100% undercounted" result regardless of
what model was actually deployed, and completely ignored the caller's
payload. These tests cover the real generator (packages/cs_eval/
replay_scenarios.py) that replaced it.
"""

import numpy as np
import pytest

from packages.cs_eval.replay_engine import ReplayEngine
from packages.cs_eval.replay_scenarios import (
    _count_real_crossings,
    _MovingBag,
    _x_at_frame,
    build_default_scenario_suite,
    build_scenario,
)
from packages.cs_storage.models_orm import JobORM
from services.jobrunner.worker import JobrunnerWorker

ALL_SCENARIO_TYPES = ["heavy_shingling", "sparse_flow", "start_stop", "backward_slip", "light_change"]


@pytest.mark.parametrize("scenario_type", ALL_SCENARIO_TYPES)
def test_build_scenario_produces_real_nonempty_frames(scenario_type):
    scenario = build_scenario(scenario_type, seed=1)
    assert len(scenario.frames) > 0
    assert scenario.scenario_type == scenario_type
    for frame in scenario.frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (640, 640, 3)
        # Real rendered content, not an all-zero/blank frame.
        assert np.count_nonzero(frame) > 0


@pytest.mark.parametrize("scenario_type", ALL_SCENARIO_TYPES)
def test_build_scenario_ground_truth_is_positive_and_plausible(scenario_type):
    """Every scenario type places 2-4 bags that are kinematically given
    enough frames*speed budget to actually finish crossing the gate --
    ground_truth_count must reflect that, not an arbitrary constant."""
    scenario = build_scenario(scenario_type, seed=1)
    assert 1 <= scenario.ground_truth_count <= 4


def test_heavy_shingling_has_more_bags_than_sparse_flow():
    """A real, meaningful difference between scenario types, not two
    copies of the same thing under different names."""
    shingling = build_scenario("heavy_shingling", seed=1)
    sparse = build_scenario("sparse_flow", seed=1)
    assert shingling.ground_truth_count > sparse.ground_truth_count


def test_light_change_scenario_actually_changes_brightness():
    """The named scenario must do the thing its name claims -- verify the
    back half of the frame sequence is genuinely darker than the front
    half, not just relabeled sparse_flow frames."""
    scenario = build_scenario("light_change", seed=1)
    n = len(scenario.frames)
    front_mean = float(np.mean(scenario.frames[: n // 4]))
    back_mean = float(np.mean(scenario.frames[3 * n // 4 :]))
    assert back_mean < front_mean


def test_build_scenario_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_scenario("not_a_real_scenario_type")


def test_build_default_scenario_suite_returns_all_five_types():
    suite = build_default_scenario_suite(seed=1)
    assert {s.scenario_type for s in suite} == set(ALL_SCENARIO_TYPES)


def test_count_real_crossings_matches_hand_worked_kinematics():
    """Direct unit check of the crossing-derivation math itself, isolated
    from rendering: a bag starting well before the gate and moving forward
    fast enough must cross exactly once."""
    bag = _MovingBag(start_x=0.0, y=320.0, speed_px=50.0, color=(0, 0, 0))
    assert _x_at_frame(bag, 0) == 0.0
    assert _x_at_frame(bag, 1) == 50.0
    assert _count_real_crossings(bag, num_frames=20) == 1  # starts pre-gate (320), crosses once


def test_count_real_crossings_backward_slip_nets_to_one():
    """A bag that crosses forward, slips back across the gate, then
    crosses forward again nets to +1 -- matches this whole session's
    established net-signed-direction counting design (SUM(direction))."""
    bag = _MovingBag(start_x=0.0, y=320.0, speed_px=50.0, color=(0, 0, 0), slip_at_frame=8, slip_duration=3)
    assert _count_real_crossings(bag, num_frames=20) == 1


def test_real_replay_engine_predicts_correctly_on_sparse_flow():
    """End-to-end: the real CountingEngine (real detector, real tracker,
    real gate state machine) actually processes the generated frames and
    arrives at the correct count -- this is the check that was impossible
    before, since frames=[] meant nothing ever ran."""
    scenario = build_scenario("sparse_flow", seed=1)
    engine = ReplayEngine()
    result = engine.run_scenario(scenario)
    assert result.predicted_count == scenario.ground_truth_count
    assert result.dropped_frames == 0


def test_jobrunner_replay_job_uses_real_scenarios_not_hardcoded_ones():
    """The job handler itself: kind="replay" with a payload requesting one
    scenario type must run that real scenario (fast: one type, not all
    five) and report its real, derived ground_truth_count -- not the old
    hardcoded 20/15 with empty frames, and not silently ignoring payload."""
    worker = JobrunnerWorker()
    job = JobORM(kind="replay", payload={"scenario_types": ["sparse_flow"], "seed": 7})
    result = worker.execute_job(job)

    assert "scenarios" in result
    assert len(result["scenarios"]) == 1
    assert result["scenarios"][0]["name"] == "sparse_flow"
    # The old hardcoded values were 20 and 15 -- a real sparse_flow scenario
    # (2 bags) must not coincidentally match either.
    assert result["scenarios"][0]["ground_truth_count"] not in (20, 15)
    assert "metrics" in result
