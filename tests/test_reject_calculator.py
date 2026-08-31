"""Unit tests for Deterministic Conveyor Reject Delay Calculator (§4.4, §5.5)."""

import math
import time
import pytest
from packages.cs_counting.reject_calculator import (
    DeterministicRejectCalculator,
    PneumaticDiverterSpec,
)


def test_reject_calculator_kinematic_constant_velocity():
    spec = PneumaticDiverterSpec(
        diverter_x_px=1000.0,
        pneumatic_stroke_time_ms=50.0,
        pneumatic_hold_time_ms=200.0,
        nominal_pressure_bar=6.0,
    )
    calc = DeterministicRejectCalculator(spec=spec)

    t_now = 1000.0
    # Δx = 500 px, v = 500 px/s -> transit = 1.0 s
    # Stroke delay @ 6.0 bar = 0.05 s
    # Trigger = 1000.0 + 1.0 - 0.05 = 1000.95 s
    event = calc.schedule_reject(
        track_id=42,
        current_x_px=500.0,
        belt_speed_px_per_s=500.0,
        defect_reason="torn_bag",
        belt_acceleration_px_per_s2=0.0,
        current_pressure_bar=6.0,
        detection_timestamp=t_now,
    )

    assert event.track_id == 42
    assert event.scheduled_trigger_time == pytest.approx(1000.95, rel=1e-4)
    assert event.scheduled_retract_time == pytest.approx(1001.15, rel=1e-4)


def test_reject_calculator_kinematic_acceleration():
    spec = PneumaticDiverterSpec(
        diverter_x_px=1000.0,
        pneumatic_stroke_time_ms=50.0,
        nominal_pressure_bar=6.0,
    )
    calc = DeterministicRejectCalculator(spec=spec)

    # Δx = 300 px, v0 = 100 px/s, a = 50 px/s²
    # 0.5 * 50 * t^2 + 100 * t - 300 = 0 -> 25 t^2 + 100 t - 300 = 0 -> t^2 + 4t - 12 = 0 -> (t+6)(t-2) = 0 -> t = 2.0 s
    t_transit = calc.compute_kinematic_transit_time(
        delta_x_px=300.0,
        initial_speed_px_per_s=100.0,
        acceleration_px_per_s2=50.0,
    )
    assert t_transit == pytest.approx(2.0, rel=1e-4)


def test_reject_calculator_pneumatic_pressure_compensation():
    spec = PneumaticDiverterSpec(
        pneumatic_stroke_time_ms=60.0,
        nominal_pressure_bar=6.0,
    )
    calc = DeterministicRejectCalculator(spec=spec)

    # At 6.0 bar: exactly 60.0 ms
    assert calc.compute_pneumatic_stroke_delay(current_pressure_bar=6.0) == pytest.approx(60.0, rel=1e-4)

    # At lower pressure (e.g. 4.0 bar): stroke time extends: 60 * sqrt(6/4) = 60 * 1.2247 = 73.48 ms
    delay_low_p = calc.compute_pneumatic_stroke_delay(current_pressure_bar=4.0)
    assert delay_low_p == pytest.approx(60.0 * math.sqrt(1.5), rel=1e-4)
    assert delay_low_p > 60.0
