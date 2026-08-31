"""Deterministic Conveyor Reject Delay and Kinematic Trajectory Calculator (§4.4, §5.5).

Computes sub-millimeter spatial trajectories and exact pneumatic cylinder trigger timestamps
accounting for belt motion kinematics, non-zero acceleration profiles, pneumatic air pressure curves,
and optical homography spatial transformation.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PneumaticDiverterSpec:
    """Hardware and pneumatic specification of physical conveyor diverter."""

    diverter_x_px: float = 950.0
    optical_gate_x_px: float = 500.0
    pneumatic_stroke_time_ms: float = 65.0       # Nominal stroke extension delay at 6.0 bar
    pneumatic_hold_time_ms: float = 220.0        # Active push hold duration
    nominal_pressure_bar: float = 6.0            # Nominal factory airline pressure
    px_per_mm: float = 0.85                      # Optical scale factor
    max_diverter_jitter_ms: float = 5.0          # Max acceptable timing jitter tolerance


@dataclass
class ScheduledRejectEvent:
    """Kinematically calculated pneumatic trigger event."""

    track_id: int
    scheduled_trigger_time: float
    scheduled_retract_time: float
    defect_reason: str
    detected_at: float
    initial_x_px: float
    target_diverter_x_px: float
    belt_speed_px_per_s: float
    belt_accel_px_per_s2: float
    actual_stroke_delay_ms: float
    transit_duration_s: float
    executed: bool = False
    retracted: bool = False


class DeterministicRejectCalculator:
    """High-precision kinematic trajectory and pneumatic trigger calculator."""

    def __init__(self, spec: PneumaticDiverterSpec | None = None) -> None:
        self.spec = spec or PneumaticDiverterSpec()
        self._queue: list[ScheduledRejectEvent] = []

    def compute_pneumatic_stroke_delay(self, current_pressure_bar: float = 6.0) -> float:
        """Compute pressure-compensated cylinder extension time using pneumatic flow dynamics.

        t_stroke(P) = t_nominal * sqrt(P_nominal / P_actual)
        """
        p_act = max(2.5, min(10.0, float(current_pressure_bar)))
        p_nom = self.spec.nominal_pressure_bar
        ratio = math.sqrt(p_nom / p_act)
        return self.spec.pneumatic_stroke_time_ms * ratio

    def compute_kinematic_transit_time(
        self,
        delta_x_px: float,
        initial_speed_px_per_s: float,
        acceleration_px_per_s2: float = 0.0,
    ) -> float:
        """Solve kinematic transit time: Δx = v0*t + 0.5*a*t^2.

        Uses stable quadratic root finding with fallback to constant velocity.
        """
        if delta_x_px <= 0:
            return 0.0

        v0 = max(1.0, float(initial_speed_px_per_s))
        a = float(acceleration_px_per_s2)

        # Constant velocity case
        if abs(a) < 1e-4:
            return delta_x_px / v0

        # Non-zero acceleration quadratic: 0.5*a*t^2 + v0*t - Δx = 0
        discriminant = v0 * v0 + 2.0 * a * delta_x_px
        if discriminant < 0:
            # Bag stops before reaching diverter; use instantaneous velocity
            return delta_x_px / v0

        t_transit = (-v0 + math.sqrt(discriminant)) / a
        if t_transit < 0:
            t_transit = delta_x_px / v0

        return t_transit

    def schedule_reject(
        self,
        track_id: int,
        current_x_px: float,
        belt_speed_px_per_s: float,
        defect_reason: str,
        belt_acceleration_px_per_s2: float = 0.0,
        current_pressure_bar: float = 6.0,
        detection_timestamp: float | None = None,
        homography_matrix: np.ndarray | None = None,
    ) -> ScheduledRejectEvent:
        """Calculate exact millisecond trigger timestamp with kinematic integration."""
        t_detect = detection_timestamp if detection_timestamp is not None else time.time()
        speed = max(5.0, float(belt_speed_px_per_s))

        # 1. Coordinate calculation (with homography if provided)
        target_x = self.spec.diverter_x_px
        if homography_matrix is not None:
            pt = np.array([[[current_x_px, 320.0]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(pt, homography_matrix)
            current_x_px = float(warped[0][0][0])

        delta_x = max(0.0, target_x - current_x_px)

        # 2. Kinematic transit time
        transit_time_s = self.compute_kinematic_transit_time(
            delta_x_px=delta_x,
            initial_speed_px_per_s=speed,
            acceleration_px_per_s2=belt_acceleration_px_per_s2,
        )

        # 3. Pressure-compensated stroke delay
        stroke_delay_ms = self.compute_pneumatic_stroke_delay(current_pressure_bar=current_pressure_bar)
        stroke_delay_s = stroke_delay_ms / 1000.0
        hold_time_s = self.spec.pneumatic_hold_time_ms / 1000.0

        # Exact trigger and retract timestamps
        t_trigger = t_detect + transit_time_s - stroke_delay_s
        t_retract = t_trigger + hold_time_s

        event = ScheduledRejectEvent(
            track_id=track_id,
            scheduled_trigger_time=t_trigger,
            scheduled_retract_time=t_retract,
            defect_reason=defect_reason,
            detected_at=t_detect,
            initial_x_px=current_x_px,
            target_diverter_x_px=target_x,
            belt_speed_px_per_s=speed,
            belt_accel_px_per_s2=belt_acceleration_px_per_s2,
            actual_stroke_delay_ms=round(stroke_delay_ms, 2),
            transit_duration_s=round(transit_time_s, 4),
        )

        self._queue.append(event)
        logger.info(
            f"[RejectCalc] Scheduled Track #{track_id}: "
            f"Δx={delta_x:.1f}px, transit={transit_time_s*1000:.1f}ms, "
            f"stroke={stroke_delay_ms:.1f}ms, trigger_in={(t_trigger - time.time())*1000:.1f}ms (Reason: {defect_reason})"
        )
        return event

    def poll_due_commands(self, now: float | None = None) -> tuple[list[ScheduledRejectEvent], list[ScheduledRejectEvent]]:
        """Poll and return (triggers_due, retractions_due) at current timestamp."""
        current_time = now if now is not None else time.time()
        triggers_due: list[ScheduledRejectEvent] = []
        retractions_due: list[ScheduledRejectEvent] = []
        active_queue: list[ScheduledRejectEvent] = []

        for event in self._queue:
            if not event.executed and current_time >= event.scheduled_trigger_time:
                event.executed = True
                triggers_due.append(event)

            if event.executed and not event.retracted and current_time >= event.scheduled_retract_time:
                event.retracted = True
                retractions_due.append(event)

            if not (event.executed and event.retracted):
                active_queue.append(event)

        self._queue = active_queue
        return triggers_due, retractions_due

    def get_pending_count(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()
