"""GateStateMachine: PRE_APPROACH -> PRE -> POST -> POST_DEPART zone transition
detection on belt axis (§6.8).

Four zones straddle the gate line (not three): PRE_APPROACH and POST_DEPART
are outer "approach"/"departed" buffer zones beyond the pre/post boundaries,
while PRE and POST are the immediate zones bracketing the gate position
itself. A crossing event fires when a track moves from a PRE-side zone
(PRE_APPROACH or PRE) to a POST-side zone (POST or POST_DEPART), or vice
versa for a backward slip.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence
from packages.cs_core.geometry import project_point_on_axis


def _validate_axis_vector(axis_vector: tuple[float, float]) -> None:
    """Fail fast on a degenerate (zero-length) belt motion axis vector.

    `project_point_on_axis` raises ValueError for a zero-norm axis vector
    rather than silently substituting norm=1.0 (§geometry.py). Validating
    here -- at gate configuration time (__init__/update_geometry) -- means
    a misconfigured gate raises one clear, actionable error immediately
    instead of raising (or previously, silently producing wrong positions)
    once per track on every frame inside `process_tracks`.
    """
    vx, vy = axis_vector
    if vx == 0.0 and vy == 0.0:
        raise ValueError(
            f"GateStateMachine axis_vector must be non-zero (got {axis_vector!r}); "
            "a zero vector cannot define a belt motion axis to project track "
            "positions onto."
        )


@dataclass
class GateCrossingEvent:
    """Event emitted when a track traverses the gate line."""
    track_id: int
    crossing_seq: int
    gate_id: int
    direction: int            # +1 for forward, -1 for backward slip
    crossing_timestamp: datetime
    frame_index: int
    monotonic_ns: int
    confidence: float
    merge_flag: bool
    centroid: tuple[float, float]


class GateStateMachine:
    """Monitors track trajectories relative to the gate line along conveyor motion axis."""

    def __init__(
        self,
        gate_id: int = 1,
        axis_origin: tuple[float, float] = (0.0, 0.0),
        axis_vector: tuple[float, float] = (1.0, 0.0),
        gate_position_along_axis: float = 300.0,
        pre_gate_offset: float = 60.0,
        post_gate_offset: float = 60.0,
    ) -> None:
        _validate_axis_vector(axis_vector)
        self.gate_id = gate_id
        self.axis_origin = axis_origin
        self.axis_vector = axis_vector
        self.gate_pos = gate_position_along_axis
        self.pre_boundary = self.gate_pos - pre_gate_offset
        self.post_boundary = self.gate_pos + post_gate_offset

        # Track state memory: track_id -> last_zone
        # ('PRE_APPROACH', 'PRE', 'POST', or 'POST_DEPART')
        self._track_zones: dict[int, str] = {}

    def update_geometry(
        self,
        axis_origin: tuple[float, float],
        axis_vector: tuple[float, float],
        gate_pos: float,
        pre_offset: float = 60.0,
        post_offset: float = 60.0,
    ) -> None:
        """Update gate line position and orientation."""
        _validate_axis_vector(axis_vector)
        self.axis_origin = axis_origin
        self.axis_vector = axis_vector
        self.gate_pos = gate_pos
        self.pre_boundary = self.gate_pos - pre_offset
        self.post_boundary = self.gate_pos + post_offset

    def process_tracks(
        self,
        tracks: Sequence[Any],
        frame_index: int,
        monotonic_ns: int,
        wall_clock: datetime,
    ) -> list[GateCrossingEvent]:
        """Evaluate positions of all active tracks and trigger crossing events."""
        events: list[GateCrossingEvent] = []

        for track in tracks:
            cx, cy = track.centroid
            pos = project_point_on_axis(
                point=(cx, cy),
                axis_origin=self.axis_origin,
                axis_vector=self.axis_vector,
            )

            # Determine current zone
            if pos < self.pre_boundary:
                curr_zone = "PRE_APPROACH"
            elif pos <= self.gate_pos:
                curr_zone = "PRE"
            elif pos <= self.post_boundary:
                curr_zone = "POST"
            else:
                curr_zone = "POST_DEPART"

            prev_zone = self._track_zones.get(track.track_id)
            if prev_zone is None and hasattr(track, "history") and len(track.history) >= 2:
                prev_cx, prev_cy = track.history[-2]
                prev_p = project_point_on_axis((prev_cx, prev_cy), self.axis_origin, self.axis_vector)
                if prev_p <= self.gate_pos:
                    prev_zone = "PRE"
                elif prev_p > self.gate_pos:
                    prev_zone = "POST"

            # ---------------------------------------------------------------
            # 1. Forward crossing: from PRE (or PRE_APPROACH) to POST / POST_DEPART
            # ---------------------------------------------------------------
            if prev_zone in ["PRE_APPROACH", "PRE"] and curr_zone in ["POST", "POST_DEPART"]:
                track.crossing_seq += 1
                events.append(
                    GateCrossingEvent(
                        track_id=track.track_id,
                        crossing_seq=track.crossing_seq,
                        gate_id=self.gate_id,
                        direction=1,  # Forward
                        crossing_timestamp=wall_clock,
                        frame_index=frame_index,
                        monotonic_ns=monotonic_ns,
                        confidence=track.score,
                        merge_flag=getattr(track, "merge_flag", False),
                        centroid=(cx, cy),
                    )
                )

            # ---------------------------------------------------------------
            # 2. Backward slip crossing: from POST (or POST_DEPART) to PRE
            #
            # Same reasoning as the forward-crossing check above: the zone
            # transition already implies prev_pos >= gate_pos > pos, so the
            # separate raw-position clause was redundant and was removed.
            # ---------------------------------------------------------------
            elif prev_zone in ["POST", "POST_DEPART"] and curr_zone in ["PRE", "PRE_APPROACH"]:
                track.crossing_seq += 1
                events.append(
                    GateCrossingEvent(
                        track_id=track.track_id,
                        crossing_seq=track.crossing_seq,
                        gate_id=self.gate_id,
                        direction=-1,  # Backward
                        crossing_timestamp=wall_clock,
                        frame_index=frame_index,
                        monotonic_ns=monotonic_ns,
                        confidence=track.score,
                        merge_flag=getattr(track, "merge_flag", False),
                        centroid=(cx, cy),
                    )
                )

            self._track_zones[track.track_id] = curr_zone

        # Cleanup deleted tracks
        active_ids = {t.track_id for t in tracks}
        self._track_zones = {k: v for k, v in self._track_zones.items() if k in active_ids}

        return events
