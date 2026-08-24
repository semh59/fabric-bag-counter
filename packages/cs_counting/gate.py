"""GateStateMachine: PRE -> GATE -> POST transition detection on belt axis (§6.8)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence
from packages.cs_core.geometry import project_point_on_axis


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
        self.gate_id = gate_id
        self.axis_origin = axis_origin
        self.axis_vector = axis_vector
        self.gate_pos = gate_position_along_axis
        self.pre_boundary = self.gate_pos - pre_gate_offset
        self.post_boundary = self.gate_pos + post_gate_offset

        # Track state memory: track_id -> last_zone ('PRE', 'GATE', 'POST')
        self._track_zones: dict[int, str] = {}
        self._track_last_pos: dict[int, float] = {}

    def update_geometry(
        self,
        axis_origin: tuple[float, float],
        axis_vector: tuple[float, float],
        gate_pos: float,
        pre_offset: float = 60.0,
        post_offset: float = 60.0,
    ) -> None:
        """Update gate line position and orientation."""
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
            prev_pos = self._track_last_pos.get(track.track_id, pos)

            # ---------------------------------------------------------------
            # 1. Forward crossing: from PRE (or PRE_APPROACH) to POST / POST_DEPART
            # ---------------------------------------------------------------
            if (prev_zone in ["PRE_APPROACH", "PRE"] and curr_zone in ["POST", "POST_DEPART"]) or (
                prev_pos <= self.gate_pos < pos and prev_zone is not None
            ):
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
            # ---------------------------------------------------------------
            elif (prev_zone in ["POST", "POST_DEPART"] and curr_zone in ["PRE", "PRE_APPROACH"]) or (
                prev_pos >= self.gate_pos > pos and prev_zone is not None
            ):
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
            self._track_last_pos[track.track_id] = pos

        # Cleanup deleted tracks
        active_ids = {t.track_id for t in tracks}
        self._track_zones = {k: v for k, v in self._track_zones.items() if k in active_ids}
        self._track_last_pos = {k: v for k, v in self._track_last_pos.items() if k in active_ids}

        return events
