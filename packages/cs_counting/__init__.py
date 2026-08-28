"""Counting pipeline (Gate state machine, Area integral counter, and CountingEngine)."""

from packages.cs_counting.area_counter import AreaIntegralCounter
from packages.cs_counting.engine import CountingEngine
from packages.cs_counting.event_handler import CountingEventHandler, estimate_simulated_area
from packages.cs_counting.events import (
    GateCrossingRecorded,
    SessionAreaEstimateUpdated,
    SessionDegraded,
    SessionDiscrepancyDetected,
)
from packages.cs_counting.gate import GateCrossingEvent, GateStateMachine

__all__ = [
    "GateStateMachine",
    "GateCrossingEvent",
    "AreaIntegralCounter",
    "CountingEngine",
    "CountingEventHandler",
    "estimate_simulated_area",
    "GateCrossingRecorded",
    "SessionAreaEstimateUpdated",
    "SessionDegraded",
    "SessionDiscrepancyDetected",
]
