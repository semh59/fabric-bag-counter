"""Domain events emitted by the counting pipeline (§5.5, §5.6, §5.7).

The explicit contract between the counting domain (CountingEngine,
GateStateMachine -- neither of which has any DB coupling, see their tests
in tests/test_counting_and_gate.py and tests/test_merge_detector.py) and
persistence. Before these existed, every caller of CountingEngine
(LiveStreamRenderer's two frame paths, InferenceWorker, and the
simulate_bag_crossing endpoint) hand-wrote its own copy of "turn a crossing
into a ledger row and update session totals" -- four independent copies,
two of which had already diverged on the area-estimate formula they used.
CountingEventHandler (event_handler.py, same package) is the one place
these events get turned into repository calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from packages.cs_counting.gate import GateCrossingEvent


@dataclass(frozen=True)
class GateCrossingRecorded:
    """A track crossed the gate line and should be appended to the ledger."""
    line_id: int
    camera_id: int
    session_id: int
    stream_epoch: int
    deployment_bundle_id: int
    crossing: GateCrossingEvent
    is_simulated: bool = False
    defect_reason: str | None = None
    evidence_ref: str | None = None


@dataclass(frozen=True)
class SessionAreaEstimateUpdated:
    """The session's live area-integral estimate changed."""
    session_id: int
    area_estimate: float


@dataclass(frozen=True)
class SessionDiscrepancyDetected:
    """The ledger count and area estimate disagree beyond tolerance --
    requires human reconciliation."""
    session_id: int
    area_estimate: float


@dataclass(frozen=True)
class SessionDegraded:
    """A camera has exceeded its consecutive-frame-drop threshold."""
    session_id: int
    camera_id: int
    consecutive_drops: int
