"""CountingEventHandler: the one place counting domain events become writes.

Mirrors services/inference/worker.py::InferenceWorker.run_step()'s ledger/
session-update logic exactly -- that was already the correct, reference
implementation (no rendering/camera-I/O mixed in). This module gives the
other three call sites (LiveStreamRenderer's real and simulated frame
paths, and services/api/routes.py::simulate_bag_crossing) a single shared
implementation to route through instead of each reimplementing it.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from packages.cs_counting.engine import FrameProcessingOutput
from packages.cs_counting.events import (
    GateCrossingRecorded,
    SessionAreaEstimateUpdated,
    SessionDegraded,
    SessionDiscrepancyDetected,
)
from packages.cs_storage.models_orm import CountEventORM
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.session_repo import SessionRepository


def estimate_simulated_area(counted_total: int) -> float:
    """Canonical simulated-area heuristic for the two demo/manual call
    sites (LiveStreamRenderer._process_simulated_frame, simulate_bag_crossing)
    that have no real CountingEngine output to derive area_estimate from.

    Previously these two sites used two different, already-diverged
    formulas (a flat `counted_total * 0.998` multiply in one, an
    incremental `+/-0.998` running delta in the other) -- both were
    approximations for the same thing, so there was never a reason for them
    to differ. This flat multiply is also self-correcting (derived fresh
    from the ledger-true counted_total each time) rather than accumulating
    drift the way an incremental delta can.
    """
    return float(counted_total) * 0.998


class CountingEventHandler:
    """Turns counting domain events into the real repository writes."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.ledger_repo = LedgerRepository(db)
        self.session_repo = SessionRepository(db)

    def handle_gate_crossing(self, event: GateCrossingRecorded) -> tuple[CountEventORM | None, bool]:
        """Append the crossing to the ledger and refresh the session's live
        counted_total from the ledger's true net sum. Returns (event, created)
        exactly as LedgerRepository.record_event() does -- created=False
        means an idempotency-constraint duplicate was ignored, not an error."""
        crossing = event.crossing
        record, created = self.ledger_repo.record_event(
            session_id=event.session_id,
            line_id=event.line_id,
            camera_id=event.camera_id,
            stream_epoch=event.stream_epoch,
            track_id=crossing.track_id,
            crossing_seq=crossing.crossing_seq,
            gate_id=crossing.gate_id,
            crossing_timestamp=crossing.crossing_timestamp,
            frame_index=crossing.frame_index,
            direction=crossing.direction,
            confidence=crossing.confidence,
            merge_flag=crossing.merge_flag,
            deployment_bundle_id=event.deployment_bundle_id,
            evidence_ref=event.evidence_ref,
            defect_reason=event.defect_reason,
            is_simulated=event.is_simulated,
        )
        if created:
            net_total = self.ledger_repo.get_session_total_count(event.session_id)
            self.session_repo.update_counted_total(event.session_id, net_total)
        return record, created

    def handle_area_updated(self, event: SessionAreaEstimateUpdated) -> None:
        self.session_repo.update_area_estimate(event.session_id, event.area_estimate)

    def handle_discrepancy(self, event: SessionDiscrepancyDetected) -> None:
        self.session_repo.flag_discrepancy(event.session_id, event.area_estimate)

    def handle_degraded(self, event: SessionDegraded) -> None:
        self.session_repo.mark_degraded(event.session_id)

    def handle_frame_output(
        self,
        output: FrameProcessingOutput,
        *,
        line_id: int,
        camera_id: int,
        session_id: int,
        stream_epoch: int,
        deployment_bundle_id: int = 1,
    ) -> list[GateCrossingRecorded | SessionAreaEstimateUpdated | SessionDiscrepancyDetected]:
        """Apply a full CountingEngine.process_frame() output: one
        GateCrossingRecorded per crossing, then SessionAreaEstimateUpdated,
        then SessionDiscrepancyDetected if flagged. Returns the events that
        were actually applied, in order, for logging/testing -- a crossing
        that hit the ledger's idempotency constraint (already recorded, a
        redelivered duplicate) is not included, matching callers that only
        want to log/react to genuinely new crossings."""
        applied: list[GateCrossingRecorded | SessionAreaEstimateUpdated | SessionDiscrepancyDetected] = []

        for crossing in output.gate_crossings:
            evt = GateCrossingRecorded(
                line_id=line_id,
                camera_id=camera_id,
                session_id=session_id,
                stream_epoch=stream_epoch,
                deployment_bundle_id=deployment_bundle_id,
                crossing=crossing,
            )
            _, created = self.handle_gate_crossing(evt)
            if created:
                applied.append(evt)

        area_evt = SessionAreaEstimateUpdated(session_id=session_id, area_estimate=output.area_estimate)
        self.handle_area_updated(area_evt)
        applied.append(area_evt)

        if output.discrepancy_flag:
            disc_evt = SessionDiscrepancyDetected(session_id=session_id, area_estimate=output.area_estimate)
            self.handle_discrepancy(disc_evt)
            applied.append(disc_evt)

        return applied
