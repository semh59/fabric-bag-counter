"""Tests for CountingEventHandler (§5.5, §5.6, §5.7) -- the single shared
implementation of "counting domain event -> ledger/session writes" that
replaced four independent hand-written copies: LiveStreamRenderer's real
and simulated frame paths, InferenceWorker.run_step, and
services/api/routes.py::simulate_bag_crossing.
"""

from datetime import UTC, datetime

from packages.cs_counting.engine import FrameProcessingOutput
from packages.cs_counting.event_handler import CountingEventHandler, estimate_simulated_area
from packages.cs_counting.events import (
    GateCrossingRecorded,
    SessionAreaEstimateUpdated,
    SessionDegraded,
    SessionDiscrepancyDetected,
)
from packages.cs_counting.gate import GateCrossingEvent
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import LineORM, ProductProfileORM, SiteORM
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository


def _setup_session() -> int:
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Handler Test Site")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()
        prof = ProductProfileORM(site_id=site.id, name="Bag", nominal_dims_mm={})
        db.add(prof)
        db.commit()
        sess = SessionRepository(db).create_session(line_id=line.id, product_profile_id=prof.id)
        return sess.id


def _crossing(track_id: int = 1, direction: int = 1, crossing_seq: int = 1) -> GateCrossingEvent:
    return GateCrossingEvent(
        track_id=track_id,
        crossing_seq=crossing_seq,
        gate_id=1,
        direction=direction,
        crossing_timestamp=datetime.now(UTC),
        frame_index=1,
        monotonic_ns=1_000_000,
        confidence=0.97,
        merge_flag=False,
        centroid=(100.0, 200.0),
    )


def test_handle_gate_crossing_records_and_updates_counted_total():
    sess_id = _setup_session()
    with get_sync_session() as db:
        handler = CountingEventHandler(db)
        record, created = handler.handle_gate_crossing(GateCrossingRecorded(
            line_id=1, camera_id=1, session_id=sess_id, stream_epoch=1,
            deployment_bundle_id=1, crossing=_crossing(),
        ))
        assert created is True
        assert record is not None

        sess = SessionRepository(db).get_by_id(sess_id)
        assert sess.counted_total == 1


def test_handle_gate_crossing_idempotent_duplicate_does_not_double_count():
    sess_id = _setup_session()
    with get_sync_session() as db:
        handler = CountingEventHandler(db)
        crossing = _crossing()
        handler.handle_gate_crossing(GateCrossingRecorded(
            line_id=1, camera_id=1, session_id=sess_id, stream_epoch=1,
            deployment_bundle_id=1, crossing=crossing,
        ))
        _, created_again = handler.handle_gate_crossing(GateCrossingRecorded(
            line_id=1, camera_id=1, session_id=sess_id, stream_epoch=1,
            deployment_bundle_id=1, crossing=crossing,  # identical -> idempotency hit
        ))
        assert created_again is False

        sess = SessionRepository(db).get_by_id(sess_id)
        assert sess.counted_total == 1  # not double-counted


def test_handle_frame_output_applies_crossings_area_and_discrepancy():
    sess_id = _setup_session()
    output = FrameProcessingOutput(
        frame_index=1, monotonic_ns=1, wall_clock=datetime.now(UTC),
        detections=None, active_tracks=[],
        gate_crossings=[_crossing(track_id=1), _crossing(track_id=2, crossing_seq=1)],
        running_net_count=2, area_estimate=42.5, discrepancy_flag=True,
    )
    with get_sync_session() as db:
        handler = CountingEventHandler(db)
        applied = handler.handle_frame_output(
            output, line_id=1, camera_id=1, session_id=sess_id, stream_epoch=1,
        )

        kinds = [type(e) for e in applied]
        assert kinds.count(GateCrossingRecorded) == 2
        assert SessionAreaEstimateUpdated in kinds
        assert SessionDiscrepancyDetected in kinds

        sess = SessionRepository(db).get_by_id(sess_id)
        assert sess.counted_total == 2
        assert sess.area_estimate_total == 42.5
        assert sess.discrepancy_flag is True
        assert sess.status == "reconcile_required"


def test_handle_frame_output_omits_duplicate_crossings_from_applied_list():
    """A crossing that hits the ledger's idempotency constraint must not
    appear in the returned applied list -- callers (InferenceWorker) log
    only genuinely new crossings from this list."""
    sess_id = _setup_session()
    crossing = _crossing()
    output = FrameProcessingOutput(
        frame_index=1, monotonic_ns=1, wall_clock=datetime.now(UTC),
        detections=None, active_tracks=[], gate_crossings=[crossing],
        running_net_count=1, area_estimate=1.0, discrepancy_flag=False,
    )
    with get_sync_session() as db:
        handler = CountingEventHandler(db)
        first = handler.handle_frame_output(output, line_id=1, camera_id=1, session_id=sess_id, stream_epoch=1)
        assert len([e for e in first if isinstance(e, GateCrossingRecorded)]) == 1

        second = handler.handle_frame_output(output, line_id=1, camera_id=1, session_id=sess_id, stream_epoch=1)
        assert len([e for e in second if isinstance(e, GateCrossingRecorded)]) == 0


def test_handle_degraded_transitions_session_status():
    sess_id = _setup_session()
    with get_sync_session() as db:
        SessionRepository(db).update_status(sess_id, "counting")
    with get_sync_session() as db:
        handler = CountingEventHandler(db)
        handler.handle_degraded(SessionDegraded(session_id=sess_id, camera_id=1, consecutive_drops=5))
        sess = SessionRepository(db).get_by_id(sess_id)
        assert sess.status == "degraded"


def test_estimate_simulated_area_matches_between_both_simulated_call_sites():
    """LiveStreamRenderer._process_simulated_frame and
    simulate_bag_crossing previously used two different, diverged formulas
    (a flat multiply vs. an incremental delta). Both now call this same
    function -- this test locks in that there is exactly one formula."""
    assert estimate_simulated_area(0) == 0.0
    assert estimate_simulated_area(10) == 9.98
    assert estimate_simulated_area(200) == 199.6


def test_require_reconciliation_shared_by_flag_discrepancy_and_reconciliation_repo():
    """The reconcile_required transition is applied by SessionRepository's
    private _set_reconcile_required helper from three call sites
    (flag_discrepancy, close_session, ReconciliationRepository.
    create_reconciliation) -- verify two of them independently reach the
    same status."""
    sess_id_a = _setup_session()
    sess_id_b = _setup_session()

    with get_sync_session() as db:
        SessionRepository(db).flag_discrepancy(sess_id_a, area_estimate=10.0)
    with get_sync_session() as db:
        assert SessionRepository(db).get_by_id(sess_id_a).status == "reconcile_required"

    with get_sync_session() as db:
        ReconciliationRepository(db).create_reconciliation(session_id=sess_id_b, trigger_reason="erp_conflict")
    with get_sync_session() as db:
        assert SessionRepository(db).get_by_id(sess_id_b).status == "reconcile_required"
