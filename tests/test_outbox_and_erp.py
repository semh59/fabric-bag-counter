"""Unit tests for Transactional Outbox and ERP Relay idempotency / fallback (§5.8, §11 M7)."""

from datetime import datetime, timedelta, timezone

from packages.cs_core.interfaces.erp_adapter import ErpResult, ErpStatus, ErpStatusState, SessionPayload
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import LineORM, OutboxORM, ProductProfileORM, SessionORM, SiteORM
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from services.erp_relay.worker import ErpRelayWorker
from drivers.erp_csv.adapter import CsvErpAdapter
from drivers.erp_sap_odata.adapter import SapODataErpAdapter


class _AlwaysFailingStatusQueryAdapter:
    """A status-query-capable adapter (like the real SAP OData one) whose
    submissions always fail and whose status query always reports the
    session as still not posted -- forces mark_failed()'s retry-exhausted
    escalation path."""

    supports_status_query = True

    def submit_session(self, payload: SessionPayload) -> ErpResult:
        return ErpResult(success=False, external_tx_id=None, error_message="simulated ERP 503", retryable=True)

    def query_status(self, external_ref: str) -> ErpStatus:
        return ErpStatus(state=ErpStatusState.PENDING, external_tx_id=None, message="still pending")


def setup_erp_test():
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="ERP Site")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line ERP")
        db.add(line)
        db.commit()

        prof = ProductProfileORM(site_id=site.id, name="Flour 25kg", erp_material_code="MAT_25KG", nominal_dims_mm={})
        db.add(prof)
        db.commit()

        sess = SessionORM(line_id=line.id, product_profile_id=prof.id, counted_total=500, external_ref="IRS_12345")
        db.add(sess)
        db.commit()
        db.refresh(sess)
        return sess.id


def test_outbox_csv_delivery():
    sess_id = setup_erp_test()
    entry_id = None
    with get_sync_session() as db:
        outbox_repo = OutboxRepository(db)
        entry = outbox_repo.create_entry(
            session_id=sess_id,
            payload={"line_id": 1, "counted_total": 500, "erp_material_code": "MAT_25KG"},
            external_ref="IRS_12345",
        )
        assert entry.status == "pending"
        entry_id = entry.id

    # Process via ErpRelayWorker with CSV adapter
    csv_adapter = CsvErpAdapter(export_dir="./data/test_exports")
    worker = ErpRelayWorker(adapter=csv_adapter)
    processed = worker.run_step()
    assert processed >= 1

    with get_sync_session() as db:
        from packages.cs_storage.models_orm import OutboxORM
        updated_entry = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
        assert updated_entry is not None
        assert updated_entry.status == "sent"


def test_status_query_adapter_exhausting_retries_creates_real_reconciliation_case():
    """Regression test: a status-query-capable adapter (e.g. SAP OData) whose
    submissions keep failing and whose status query never reports "posted"
    must, once mark_failed() escalates the outbox entry to reconcile_required
    (a terminal state claim_pending_entries() never selects again), leave a
    real ReconciliationORM row and session.status == "reconcile_required" --
    otherwise that session is stuck with zero operator-visible signal
    anywhere in the app (not /reconciliations, not the session status badge).
    """
    sess_id = setup_erp_test()
    with get_sync_session() as db:
        outbox_repo = OutboxRepository(db)
        entry = outbox_repo.create_entry(
            session_id=sess_id,
            payload={"line_id": 1, "counted_total": 500, "erp_material_code": "MAT_25KG"},
            external_ref="IRS_12345",
        )
        entry_id = entry.id

    worker = ErpRelayWorker(adapter=_AlwaysFailingStatusQueryAdapter())

    # max_attempts defaults to 5 inside OutboxRepository.mark_failed(); drive
    # enough real run_step() cycles to exhaust it. Each cycle re-claims the
    # entry (claim_pending_entries requires next_attempt_at <= now).
    for _ in range(6):
        with get_sync_session() as db:
            entry = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
            entry.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            db.commit()
        worker.run_step()

    with get_sync_session() as db:
        entry = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
        assert entry.status == "reconcile_required"

        rec_repo = ReconciliationRepository(db)
        open_cases = rec_repo.list_open_reconciliations()
        matching = [r for r in open_cases if r.session_id == sess_id]
        assert matching, "no ReconciliationORM row was created for the session -- it is now invisible to /reconciliations"

        sess_repo = SessionRepository(db)
        sess = sess_repo.get_by_id(sess_id)
        assert sess.status == "reconcile_required", "session.status never updated -- no UI badge would show this session needs attention"
