"""Unit tests for Transactional Outbox and ERP Relay idempotency / fallback (§5.8, §11 M7)."""

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import LineORM, ProductProfileORM, SessionORM, SiteORM
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from services.erp_relay.worker import ErpRelayWorker
from drivers.erp_csv.adapter import CsvErpAdapter
from drivers.erp_sap_odata.adapter import SapODataErpAdapter


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
