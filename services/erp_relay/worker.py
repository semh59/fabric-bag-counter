"""ERP Relay Worker: Transactional outbox consumer with idempotency and reconciliation routing (§5.8, §11 M7)."""

from __future__ import annotations

import logging
import time
from typing import Any
from packages.cs_core.interfaces.erp_adapter import ErpAdapter, SessionPayload
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.models_orm import OutboxORM, SessionORM
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from drivers.erp_csv.adapter import CsvErpAdapter
from drivers.erp_sap_odata.adapter import SapODataErpAdapter

logger = logging.getLogger(__name__)


class ErpRelayWorker:
    """Consumes transactional outbox events and guarantees reliable ERP synchronization."""

    def __init__(self, adapter: ErpAdapter | None = None, poll_interval_sec: float = 3.0) -> None:
        self.adapter = adapter or CsvErpAdapter()
        self.poll_interval = poll_interval_sec
        self.is_running = False

    def process_entry(self, entry: OutboxORM) -> bool:
        """Process a single outbox record with idempotency and reconciliation fallback."""
        payload_dict = entry.payload
        session_payload = SessionPayload(
            session_id=entry.session_id,
            line_id=payload_dict.get("line_id", 1),
            external_ref=entry.external_ref,
            product_profile_id=payload_dict.get("product_profile_id", 1),
            erp_material_code=payload_dict.get("erp_material_code"),
            counted_total=payload_dict.get("counted_total", 0),
            area_estimate_total=payload_dict.get("area_estimate_total", 0.0),
            opened_at=entry.created_at,
            closed_at=None,
            metadata=payload_dict,
        )

        logger.info(f"[ErpRelay] Dispatching session {entry.session_id} to ERP (Ref: {entry.external_ref})")
        result = self.adapter.submit_session(session_payload)

        with get_sync_session() as db:
            outbox_repo = OutboxRepository(db)
            session_repo = SessionRepository(db)
            rec_repo = ReconciliationRepository(db)

            if result.success:
                outbox_repo.mark_sent(entry.id)
                logger.info(f"[ErpRelay] Session {entry.session_id} successfully delivered to ERP (TX: {result.external_tx_id})")
                return True
            else:
                logger.warning(f"[ErpRelay] Submission error for session {entry.session_id}: {result.error_message}")

                # Idempotency / Reconciliation check (§11 M7)
                if self.adapter.supports_status_query and entry.external_ref:
                    # Query remote ERP status
                    status = self.adapter.query_status(entry.external_ref)
                    if status.state.value == "posted":
                        logger.info(f"[ErpRelay] Verified session {entry.session_id} was actually posted to ERP. Resolving.")
                        outbox_repo.mark_sent(entry.id)
                        return True
                    else:
                        outbox_repo.mark_failed(entry.id, error_msg=result.error_message or "Unknown ERP error")
                else:
                    # Unidirectional adapter without status query: route to reconciliation!
                    logger.error(
                        f"[ErpRelay] Adapter does not support status query. Routing session {entry.session_id} to human RECONCILIATION."
                    )
                    outbox_repo.route_to_reconciliation(entry.id, reason=result.error_message or "Unidirectional adapter timeout")
                    rec_repo.create_reconciliation(
                        session_id=entry.session_id,
                        trigger_reason="erp_conflict",
                        evidence_refs={"last_error": result.error_message, "outbox_id": entry.id},
                    )

                return False

    def run_step(self) -> int:
        """Poll and process pending outbox entries.

        Uses claim_pending_entries() rather than fetch_pending_entries() +
        a per-entry mark_in_progress() loop: the latter reads candidates and
        claims them in two separate steps, so a second ErpRelayWorker process
        polling concurrently could read and dispatch the same outbox entries
        before either claim lands -- a duplicate ERP submission. The atomic
        claim closes that window (see OutboxRepository.claim_pending_entries).
        """
        with get_sync_session() as db:
            outbox_repo = OutboxRepository(db)
            claimed = outbox_repo.claim_pending_entries(limit=5)
            if not claimed:
                return 0

            entries_to_process = list(claimed)

        processed = 0
        for entry in entries_to_process:
            success = self.process_entry(entry)
            if success:
                processed += 1

        return processed

    def start_loop(self) -> None:
        self.is_running = True
        logger.info("[ErpRelay] Worker loop started.")
        while self.is_running:
            count = self.run_step()
            if count == 0:
                time.sleep(self.poll_interval)

    def stop(self) -> None:
        self.is_running = False
        logger.info("[ErpRelay] Worker stopped.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    worker = ErpRelayWorker()
    try:
        worker.start_loop()
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
