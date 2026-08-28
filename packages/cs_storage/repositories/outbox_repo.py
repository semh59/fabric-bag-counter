"""Repository for Transactional Outbox (§5.8, §11 M7)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import OutboxORM
from packages.cs_storage.repositories._dialect import is_postgres


class OutboxRepository:
    """Manages transactional outbox entries for ERP delivery with backoff and retry."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_entry(
        self,
        session_id: int,
        payload: dict[str, Any],
        external_ref: str | None = None,
    ) -> OutboxORM:
        """Create a pending outbox entry within the active database transaction."""
        entry = OutboxORM(
            session_id=session_id,
            payload=payload,
            status="pending",
            attempts=0,
            next_attempt_at=datetime.now(timezone.utc),
            external_ref=external_ref,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def fetch_pending_entries(self, limit: int = 10) -> Sequence[OutboxORM]:
        """Read-only view of pending outbox records ready for delivery.

        This is a plain SELECT with no claim/lock semantics -- two callers
        can both see the same rows. Safe for introspection/diagnostics
        (`tools/deep_diagnostic_suite.py`) or tests that then call
        mark_in_progress() explicitly on a specific entry, but NOT safe as
        the read half of a dispatcher's "read pending, then claim" loop with
        more than one worker process. Use claim_pending_entries() for that.
        """
        now = datetime.now(timezone.utc)
        stmt = (
            select(OutboxORM)
            .where(
                OutboxORM.status.in_(["pending", "failed"]),
                OutboxORM.next_attempt_at <= now,
            )
            .order_by(OutboxORM.created_at.asc())
            .limit(limit)
        )
        return self.db.execute(stmt).scalars().all()

    def claim_pending_entries(self, limit: int = 10) -> Sequence[OutboxORM]:
        """Atomically select AND claim up to `limit` pending entries in one step.

        fetch_pending_entries() followed by a separate mark_in_progress() per
        entry is not atomic: two dispatcher workers can both fetch the same
        pending rows before either marks them in_progress, and both then
        attempt delivery -- a duplicate ERP submission. This claims rows with
        a single conditional UPDATE (status re-checked at write time, same
        pattern as JobRepository.acquire_next_job), so a row already claimed
        by another worker's UPDATE can no longer match this one's WHERE
        clause.

        - PostgreSQL: candidates are chosen via `SELECT ... FOR UPDATE SKIP
          LOCKED`, so concurrent workers don't block each other -- each skips
          rows currently locked by another in-flight claim.
        - SQLite (dev/tests): `FOR UPDATE SKIP LOCKED` is Postgres-only
          syntax and unsupported here; the single UPDATE statement is still
          atomic since SQLite serializes writers.
        """
        now = datetime.now(timezone.utc)
        candidate_ids = select(OutboxORM.id).where(
            OutboxORM.status.in_(["pending", "failed"]),
            OutboxORM.next_attempt_at <= now,
        ).order_by(OutboxORM.created_at.asc()).limit(limit)
        if is_postgres(self.db):
            candidate_ids = candidate_ids.with_for_update(skip_locked=True)

        claim_stmt = (
            update(OutboxORM)
            .where(
                OutboxORM.id.in_(candidate_ids),
                OutboxORM.status.in_(["pending", "failed"]),
            )
            .values(status="in_progress", attempts=OutboxORM.attempts + 1)
            .returning(OutboxORM.id)
        )
        claimed_ids = self.db.execute(claim_stmt).scalars().all()
        self.db.commit()

        if not claimed_ids:
            return []
        stmt = select(OutboxORM).where(OutboxORM.id.in_(claimed_ids)).order_by(OutboxORM.created_at.asc())
        return self.db.execute(stmt).scalars().all()

    def mark_in_progress(self, entry_id: int) -> None:
        """Mark a single, already-known outbox entry as currently being processed.

        Use claim_pending_entries() instead when selecting *which* entries to
        process under multi-worker concurrency -- this method's own
        read-then-write is not atomic against a concurrent claim of the same
        entry_id.
        """
        entry = self.db.execute(select(OutboxORM).where(OutboxORM.id == entry_id)).scalar_one_or_none()
        if entry:
            entry.status = "in_progress"
            entry.attempts += 1
            self.db.commit()

    def mark_sent(self, entry_id: int) -> None:
        """Mark outbox entry as successfully delivered to ERP."""
        entry = self.db.execute(select(OutboxORM).where(OutboxORM.id == entry_id)).scalar_one_or_none()
        if entry:
            entry.status = "sent"
            entry.last_error = None
            self.db.commit()

    def mark_failed(
        self,
        entry_id: int,
        error_msg: str,
        backoff_seconds: int = 30,
        max_attempts: int = 5,
    ) -> bool:
        """Record delivery failure and schedule retry or route to reconciliation.

        Returns True when this call escalated the entry to reconcile_required
        (retries exhausted) -- claim_pending_entries() never selects that
        status again, so this is the caller's only chance to also create a
        real ReconciliationORM case / update the session's own status
        (session_repo, before this fix, was constructed for exactly that in
        ErpRelayWorker.process_entry() but never actually called with this
        outcome, leaving those sessions stuck with no human-visible signal
        anywhere -- not in /reconciliations, not in the session status badge).
        """
        entry = self.db.execute(select(OutboxORM).where(OutboxORM.id == entry_id)).scalar_one_or_none()
        if entry:
            entry.last_error = error_msg
            if entry.attempts >= max_attempts:
                # Route to reconcile_required
                entry.status = "reconcile_required"
                self.db.commit()
                return True
            else:
                entry.status = "pending"
                # In every call site today (ErpRelayWorker.process_entry, and the
                # tests exercising this path) mark_in_progress() runs first and
                # increments attempts, so attempts is already >= 1 by the time we
                # get here -- exponent 0 on the first real failure, doubling from
                # there. max(0, ...) is defensive: if mark_failed() were ever
                # called directly with attempts still at 0, `2 ** -1` would not
                # raise (Python allows negative int exponents on floats/ints
                # promoted to float) but would silently collapse the backoff to a
                # sub-second delay instead of the intended full backoff_seconds.
                exponent = max(0, entry.attempts - 1)
                entry.next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds * (2 ** exponent))
            self.db.commit()
        return False

    def route_to_reconciliation(self, entry_id: int, reason: str) -> None:
        """Explicitly route an outbox failure directly to human reconciliation."""
        entry = self.db.execute(select(OutboxORM).where(OutboxORM.id == entry_id)).scalar_one_or_none()
        if entry:
            entry.status = "reconcile_required"
            entry.last_error = reason
            self.db.commit()
