"""Repository for Transactional Outbox (§5.8, §11 M7)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import OutboxORM


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
        """Fetch pending outbox records ready for delivery."""
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

    def mark_in_progress(self, entry_id: int) -> None:
        """Mark outbox entry as currently being processed."""
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
    ) -> None:
        """Record delivery failure and schedule retry or route to reconciliation."""
        entry = self.db.execute(select(OutboxORM).where(OutboxORM.id == entry_id)).scalar_one_or_none()
        if entry:
            entry.last_error = error_msg
            if entry.attempts >= max_attempts:
                # Route to reconcile_required
                entry.status = "reconcile_required"
            else:
                entry.status = "pending"
                entry.next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds * (2 ** (entry.attempts - 1)))
            self.db.commit()

    def route_to_reconciliation(self, entry_id: int, reason: str) -> None:
        """Explicitly route an outbox failure directly to human reconciliation."""
        entry = self.db.execute(select(OutboxORM).where(OutboxORM.id == entry_id)).scalar_one_or_none()
        if entry:
            entry.status = "reconcile_required"
            entry.last_error = reason
            self.db.commit()
