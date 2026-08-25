"""Repository for Reconciliation records and human auditing (§5.7)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import ReconciliationORM, SessionORM


class ReconciliationRepository:
    """Manages reconciliation cases created when counting integrity is compromised."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_reconciliation(
        self,
        session_id: int,
        trigger_reason: str,
        evidence_refs: dict[str, Any] | None = None,
        assigned_role: str = "engineer",
    ) -> ReconciliationORM:
        """Create a reconciliation case and link it to the session."""
        rec = ReconciliationORM(
            session_id=session_id,
            trigger_reason=trigger_reason,
            assigned_role=assigned_role,
            evidence_refs=evidence_refs or {},
            opened_at=datetime.now(timezone.utc),
        )
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)

        session = self.db.execute(select(SessionORM).where(SessionORM.id == session_id)).scalar_one_or_none()
        if session:
            session.reconciliation_id = rec.id
            session.status = "reconcile_required"
            self.db.commit()

        return rec

    def resolve_reconciliation(
        self,
        reconciliation_id: int,
        resolution: str,  # accept_system | manual_override | void_session
        resolved_count: int | None = None,
        resolved_by: str | None = None,
        note: str | None = None,
    ) -> ReconciliationORM | None:
        """Resolve a reconciliation case with human decision (§5.7)."""
        rec = self.db.execute(
            select(ReconciliationORM).where(ReconciliationORM.id == reconciliation_id)
        ).scalar_one_or_none()

        if not rec:
            return None

        now = datetime.now(timezone.utc)
        rec.resolution = resolution
        rec.resolved_count = resolved_count
        rec.resolved_by = resolved_by
        rec.resolved_at = now
        rec.note = note

        session = self.db.execute(select(SessionORM).where(SessionORM.id == rec.session_id)).scalar_one_or_none()
        if session:
            if resolution == "accept_system":
                session.status = "reconciled"
            elif resolution == "manual_override":
                session.status = "reconciled"
                if resolved_count is not None:
                    session.counted_total = resolved_count
            elif resolution == "void_session":
                session.status = "closed"
                session.counted_total = 0

        self.db.commit()
        self.db.refresh(rec)
        return rec

    def list_open_reconciliations(self) -> Sequence[ReconciliationORM]:
        """Fetch all unresolved reconciliation cases."""
        stmt = (
            select(ReconciliationORM)
            .where(ReconciliationORM.resolved_at == None)  # noqa: E711
            .order_by(ReconciliationORM.opened_at.desc())
        )
        return self.db.execute(stmt).scalars().all()

    def get_by_id(self, reconciliation_id: int) -> ReconciliationORM | None:
        """Fetch reconciliation case by ID."""
        return self.db.execute(
            select(ReconciliationORM).where(ReconciliationORM.id == reconciliation_id)
        ).scalar_one_or_none()
