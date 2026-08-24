"""Repository for Session lifecycle and status management (§5.6)."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import SessionORM
from packages.cs_storage.repositories.ledger_repo import LedgerRepository


class SessionRepository:
    """Manages counting sessions, lifecycle transitions, and reconciliation triggers."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.ledger_repo = LedgerRepository(db)

    def create_session(
        self,
        line_id: int,
        product_profile_id: int,
        external_ref: str | None = None,
        target_count: int | None = None,
    ) -> SessionORM:
        """Create and open a new counting session."""
        session = SessionORM(
            line_id=line_id,
            product_profile_id=product_profile_id,
            external_ref=external_ref,
            target_count=target_count,
            status="open",
            opened_at=datetime.utcnow(),
            counted_total=0,
            area_estimate_total=0.0,
            discrepancy_flag=False,
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_by_id(self, session_id: int) -> SessionORM | None:
        """Fetch session by ID."""
        return self.db.execute(
            select(SessionORM).where(SessionORM.id == session_id)
        ).scalar_one_or_none()

    def get_active_session(self, line_id: int) -> SessionORM | None:
        """Fetch current non-closed session for a line."""
        stmt = select(SessionORM).where(
            SessionORM.line_id == line_id,
            SessionORM.status.in_(["open", "counting", "paused", "degraded"])
        ).order_by(SessionORM.opened_at.desc())
        return self.db.execute(stmt).scalar_one_or_none()

    def list_sessions(self, line_id: int | None = None, limit: int = 50, offset: int = 0) -> Sequence[SessionORM]:
        """List sessions with pagination."""
        stmt = select(SessionORM)
        if line_id is not None:
            stmt = stmt.where(SessionORM.line_id == line_id)
        stmt = stmt.order_by(SessionORM.opened_at.desc()).limit(limit).offset(offset)
        return self.db.execute(stmt).scalars().all()

    def update_status(self, session_id: int, status: str) -> SessionORM | None:
        """Update lifecycle status of a session."""
        session = self.get_by_id(session_id)
        if session is not None:
            session.status = status
            self.db.commit()
            self.db.refresh(session)
        return session

    def update_area_estimate(self, session_id: int, area_estimate: float) -> None:
        """Update live running area-based count estimate."""
        session = self.get_by_id(session_id)
        if session:
            session.area_estimate_total = area_estimate
            self.db.commit()

    def flag_discrepancy(self, session_id: int, area_estimate: float) -> SessionORM | None:
        """Mark session with discrepancy flag and change status to reconcile_required."""
        session = self.get_by_id(session_id)
        if session:
            session.discrepancy_flag = True
            session.area_estimate_total = area_estimate
            session.status = "reconcile_required"
            self.db.commit()
            self.db.refresh(session)
        return session

    def pause_session(self, session_id: int) -> SessionORM | None:
        """Pause counting on an open or counting session."""
        session = self.get_by_id(session_id)
        if session and session.status in ["open", "counting"]:
            session.status = "paused"
            self.db.commit()
            self.db.refresh(session)
        return session

    def resume_session(self, session_id: int) -> SessionORM | None:
        """Resume counting on a paused session."""
        session = self.get_by_id(session_id)
        if session and session.status == "paused":
            session.status = "counting"
            self.db.commit()
            self.db.refresh(session)
        return session

    def mark_degraded(self, session_id: int) -> SessionORM | None:
        """Mark session as degraded due to camera drops or observation loss."""
        session = self.get_by_id(session_id)
        if session and session.status in ["open", "counting"]:
            session.status = "degraded"
            self.db.commit()
            self.db.refresh(session)
        return session

    def close_session(self, session_id: int) -> SessionORM | None:
        """Close and lock session, deriving the final counted_total from the ledger (§5.5)."""
        session = self.get_by_id(session_id)
        if not session:
            return None

        # Derive final total directly from ledger sum(direction)
        total_count = self.ledger_repo.get_session_total_count(session_id)
        session.counted_total = total_count
        now = datetime.utcnow()
        session.closed_at = now
        session.locked_at = now

        # If session was degraded or had discrepancy, move to reconcile_required
        if session.status == "degraded" or session.discrepancy_flag:
            session.status = "reconcile_required"
        else:
            session.status = "closed"

        self.db.commit()
        self.db.refresh(session)
        return session
