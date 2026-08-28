"""Repository for Session lifecycle and status management (§5.6)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_storage.errors import ActiveSessionConflictError
from packages.cs_storage.models_orm import SessionORM
from packages.cs_storage.repositories._dialect import is_postgres
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
        vehicle_plate: str | None = None,
        driver_name: str | None = None,
        carrier_company: str | None = None,
    ) -> SessionORM:
        """Create and open a new counting session.

        Raises:
            ActiveSessionConflictError: if the target line already has a
            non-terminal (open/counting/paused/degraded) session. A line's
            ledger and area-estimate reconciliation assume a single active
            session, so a second concurrent one would corrupt counting.
            Note: this check-then-insert has an inherent race window between
            processes without a DB-level uniqueness guarantee (there is no
            existing row to lock via `with_for_update()` before the insert
            happens) -- it is a best-effort guard, not a hard database
            constraint, but it closes the common case of a double-click or a
            retried request racing itself.
        """
        existing = self.get_active_session(line_id, for_update=True)
        if existing is not None:
            raise ActiveSessionConflictError(line_id=line_id, existing_session_id=existing.id)

        session = SessionORM(
            line_id=line_id,
            product_profile_id=product_profile_id,
            external_ref=external_ref,
            target_count=target_count,
            status="open",
            opened_at=datetime.now(timezone.utc),
            counted_total=0,
            area_estimate_total=0.0,
            discrepancy_flag=False,
            vehicle_plate=vehicle_plate,
            driver_name=driver_name,
            carrier_company=carrier_company,
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_by_id(self, session_id: int, for_update: bool = False) -> SessionORM | None:
        """Fetch session by ID.

        for_update: lock the row (`SELECT ... FOR UPDATE`) so a concurrent
        reader blocks until this transaction commits/rolls back. Only takes
        effect on PostgreSQL (see packages.cs_storage.repositories._dialect) --
        SQLite has no row-level locking and the test suite runs on SQLite, so
        this is a no-op there and callers must not rely on it for isolation
        in tests.
        """
        stmt = select(SessionORM).where(SessionORM.id == session_id)
        if for_update and is_postgres(self.db):
            stmt = stmt.with_for_update()
        return self.db.execute(stmt).scalar_one_or_none()

    def get_active_session(self, line_id: int, for_update: bool = False) -> SessionORM | None:
        """Fetch current non-closed session for a line.

        Uses order_by(...).first() rather than scalar_one_or_none(): the
        latter raises if more than one row matches, which would turn a data
        integrity bug (two active sessions somehow existing) into a hard
        crash on every subsequent read of this line. Ordering by most recent
        and taking the first is defense-in-depth -- it degrades gracefully
        and still returns the session callers almost certainly mean.
        """
        stmt = select(SessionORM).where(
            SessionORM.line_id == line_id,
            SessionORM.status.in_(["open", "counting", "paused", "degraded"])
        ).order_by(SessionORM.opened_at.desc())
        if for_update and is_postgres(self.db):
            stmt = stmt.with_for_update()
        return self.db.execute(stmt).scalars().first()

    def list_sessions(self, line_id: int | None = None, limit: int = 50, offset: int = 0) -> Sequence[SessionORM]:
        """List sessions with pagination."""
        stmt = select(SessionORM)
        if line_id is not None:
            stmt = stmt.where(SessionORM.line_id == line_id)
        stmt = stmt.order_by(SessionORM.opened_at.desc()).limit(limit).offset(offset)
        return self.db.execute(stmt).scalars().all()

    def update_status(self, session_id: int, status: str) -> SessionORM | None:
        """Update lifecycle status of a session."""
        session = self.get_by_id(session_id, for_update=True)
        if session is not None:
            session.status = status
            self.db.commit()
            self.db.refresh(session)
        return session

    def update_area_estimate(self, session_id: int, area_estimate: float) -> None:
        """Update live running area-based count estimate."""
        session = self.get_by_id(session_id, for_update=True)
        if session:
            session.area_estimate_total = area_estimate
            self.db.commit()

    def update_counted_total(self, session_id: int, counted_total: int) -> None:
        """Update the live running net bag count.

        Previously every caller (LiveStreamRenderer, InferenceWorker,
        simulate_bag_crossing) set `session.counted_total = ...` directly on
        the ORM object it had separately fetched, instead of through a repo
        method -- the same duplication CountingEventHandler
        (packages/cs_counting/event_handler.py) now closes for the
        ledger-write side of this same update.
        """
        session = self.get_by_id(session_id, for_update=True)
        if session:
            session.counted_total = counted_total
            self.db.commit()

    @staticmethod
    def _set_reconcile_required(session: SessionORM) -> None:
        """Set the reconcile_required transition on an already-fetched,
        not-yet-committed session object -- the one place this assignment
        happens. Not itself a public method: callers that already hold the
        session mid-transaction (flag_discrepancy, close_session below) use
        this so the transition lands in their single existing commit rather
        than a separate one, avoiding a window where a reader could observe
        e.g. discrepancy_flag=True but status not yet reconcile_required.
        External callers with no open transaction of their own use
        require_reconciliation() instead.
        """
        session.status = "reconcile_required"

    def require_reconciliation(self, session_id: int, reason: str) -> None:
        """Transition a session to reconcile_required (fetch + commit).

        For callers with no already-open transaction on this session, e.g.
        ReconciliationRepository.create_reconciliation() -- three call sites
        each used to inline `session.status = "reconcile_required"`
        independently (flag_discrepancy, close_session, and that one) before
        this method existed; the transition itself was copied three times
        instead of shared, though each keeps its own trigger logic for
        *when* to call this. `reason` is accepted for future audit logging
        even though nothing persists it yet, so call sites don't need to
        change again when that lands.
        """
        session = self.get_by_id(session_id, for_update=True)
        if session:
            self._set_reconcile_required(session)
            self.db.commit()

    def flag_discrepancy(self, session_id: int, area_estimate: float) -> SessionORM | None:
        """Mark session with discrepancy flag and change status to reconcile_required."""
        session = self.get_by_id(session_id, for_update=True)
        if session:
            session.discrepancy_flag = True
            session.area_estimate_total = area_estimate
            self._set_reconcile_required(session)
            self.db.commit()
            self.db.refresh(session)
        return session

    def pause_session(self, session_id: int) -> SessionORM | None:
        """Pause counting on an open or counting session."""
        session = self.get_by_id(session_id, for_update=True)
        if session and session.status in ["open", "counting"]:
            session.status = "paused"
            self.db.commit()
            self.db.refresh(session)
        return session

    def resume_session(self, session_id: int) -> SessionORM | None:
        """Resume counting on a paused session."""
        session = self.get_by_id(session_id, for_update=True)
        if session and session.status == "paused":
            session.status = "counting"
            self.db.commit()
            self.db.refresh(session)
        return session

    def mark_degraded(self, session_id: int) -> SessionORM | None:
        """Mark session as degraded due to camera drops or observation loss."""
        session = self.get_by_id(session_id, for_update=True)
        if session and session.status in ["open", "counting"]:
            session.status = "degraded"
            self.db.commit()
            self.db.refresh(session)
        return session

    def close_session(self, session_id: int) -> SessionORM | None:
        """Close and lock session, deriving the final counted_total from the ledger (§5.5)."""
        session = self.get_by_id(session_id, for_update=True)
        if not session:
            return None

        # Derive final total directly from ledger sum(direction)
        total_count = self.ledger_repo.get_session_total_count(session_id)
        session.counted_total = total_count
        now = datetime.now(timezone.utc)
        session.closed_at = now
        session.locked_at = now

        # If session was degraded or had discrepancy, move to reconcile_required
        if session.status == "degraded" or session.discrepancy_flag:
            self._set_reconcile_required(session)
        else:
            session.status = "closed"

        self.db.commit()
        self.db.refresh(session)
        return session
