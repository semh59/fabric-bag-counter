"""Repository for the Count Event Ledger (§5.5) — single source of truth."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Sequence
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import CountEventORM


class LedgerRepository:
    """Manages appending and querying count events with strict idempotency."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def record_event(
        self,
        session_id: int,
        line_id: int,
        camera_id: int,
        stream_epoch: int,
        track_id: int,
        crossing_seq: int,
        gate_id: int,
        crossing_timestamp: datetime,
        frame_index: int,
        direction: int,
        confidence: float | None = None,
        merge_flag: bool = False,
        deployment_bundle_id: int = 1,
        evidence_ref: str | None = None,
        event_id: str | None = None,
        defect_reason: str | None = None,
        is_simulated: bool = False,
    ) -> tuple[CountEventORM | None, bool]:
        """Record a crossing event in the ledger.
        
        Returns:
            (event, created): tuple where created is True if newly inserted,
            or False if ignored due to idempotency constraint duplicate.
        """
        eid = event_id or str(uuid.uuid4())
        event = CountEventORM(
            event_id=eid,
            session_id=session_id,
            line_id=line_id,
            camera_id=camera_id,
            stream_epoch=stream_epoch,
            track_id=track_id,
            crossing_seq=crossing_seq,
            gate_id=gate_id,
            crossing_timestamp=crossing_timestamp,
            frame_index=frame_index,
            direction=direction,
            confidence=confidence,
            merge_flag=merge_flag,
            deployment_bundle_id=deployment_bundle_id,
            evidence_ref=evidence_ref,
            defect_reason=defect_reason,
            is_simulated=is_simulated,
        )
        try:
            self.db.add(event)
            self.db.commit()
            return event, True
        except IntegrityError:
            self.db.rollback()
            # Already exists (idempotency hit)
            existing = self.db.execute(
                select(CountEventORM).where(
                    CountEventORM.session_id == session_id,
                    CountEventORM.camera_id == camera_id,
                    CountEventORM.stream_epoch == stream_epoch,
                    CountEventORM.track_id == track_id,
                    CountEventORM.gate_id == gate_id,
                    CountEventORM.crossing_seq == crossing_seq,
                )
            ).scalar_one_or_none()
            return existing, False

    def get_session_total_count(self, session_id: int) -> int:
        """Derive the true net bag count directly from the immutable ledger.
        
        Formula: SELECT COALESCE(SUM(direction), 0) FROM count_event WHERE session_id = :sid;
        """
        stmt = select(func.coalesce(func.sum(CountEventORM.direction), 0)).where(
            CountEventORM.session_id == session_id
        )
        result = self.db.execute(stmt).scalar()
        return int(result) if result is not None else 0

    def get_session_events(
        self,
        session_id: int,
        limit: int = 1000,
        offset: int = 0,
    ) -> Sequence[CountEventORM]:
        """Fetch chronological count events for an audit / session detail view."""
        stmt = (
            select(CountEventORM)
            .where(CountEventORM.session_id == session_id)
            .order_by(CountEventORM.crossing_timestamp.asc(), CountEventORM.frame_index.asc())
            .limit(limit)
            .offset(offset)
        )
        return self.db.execute(stmt).scalars().all()

    def get_defect_events(self, line_id: int | None = None, limit: int = 200) -> Sequence[CountEventORM]:
        """Audit log of counted bags later excluded as defective (§ post-gate exclusion)."""
        stmt = select(CountEventORM).where(CountEventORM.defect_reason.isnot(None))
        if line_id is not None:
            stmt = stmt.where(CountEventORM.line_id == line_id)
        stmt = stmt.order_by(CountEventORM.crossing_timestamp.desc()).limit(limit)
        return self.db.execute(stmt).scalars().all()

    def get_merge_count(self, session_id: int) -> int:
        """Count total merge flag occurrences in a session."""
        stmt = select(func.count(CountEventORM.event_id)).where(
            CountEventORM.session_id == session_id,
            CountEventORM.merge_flag == True,  # noqa: E712
        )
        result = self.db.execute(stmt).scalar()
        return int(result) if result is not None else 0
