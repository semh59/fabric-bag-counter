"""Repository for Job queue leasing, heartbeats, and recovery (§5.8)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from sqlalchemy import case, or_, select, update
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import JobORM
from packages.cs_storage.repositories._dialect import is_postgres


class JobRepository:
    """Manages asynchronous background jobs with distributed lease and recovery semantics."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def submit_job(
        self,
        kind: str,
        payload: dict[str, Any],
        priority: int = 0,
        requires_gpu: bool = False,
        max_attempts: int = 3,
    ) -> JobORM:
        """Create and queue a new background job."""
        job = JobORM(
            kind=kind,
            payload=payload,
            status="queued",
            priority=priority,
            requires_gpu=requires_gpu,
            attempts=0,
            max_attempts=max_attempts,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def acquire_next_job(
        self,
        lease_seconds: int = 60,
        gpu_available: bool = True,
    ) -> JobORM | None:
        """Atomically acquire the next queued job with a lease.

        The naive "SELECT candidate, then UPDATE it" approach used previously
        is not atomic: two workers can both SELECT the same queued job before
        either commits its UPDATE, and both then "acquire" it. This is fixed
        with a single conditional UPDATE whose WHERE clause re-checks
        status == 'queued' at write time, so only the worker whose UPDATE
        actually lands first can flip the row -- the second worker's UPDATE
        matches zero rows and acquire_next_job() correctly returns None for it.

        - PostgreSQL: the candidate id is chosen via a `SELECT ... FOR UPDATE
          SKIP LOCKED` subquery, so concurrent callers don't even block on
          each other trying for different rows -- each just skips whatever
          another transaction currently has locked and picks its own
          candidate.
        - SQLite (dev/tests): `FOR UPDATE SKIP LOCKED` is Postgres-only syntax
          and SQLite has no row-level locking at all, so the subquery is left
          bare. Atomicity there instead comes from SQLite only ever running
          one writer at a time -- the single UPDATE statement (candidate
          selection + status re-check) is still one atomic operation from the
          database's point of view.
        """
        # First, recover any expired leases
        self.reclaim_expired_leases()

        now = datetime.now(timezone.utc)
        lease_until = now + timedelta(seconds=lease_seconds)

        candidate = select(JobORM.id).where(JobORM.status == "queued")
        if not gpu_available:
            candidate = candidate.where(JobORM.requires_gpu == False)  # noqa: E712
        candidate = candidate.order_by(JobORM.priority.desc(), JobORM.created_at.asc()).limit(1)
        if is_postgres(self.db):
            candidate = candidate.with_for_update(skip_locked=True)

        claim_stmt = (
            update(JobORM)
            .where(JobORM.id == candidate.scalar_subquery(), JobORM.status == "queued")
            .values(
                status="running",
                attempts=JobORM.attempts + 1,
                lease_until=lease_until,
                heartbeat_at=now,
                started_at=case((JobORM.started_at.is_(None), now), else_=JobORM.started_at),
            )
            .returning(JobORM.id)
        )
        claimed_id = self.db.execute(claim_stmt).scalars().first()
        self.db.commit()

        if claimed_id is None:
            return None
        return self.get_job(claimed_id)

    def heartbeat(self, job_id: int, extension_seconds: int = 60) -> bool:
        """Extend lease and update heartbeat timestamp."""
        job = self.db.execute(select(JobORM).where(JobORM.id == job_id)).scalar_one_or_none()
        if job and job.status == "running":
            now = datetime.now(timezone.utc)
            job.heartbeat_at = now
            job.lease_until = now + timedelta(seconds=extension_seconds)
            self.db.commit()
            return True
        return False

    def complete_job(self, job_id: int, result_payload: dict[str, Any] | None = None) -> None:
        """Mark job as successfully completed."""
        job = self.db.execute(select(JobORM).where(JobORM.id == job_id)).scalar_one_or_none()
        if job:
            job.status = "completed"
            job.finished_at = datetime.now(timezone.utc)
            if result_payload:
                job.payload = {**job.payload, "result": result_payload}
            self.db.commit()

    def fail_job(self, job_id: int, error_message: str) -> None:
        """Mark job as failed with error details."""
        job = self.db.execute(select(JobORM).where(JobORM.id == job_id)).scalar_one_or_none()
        if job:
            job.last_error = error_message
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                job.finished_at = datetime.now(timezone.utc)
            else:
                # Return to queued for retry
                job.status = "queued"
                job.lease_until = None
            self.db.commit()

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a queued or running job."""
        job = self.db.execute(select(JobORM).where(JobORM.id == job_id)).scalar_one_or_none()
        if job and job.status in ["queued", "running"]:
            job.status = "cancelled"
            job.finished_at = datetime.now(timezone.utc)
            self.db.commit()
            return True
        return False

    def reclaim_expired_leases(self) -> int:
        """Recover stalled running jobs whose lease expired."""
        now = datetime.now(timezone.utc)
        stmt = select(JobORM).where(
            JobORM.status == "running",
            JobORM.lease_until < now,
        )
        expired_jobs = self.db.execute(stmt).scalars().all()
        recovered = 0
        for job in expired_jobs:
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                job.last_error = "Job lease expired after max attempts exceeded."
                job.finished_at = now
            else:
                job.status = "queued"
                job.lease_until = None
            recovered += 1

        if recovered > 0:
            self.db.commit()
        return recovered

    def list_jobs(self, limit: int = 50) -> Sequence[JobORM]:
        """List recent jobs."""
        stmt = select(JobORM).order_by(JobORM.created_at.desc()).limit(limit)
        return self.db.execute(stmt).scalars().all()

    def get_job(self, job_id: int) -> JobORM | None:
        """Get job by ID."""
        return self.db.execute(select(JobORM).where(JobORM.id == job_id)).scalar_one_or_none()
