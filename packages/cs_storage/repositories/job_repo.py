"""Repository for Job queue leasing, heartbeats, and recovery (§5.8)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import JobORM


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
        """Atomically acquire the next queued job with a lease."""
        # First, recover any expired leases
        self.reclaim_expired_leases()

        stmt = select(JobORM).where(JobORM.status == "queued")
        if not gpu_available:
            stmt = stmt.where(JobORM.requires_gpu == False)  # noqa: E712

        stmt = stmt.order_by(JobORM.priority.desc(), JobORM.created_at.asc())
        job = self.db.execute(stmt).scalars().first()

        if job is None:
            return None

        now = datetime.now(timezone.utc)
        job.status = "running"
        job.attempts += 1
        job.lease_until = now + timedelta(seconds=lease_seconds)
        job.heartbeat_at = now
        if job.started_at is None:
            job.started_at = now

        self.db.commit()
        self.db.refresh(job)
        return job

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
