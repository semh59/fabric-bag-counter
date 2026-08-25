"""Unit tests for Job queue leasing, heartbeat, and expired lease recovery (§5.8)."""

from datetime import datetime, timedelta, timezone
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import JobORM
from packages.cs_storage.repositories.job_repo import JobRepository


def test_job_lease_and_heartbeat():
    init_db_sync()
    with get_sync_session() as db:
        repo = JobRepository(db)
        job = repo.submit_job(kind="synthesize", payload={"count": 50}, priority=999)
        assert job.status == "queued"

        # 1. Acquire lease
        leased_job = repo.acquire_next_job(lease_seconds=30)
        assert leased_job is not None
        assert leased_job.id == job.id
        assert leased_job.status == "running"
        assert leased_job.lease_until is not None

        # 2. Heartbeat extends lease
        hb_success = repo.heartbeat(leased_job.id, extension_seconds=120)
        assert hb_success is True

        # 3. Complete job
        repo.complete_job(leased_job.id, result_payload={"scenes": 50})
        completed = repo.get_job(leased_job.id)
        assert completed.status == "completed"


def test_expired_lease_reclaim():
    init_db_sync()
    with get_sync_session() as db:
        repo = JobRepository(db)
        job = repo.submit_job(kind="extract_frames", payload={"stride": 5}, priority=998, max_attempts=3)
        leased = repo.acquire_next_job(lease_seconds=1)
        assert leased is not None
        assert leased.id == job.id

        # Manually expire the lease timestamp in database
        leased.lease_until = datetime.now(timezone.utc) - timedelta(seconds=10)
        db.commit()

        # Reclaim
        reclaimed_count = repo.reclaim_expired_leases()
        assert reclaimed_count >= 1

        reclaimed_job = repo.get_job(leased.id)
        assert reclaimed_job.status == "queued"
        assert reclaimed_job.attempts == 1
