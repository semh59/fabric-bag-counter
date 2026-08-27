"""Unit tests for Job queue leasing, heartbeat, and expired lease recovery (§5.8)."""

import threading
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


def test_concurrent_acquire_next_job_never_double_acquires():
    """acquire_next_job() must be atomic: with N workers racing for 1 job,
    exactly one worker acquires it and every other worker gets None.

    A naive "SELECT candidate then UPDATE it" implementation is vulnerable to
    two threads both selecting the same queued row before either commits its
    UPDATE, so both "acquire" the same job. This drives many concurrent
    threads (real OS threads, each with its own DB session/connection -- not
    just sequential calls) at a single queued job and asserts the winner set
    has size exactly 1.
    """
    init_db_sync()
    with get_sync_session() as db:
        repo = JobRepository(db)
        job = repo.submit_job(kind="synthesize", payload={"count": 1}, priority=1)
        job_id = job.id

    winners: list[int] = []
    errors: list[Exception] = []
    lock = threading.Lock()
    n_workers = 12

    def worker() -> None:
        try:
            with get_sync_session() as db:
                repo = JobRepository(db)
                acquired = repo.acquire_next_job(lease_seconds=60)
                if acquired is not None:
                    with lock:
                        winners.append(acquired.id)
        except Exception as exc:  # pragma: no cover - surfaced via assertion below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"acquire_next_job raised under concurrency: {errors}"
    assert winners == [job_id], f"expected exactly one winner ({job_id}), got {winners}"

    with get_sync_session() as db:
        repo = JobRepository(db)
        final = repo.get_job(job_id)
        assert final.status == "running"
        assert final.attempts == 1
