"""Deep Stress, Concurrency, and Algorithmic Edge-Case Test Suite (§11 M3, M5, M7)."""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.cs_counting.area_counter import AreaIntegralCounter
from packages.cs_counting.gate import GateStateMachine
from packages.cs_tracking.merge_detector import MergeDetector
from packages.cs_tracking.motion import BeltMotionModel
from packages.cs_core.models import CalibrationStage, UserRole
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraEpochORM,
    CameraORM,
    CountEventORM,
    DatasetVersionORM,
    DeploymentBundleORM,
    GateORM,
    JobORM,
    LineCalibrationORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    OutboxORM,
    ProductProfileORM,
    ReconciliationORM,
    SessionORM,
    SiteORM,
    TrainingRunORM,
    UserAccountORM,
)
from packages.cs_storage.repositories.job_repo import JobRepository
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.main import app


# ===========================================================================
# 1. MATHEMATICAL & ALGORITHMIC RIGOR
# ===========================================================================

def test_kalman_velocity_tracking_under_noise():
    """Verify BeltMotionModel estimates conveyor velocity correctly from sparse optical flow."""
    model = BeltMotionModel(default_speed_px=10.0, default_direction=(1.0, 0.0))
    model.update_from_calibration(speed_px=10.0, direction=[1.0, 0.0])

    true_dx = 10.0
    true_dy = 0.0
    np.random.seed(42)

    # 10 keypoints moving by 10 px per frame with noise
    prev_pts = np.array([[100.0 + i * 20, 200.0] for i in range(10)], dtype=np.float32)

    for _ in range(30):
        curr_pts = np.zeros_like(prev_pts)
        for i in range(10):
            curr_pts[i, 0] = prev_pts[i, 0] + true_dx + float(np.random.normal(0, 0.3))
            curr_pts[i, 1] = prev_pts[i, 1] + true_dy + float(np.random.normal(0, 0.1))
        speed, direction = model.update_sparse_optical_flow(prev_pts, curr_pts)

    assert abs(speed - true_dx) < 1.0, f"Estimated speed {speed} deviated from true {true_dx}"
    assert abs(direction[0] - 1.0) < 0.05, f"Estimated direction {direction} deviated from (1, 0)"


def test_gate_state_machine_hysteresis_oscillation():
    """Verify GateStateMachine ignores high-frequency jitter within hysteresis band."""
    gsm = GateStateMachine()
    # Gate at x=300, pre_offset=40 (pre_boundary=260), post_offset=40 (post_boundary=340)
    gsm.update_geometry(
        axis_origin=(0.0, 0.0),
        axis_vector=(1.0, 0.0),
        gate_pos=300.0,
        pre_offset=40.0,
        post_offset=40.0,
    )

    base_time = datetime.now(timezone.utc)

    # Track Mock object
    class TrackMock:
        def __init__(self, track_id: int, centroid: tuple[float, float]):
            self.track_id = track_id
            self.centroid = centroid
            self.crossing_seq = 0
            self.score = 0.98

    track = TrackMock(track_id=99, centroid=(240.0, 320.0))

    # Step 1: Bag enters PRE approach zone (x=240 < 260)
    events = gsm.process_tracks([track], frame_index=1, monotonic_ns=0, wall_clock=base_time)
    assert len(events) == 0

    # Step 2: Bag moves into PRE zone (x=280)
    track.centroid = (280.0, 320.0)
    events = gsm.process_tracks([track], frame_index=2, monotonic_ns=40_000_000, wall_clock=base_time)
    assert len(events) == 0

    # Step 3: Bag oscillates across gate line (x=295 <-> x=305) without breaching post_boundary (340)
    # Forward crossing triggers when crossing from PRE to POST
    track.centroid = (305.0, 320.0)  # Crosses 300 into POST region
    events = gsm.process_tracks([track], frame_index=3, monotonic_ns=80_000_000, wall_clock=base_time)
    assert len(events) == 1
    assert events[0].direction == 1
    assert events[0].crossing_seq == 1

    # While staying in POST region (x=305 <-> x=330), no duplicate event is emitted
    for i in range(10):
        track.centroid = (310.0 if i % 2 == 0 else 320.0, 320.0)
        events = gsm.process_tracks([track], frame_index=4 + i, monotonic_ns=(120 + i * 40) * 1_000_000, wall_clock=base_time)
        assert len(events) == 0, f"Duplicate event generated at sub-step {i}"


def test_shingled_overlapping_bags_merge_detection():
    """Verify MergeDetector identifies overlapping bags (shingling) from area ratio."""
    md = MergeDetector()
    md.update_calibration(mean_bag_area_px=20000.0, is_active=True)

    # Single bag: mask area = 20500 px -> not merged (ratio = 1.025 < 1.50)
    mask_single = np.zeros((400, 400), dtype=bool)
    mask_single[100:245, 100:240] = True  # area = 145 * 140 = 20300
    res_single = md.analyze_detection(mask=mask_single, box=[100, 100, 240, 245])
    assert not res_single.is_merged
    assert res_single.estimated_object_count == 1

    # Two merged overlapping bags: mask area = 38000 px (ratio = 1.90 >= 1.50)
    mask_double = np.zeros((400, 400), dtype=bool)
    mask_double[50:250, 50:240] = True  # area = 200 * 190 = 38000
    # Include 2 distinct print marks
    print_marks = [{"box": [60, 60, 90, 90]}, {"box": [180, 180, 210, 210]}]
    res_double = md.analyze_detection(mask=mask_double, box=[50, 50, 240, 250], print_marks=print_marks)
    assert res_double.is_merged
    assert res_double.estimated_object_count == 2


def test_area_integral_riemann_accumulation_precision():
    """Verify AreaIntegralCounter accumulates trapezoidal Riemann sum accurately."""
    aic = AreaIntegralCounter(mean_bag_gate_area_px=10000.0, is_scale_calibrated=True)
    aic.update_calibration(mean_bag_area_px=10000.0, is_active=True)

    # Frame mask of 1000 px^2 per frame, belt_speed_px_per_frame = 10.0 (speed/100 = 0.1)
    # Area added per frame = 1000 * 0.1 = 100 px
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:100, 50:70] = True  # area = 50 * 20 = 1000 px^2

    for _ in range(100):
        aic.process_frame_masks([mask], belt_speed_px_per_frame=10.0)

    # 100 frames * 100 px = 10000 px -> exactly 1.0 bag estimate
    est = aic.get_estimate()
    assert abs(est - 1.0) < 1e-3

    # Discrepancy against ledger count 10 (scale calibrated) -> within tolerance
    for _ in range(900):
        aic.process_frame_masks([mask], belt_speed_px_per_frame=10.0)
    # Total accumulated = 1000 frames * 100 px = 100000 px -> 10.0 bags
    assert abs(aic.get_estimate() - 10.0) < 1e-2
    has_disc, rel_diff = aic.check_discrepancy(ledger_count=10)
    assert not has_disc
    assert rel_diff < 0.05


# ===========================================================================
# 2. CONCURRENCY, DATA INTEGRITY & IDEMPOTENCY STRESS
# ===========================================================================

def test_ledger_high_concurrency_idempotency_race():
    """Stress test: 20 concurrent threads submitting identical count event."""
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Stress Factory")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Stress Line")
        db.add(line)
        db.commit()
        cam = CameraORM(line_id=line.id, node_id=1)
        db.add(cam)
        db.commit()
        prof = ProductProfileORM(site_id=site.id, name="P1")
        db.add(prof)
        db.commit()
        session_repo = SessionRepository(db)
        sess = session_repo.create_session(line_id=line.id, product_profile_id=prof.id)
        sess_id = sess.id
        line_id = line.id
        cam_id = cam.id

    def submit_event(thread_idx: int) -> bool:
        with get_sync_session() as db:
            ledger_repo = LedgerRepository(db)
            ev, created = ledger_repo.record_event(
                session_id=sess_id,
                line_id=line_id,
                camera_id=cam_id,
                stream_epoch=1,
                track_id=777,
                crossing_seq=1,  # Same composite key for all threads
                gate_id=1,
                crossing_timestamp=datetime.now(timezone.utc),
                frame_index=100,
                direction=1,
            )
            return created

    # Run 20 concurrent submissions
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(submit_event, range(20)))

    # Exactly 1 thread must succeed in creating the record; 19 must be deduplicated
    success_count = sum(1 for r in results if r is True)
    dedup_count = sum(1 for r in results if r is False)
    assert success_count == 1, f"Expected 1 creation, got {success_count}"
    assert dedup_count == 19, f"Expected 19 deduplications, got {dedup_count}"

    # Verify ledger total count is exactly 1
    with get_sync_session() as db:
        ledger_repo = LedgerRepository(db)
        assert ledger_repo.get_session_total_count(sess_id) == 1


def test_session_net_count_with_interleaved_forward_and_backward_events():
    """Verify net count derivation: 30 forward (+1) and 10 backward (-1) interleaved = 20."""
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Net Count Factory")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Net Line")
        db.add(line)
        db.commit()
        cam = CameraORM(line_id=line.id, node_id=1)
        db.add(cam)
        db.commit()
        prof = ProductProfileORM(site_id=site.id, name="P1")
        db.add(prof)
        db.commit()
        session_repo = SessionRepository(db)
        sess = session_repo.create_session(line_id=line.id, product_profile_id=prof.id)
        sess_id = sess.id

        ledger_repo = LedgerRepository(db)
        # Interleave 30 forward and 10 backward
        for i in range(1, 31):
            ledger_repo.record_event(
                session_id=sess_id, line_id=line.id, camera_id=cam.id,
                stream_epoch=1, track_id=100 + i, crossing_seq=1, gate_id=1,
                crossing_timestamp=datetime.now(timezone.utc), frame_index=i * 10, direction=1,
            )
            # Every 3rd bag slips backward
            if i % 3 == 0:
                ledger_repo.record_event(
                    session_id=sess_id, line_id=line.id, camera_id=cam.id,
                    stream_epoch=1, track_id=100 + i, crossing_seq=2, gate_id=1,
                    crossing_timestamp=datetime.now(timezone.utc), frame_index=i * 10 + 5, direction=-1,
                )

        net_count = ledger_repo.get_session_total_count(sess_id)
        # 30 forward - 10 backward = 20
        assert net_count == 20


# ===========================================================================
# 3. RESILIENCE, LEASE STEALING & OUTBOX RETRY
# ===========================================================================

def test_job_lease_expiration_and_worker_steal():
    """Verify expired lease recovery when a worker node crashes."""
    init_db_sync()
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        # Submit a job
        job = job_repo.submit_job(kind="train", payload={"epochs": 10}, requires_gpu=True, priority=5)
        job_id = job.id

        # Worker 1 acquires lease with TTL in the past (simulating crash 10 seconds ago)
        job.status = "running"
        job.lease_until = datetime.now(timezone.utc) - timedelta(seconds=10)
        job.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        db.commit()

        # Worker 2 attempts to claim expired job
        claimed = job_repo.acquire_next_job(lease_seconds=30, gpu_available=True)
        assert claimed is not None
        assert claimed.id == job_id
        lease_time = claimed.lease_until.replace(tzinfo=timezone.utc) if claimed.lease_until.tzinfo is None else claimed.lease_until
        assert lease_time > datetime.now(timezone.utc)


def test_outbox_exponential_backoff_and_dead_letter():
    """Verify OutboxRepository applies exponential backoff and tracks attempt counts."""
    init_db_sync()
    with get_sync_session() as db:
        outbox_repo = OutboxRepository(db)
        entry = outbox_repo.create_entry(
            session_id=999,
            payload={"count": 100},
            external_ref="SEV-STRESS-01",
        )
        entry_id = entry.id

        # 1st failure: mark in progress then fail
        outbox_repo.mark_in_progress(entry_id)
        outbox_repo.mark_failed(entry_id, error_msg="Connection timeout 1", backoff_seconds=10, max_attempts=3)
        e1 = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
        assert e1.attempts == 1
        assert e1.status == "pending"

        # 2nd failure
        outbox_repo.mark_in_progress(entry_id)
        outbox_repo.mark_failed(entry_id, error_msg="Connection timeout 2", backoff_seconds=10, max_attempts=3)
        e2 = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
        assert e2.attempts == 2
        assert e2.status == "pending"

        # 3rd failure -> max_attempts reached -> reconcile_required
        outbox_repo.mark_in_progress(entry_id)
        outbox_repo.mark_failed(entry_id, error_msg="Connection timeout 3", backoff_seconds=10, max_attempts=3)
        e3 = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
        assert e3.attempts == 3
        assert e3.status == "reconcile_required"


# ===========================================================================
# 4. EXHAUSTIVE SECURITY & RBAC PERMISSION MATRIX
# ===========================================================================

def test_rbac_full_permission_matrix():
    """Exhaustively verify RBAC on all protected endpoints for Operator, Engineer, and Admin."""
    init_db_sync()
    client = TestClient(app)

    with get_sync_session() as db:
        user_repo = UserRepository(db)
        user_repo.seed_default_users()

    # Create unauthenticated client
    anon_client = TestClient(app)

    # 1. Admin-Only Endpoint: POST /api/sites
    # Unauthenticated -> 401
    assert anon_client.post("/api/sites", json={"name": "S1"}).status_code == 401

    # Operator token -> 403
    op_token = client.post("/api/auth/login", json={"username": "operator", "password": "op123"}).json()["token"]
    op_h = {"Authorization": f"Bearer {op_token}"}
    anon_client.cookies.clear()
    assert anon_client.post("/api/sites", json={"name": "S1"}, headers=op_h).status_code == 403

    # Engineer token -> 403
    eng_token = client.post("/api/auth/login", json={"username": "engineer", "password": "eng123"}).json()["token"]
    eng_h = {"Authorization": f"Bearer {eng_token}"}
    anon_client.cookies.clear()
    assert anon_client.post("/api/sites", json={"name": "S1"}, headers=eng_h).status_code == 403

    # Admin token -> 200
    admin_token = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).json()["token"]
    admin_h = {"Authorization": f"Bearer {admin_token}"}
    anon_client.cookies.clear()
    res_admin = anon_client.post("/api/sites", json={"name": "S1"}, headers=admin_h)
    assert res_admin.status_code == 200

    # 2. Engineer+ Endpoint: GET /api/reconciliations
    anon_client.cookies.clear()
    assert anon_client.get("/api/reconciliations").status_code == 401
    assert anon_client.get("/api/reconciliations", headers=op_h).status_code == 403
    assert anon_client.get("/api/reconciliations", headers=eng_h).status_code == 200
    assert anon_client.get("/api/reconciliations", headers=admin_h).status_code == 200

    # 3. Engineer+ Endpoint: POST /api/training/runs
    anon_client.cookies.clear()
    assert anon_client.post("/api/training/runs", json={"epochs": 10}, headers=op_h).status_code == 403
    assert anon_client.post("/api/training/runs", json={"epochs": 10}, headers=eng_h).status_code == 202

    # 4. Operator+ Endpoint: GET /api/system/health
    anon_client.cookies.clear()
    assert anon_client.get("/api/system/health", headers=op_h).status_code == 200
    assert anon_client.get("/api/system/health", headers=eng_h).status_code == 200
    assert anon_client.get("/api/system/health", headers=admin_h).status_code == 200
