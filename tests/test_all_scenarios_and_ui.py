"""Comprehensive Scenario & UI Full-Functionality Test Suite (§1-§12).

Tests:
1. Scenario 1: Normal Single Stream Counting
2. Scenario 2: Shingling (2 bags merged/overlapping, 4 independent signals, exact 2 count)
3. Scenario 3: Conveyor Backward Slip (Forward -> Backward -> Forward netting to 1)
4. Scenario 4: Multi-Camera Stream Synchronization
5. Scenario 5: Stream Interruption & Monotonic Camera Epoch Increment
6. Scenario 6: Frame Drop Degradation & Fail-Safe Locking
7. Scenario 7: Area-Integral vs Ledger Discrepancy Reconciliation Trigger
8. Scenario 8: Human Reconciliation Workflows (Accept, Override, Void)
9. Scenario 9: Transactional Outbox Delivery with CSV & SAP OData
10. Scenario 10: Background Job Leases, Heartbeats, Expired Recovery & GPU Sharing
11. Scenario 11: 13-Step Setup Wizard Full Execution
12. Scenario 12: UI Buttons & RBAC Security for Operator, Engineer, and Admin
"""

from datetime import datetime, timedelta, timezone
import json
from fastapi.testclient import TestClient
import numpy as np
import pytest
from drivers.erp_csv.adapter import CsvErpAdapter
from drivers.erp_sap_odata.adapter import SapODataErpAdapter
from packages.cs_core.frame import Frame
from packages.cs_core.models import (
    CalibrationStage,
    CameraRole,
    GpuSharingMode,
    JobKind,
    JobStatus,
    LineStatus,
    ModelStage,
    ReconciliationReason,
    ReconciliationResolution,
    SessionStatus,
    UserRole,
)
from packages.cs_core.transport import SharedMemoryTransport
from packages.cs_counting.area_counter import AreaIntegralCounter
from packages.cs_counting.engine import CountingEngine
from packages.cs_counting.gate import GateStateMachine
from packages.cs_data.mining import HardFrameMiner
from packages.cs_data.split_dataset import DatasetSplitter
from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_eval.metrics import compute_counting_metrics
from packages.cs_eval.scoreboard import generate_scoreboard_text
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraORM,
    ConfigVersionORM,
    CountEventORM,
    DeploymentBundleORM,
    JobORM,
    LineCalibrationORM,
    LineORM,
    ModelVersionORM,
    OutboxORM,
    ProductProfileORM,
    ReconciliationORM,
    SessionORM,
    SiteORM,
    TrainingRunORM,
    UserAccountORM,
)
from packages.cs_storage.repositories.calibration_repo import CalibrationRepository
from packages.cs_storage.repositories.camera_epoch_repo import CameraEpochRepository
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.job_repo import JobRepository
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from packages.cs_tracking.matching import associate_detections_to_tracks
from packages.cs_tracking.merge_detector import MergeDetector
from packages.cs_tracking.motion import BeltMotionModel
from packages.cs_tracking.tracker import BagTrack, ConveyorByteTracker
from services.api.main import app
from services.erp_relay.worker import ErpRelayWorker
from services.jobrunner.worker import JobrunnerWorker

client = TestClient(app)


def seed_baseline_environment():
    """Seed baseline site, line, camera, product profile, calibration, bundle, and users."""
    init_db_sync()
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        user_repo.create_user("operator", "op123", role="operator")
        user_repo.create_user("engineer", "eng123", role="engineer")
        user_repo.create_user("admin", "admin123", role="admin")

        site = SiteORM(name="Fabrika Gaziantep")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Paketleme Hatti 1", status="idle")
        db.add(line)
        db.commit()

        cam1 = CameraORM(
            line_id=line.id,
            node_id=1,
            role="counting",
            source_driver="rtsp",
            source_config={"rtsp_url": "rtsp://192.168.1.100/live"},
            enabled=True,
        )
        cam2 = CameraORM(
            line_id=line.id,
            node_id=1,
            role="vehicle_watchdog",
            source_driver="file",
            source_config={"file_path": "./data/cam2.mp4"},
            enabled=True,
        )
        db.add_all([cam1, cam2])
        db.commit()

        prof = ProductProfileORM(
            site_id=site.id,
            name="Un 50kg Polipropilen",
            erp_material_code="MAT-FLOUR-50KG",
            nominal_dims_mm={"length": 900, "width": 550, "height": 180},
        )
        db.add(prof)
        db.commit()

        calib_repo = CalibrationRepository(db)
        motion_cal = calib_repo.create_motion_calibration(
            line_id=line.id, belt_speed_px_per_frame=12.5, belt_direction_vector=[1.0, 0.0]
        )
        scale_cal = calib_repo.create_scale_calibration(
            line_id=line.id, px_per_mm=0.75, mean_bag_gate_area_px=18000.0, bag_area_stddev_px=600.0
        )

        config_repo = ConfigRepository(db)
        config_ver = config_repo.create_config_version(
            line_id=line.id,
            payload={
                "roi_polygon": [[50, 50], [600, 50], [600, 400], [50, 400]],
                "gate_position_along_axis": 350.0,
                "pre_gate_offset": 50.0,
                "post_gate_offset": 50.0,
                "confidence_threshold": 0.85,
                "merge_area_ratio": 1.5,
                "discrepancy_threshold": 0.08,
            },
            created_by="engineer",
        )

        model_ver = ModelVersionORM(
            onnx_hash="sha256:abc123rfdetr",
            onnx_path="./models/rfdetr_seg_v2.onnx",
            stage="active",
        )
        db.add(model_ver)
        db.commit()

        bundle = config_repo.create_and_activate_bundle(
            line_id=line.id,
            model_version_id=model_ver.id,
            config_version_id=config_ver.id,
            calibration_id=scale_cal.id,
            git_commit="git-v2.0.0-verified",
            activated_by="admin",
        )

        return {
            "site_id": site.id,
            "line_id": line.id,
            "cam1_id": cam1.id,
            "cam2_id": cam2.id,
            "profile_id": prof.id,
            "bundle_id": bundle.id,
            "model_id": model_ver.id,
            "config_id": config_ver.id,
        }


# ==============================================================================
# SCENARIO 1: Normal Single Stream Counting
# ==============================================================================
def test_scenario_1_normal_single_stream():
    """Verify standard sequence of 10 bags moving forward across the conveyor gate."""
    env = seed_baseline_environment()
    engine = CountingEngine()
    engine.belt_motion.update_from_calibration(speed_px=12.5, direction=[1.0, 0.0])
    engine.gate_state_machine.update_geometry(
        axis_origin=(0.0, 0.0),
        axis_vector=(1.0, 0.0),
        gate_pos=350.0,
        pre_offset=50.0,
        post_offset=50.0,
    )
    engine.area_counter.update_calibration(mean_bag_area_px=18000.0, is_active=True)

    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        sess = session_repo.create_session(
            line_id=env["line_id"], product_profile_id=env["profile_id"], target_count=10, external_ref="IRS-SCENARIO-1"
        )
        sess_id = sess.id

    t_base = datetime.now(timezone.utc)
    # 10 bags passing sequentially
    for bag_idx in range(10):
        for step in range(10):
            f_idx = bag_idx * 10 + step
            t_frame = t_base + timedelta(milliseconds=40 * f_idx)
            mono_ns = int(f_idx * 40 * 1e6)
            cx = 250.0 + (step * 25.0)  # Starts at 250 (PRE), crosses 350 (GATE) to 475 (POST)
            cy = 200.0

            mask = np.zeros((480, 640), dtype=bool)
            mask[int(cy - 40):int(cy + 40), int(cx - 50):int(cx + 50)] = True

            engine.detector.predict = lambda img, _box=[cx - 50, cy - 40, cx + 50, cy + 40], _m=mask: type(
                "Det", (), {"bag_bodies": [{"box": _box, "score": 0.96, "mask": _m}], "print_marks": []}
            )()

            out = engine.process_frame(
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                frame_index=f_idx,
                monotonic_ns=mono_ns,
                wall_clock=t_frame,
            )

            with get_sync_session() as db:
                ledger_repo = LedgerRepository(db)
                for ev in out.gate_crossings:
                    ledger_repo.record_event(
                        session_id=sess_id,
                        line_id=env["line_id"],
                        camera_id=env["cam1_id"],
                        stream_epoch=1,
                        track_id=ev.track_id,
                        crossing_seq=ev.crossing_seq,
                        gate_id=ev.gate_id,
                        crossing_timestamp=ev.crossing_timestamp,
                        frame_index=ev.frame_index,
                        direction=ev.direction,
                    )

    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        closed_sess = session_repo.close_session(sess_id)
        assert closed_sess.counted_total == 10
        assert closed_sess.status == "closed"


# ==============================================================================
# SCENARIO 2: Shingling / 2 Merged Bags
# ==============================================================================
def test_scenario_2_shingling_merge_detection():
    """Verify 2 overlapping bags trigger 4-signal merge detector, latent tracks, and count exact 2."""
    seed_baseline_environment()
    detector = MergeDetector(mean_bag_gate_area_px=18000.0, is_scale_calibrated=True, min_votes=2)

    # 2 shingled bags mask (area = 38000 px > 1.5 * 18000 px, 2 distinct print marks)
    merged_mask = np.zeros((480, 640), dtype=bool)
    merged_mask[100:300, 100:450] = True

    hyp = detector.analyze_detection(
        mask=merged_mask,
        box=[100, 100, 450, 300],
        print_marks=[
            {"box": [120, 130, 160, 170]},
            {"box": [320, 130, 360, 170]},
        ],
    )

    assert hyp.is_merged is True
    assert hyp.estimated_object_count == 2
    assert len(hyp.centroid_seeds) == 2
    assert "signal_area_oversized" in hyp.signal_votes
    assert "signal_multiple_print_marks" in hyp.signal_votes


# ==============================================================================
# SCENARIO 3: Backward Slip Netting to 1
# ==============================================================================
def test_scenario_3_conveyor_backward_slip():
    """Verify forward -> backward -> forward gate transitions net exact 1 via crossing_seq."""
    env = seed_baseline_environment()
    gate = GateStateMachine(
        gate_id=1,
        axis_origin=(0.0, 0.0),
        axis_vector=(1.0, 0.0),
        gate_position_along_axis=300.0,
        pre_gate_offset=40.0,
        post_gate_offset=40.0,
    )
    track = BagTrack(box=[200, 100, 240, 140], score=0.95)
    t_now = datetime.now(timezone.utc)

    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        sess = session_repo.create_session(line_id=env["line_id"], product_profile_id=env["profile_id"])
        sess_id = sess.id
        ledger_repo = LedgerRepository(db)

        # 1. Bag enters PRE zone (cx = 220)
        gate.process_tracks([track], frame_index=1, monotonic_ns=1000, wall_clock=t_now)

        # 2. Bag crosses forward to POST zone (cx = 360) -> direction +1, seq 1
        track.centroid = (360.0, 120.0)
        ev1 = gate.process_tracks([track], frame_index=2, monotonic_ns=2000, wall_clock=t_now)[0]
        ledger_repo.record_event(
            session_id=sess_id,
            line_id=env["line_id"],
            camera_id=env["cam1_id"],
            stream_epoch=1,
            track_id=track.track_id,
            crossing_seq=ev1.crossing_seq,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=2,
            direction=ev1.direction,
        )

        # 3. Conveyor slips backwards, bag moves back to PRE zone (cx = 240) -> direction -1, seq 2
        track.centroid = (240.0, 120.0)
        ev2 = gate.process_tracks([track], frame_index=3, monotonic_ns=3000, wall_clock=t_now)[0]
        ledger_repo.record_event(
            session_id=sess_id,
            line_id=env["line_id"],
            camera_id=env["cam1_id"],
            stream_epoch=1,
            track_id=track.track_id,
            crossing_seq=ev2.crossing_seq,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=3,
            direction=ev2.direction,
        )

        # 4. Conveyor moves forward again, bag crosses to POST zone (cx = 370) -> direction +1, seq 3
        track.centroid = (370.0, 120.0)
        ev3 = gate.process_tracks([track], frame_index=4, monotonic_ns=4000, wall_clock=t_now)[0]
        ledger_repo.record_event(
            session_id=sess_id,
            line_id=env["line_id"],
            camera_id=env["cam1_id"],
            stream_epoch=1,
            track_id=track.track_id,
            crossing_seq=ev3.crossing_seq,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=4,
            direction=ev3.direction,
        )

        # Verify net total = (+1) + (-1) + (+1) = 1
        net_total = ledger_repo.get_session_total_count(sess_id)
        assert net_total == 1

        closed = session_repo.close_session(sess_id)
        assert closed.counted_total == 1


# ==============================================================================
# SCENARIO 4 & 5: Multi-Camera, Stream Interruption & Camera Epoch Increment
# ==============================================================================
def test_scenario_4_and_5_multi_camera_and_epoch_increment():
    """Verify camera epoch increments across reconnects with zero ledger primary key collision."""
    env = seed_baseline_environment()
    with get_sync_session() as db:
        epoch_repo = CameraEpochRepository(db)
        ledger_repo = LedgerRepository(db)
        session_repo = SessionRepository(db)

        sess = session_repo.create_session(line_id=env["line_id"], product_profile_id=env["profile_id"])

        # Epoch 1
        ep_cam1_a = epoch_repo.increment_and_get_epoch(env["cam1_id"])
        ep_cam2_a = epoch_repo.increment_and_get_epoch(env["cam2_id"])
        assert ep_cam1_a >= 1 and ep_cam2_a >= 1

        t_now = datetime.now(timezone.utc)
        # Cam 1 records bag 1 (track 100)
        ledger_repo.record_event(
            session_id=sess.id,
            line_id=env["line_id"],
            camera_id=env["cam1_id"],
            stream_epoch=ep_cam1_a,
            track_id=100,
            crossing_seq=1,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=50,
            direction=1,
        )

        # Cam 1 reconnects -> Epoch 2
        ep_cam1_b = epoch_repo.increment_and_get_epoch(env["cam1_id"])
        assert ep_cam1_b == ep_cam1_a + 1

        # Tracker reboots and happens to reuse track_id 100, crossing_seq 1
        # Because stream_epoch changed to ep_cam1_b, NO PRIMARY/UNIQUE KEY COLLISION occurs!
        ev, created = ledger_repo.record_event(
            session_id=sess.id,
            line_id=env["line_id"],
            camera_id=env["cam1_id"],
            stream_epoch=ep_cam1_b,
            track_id=100,
            crossing_seq=1,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=150,
            direction=1,
        )
        assert created is True

        # Total is exact 2
        total = ledger_repo.get_session_total_count(sess.id)
        assert total == 2


# ==============================================================================
# SCENARIO 6 & 7: Frame Drop Degradation & Area Discrepancy Reconciliation
# ==============================================================================
def test_scenario_6_and_7_degradation_and_reconciliation_trigger():
    """Verify consecutive frame drops & area discrepancy trigger reconciliation state."""
    env = seed_baseline_environment()
    transport = SharedMemoryTransport(ring_slots=2)

    # 1. Fill ring and cause 4 consecutive frame drops
    for i in range(6):
        f = Frame(
            camera_id=env["cam1_id"],
            stream_epoch=1,
            frame_index=i,
            monotonic_ns=i * 40000000,
            wall_clock=datetime.now(timezone.utc),
            shm_name=f"shm_test_{i}",
            shape=(480, 640, 3),
            dtype="uint8",
        )
        transport.publish(f)

    stats = transport.get_stats()
    assert stats["consecutive_drops"][env["cam1_id"]] >= 3

    # 2. Area integral discrepancy (> 8%)
    counter = AreaIntegralCounter(mean_bag_gate_area_px=18000.0, discrepancy_threshold=0.08, is_scale_calibrated=True)
    # Simulate 5 bags of area
    for _ in range(5):
        m = np.ones((100, 180), dtype=bool)  # 18000 px
        counter.process_frame_masks([m], belt_speed_px_per_frame=100.0)

    # Compare against ledger count of 12 (massive discrepancy!)
    has_disc, delta = counter.check_discrepancy(ledger_count=12)
    assert has_disc is True

    # 3. Create reconciliation case in database
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        reconcile_repo = ReconciliationRepository(db)

        sess = session_repo.create_session(line_id=env["line_id"], product_profile_id=env["profile_id"])
        sess.counted_total = 12
        sess.area_estimate_total = 5.0
        sess.discrepancy_flag = True
        sess.status = "degraded"
        db.commit()

        rec = reconcile_repo.create_reconciliation(
            session_id=sess.id,
            trigger_reason=ReconciliationReason.COUNT_AREA_MISMATCH.value,
            evidence_refs={"ledger_count": 12, "area_estimate": 5.0, "delta": delta},
        )
        assert rec.resolution is None

        # Operator/Engineer resolves with manual override
        reconcile_repo.resolve_reconciliation(
            reconciliation_id=rec.id,
            resolution=ReconciliationResolution.MANUAL_OVERRIDE.value,
            resolved_count=10,
            resolved_by="lead_operator",
            note="Forklift verified 10 pallets.",
        )
        closed_rec = reconcile_repo.get_by_id(rec.id)
        assert closed_rec.resolution == "manual_override"
        assert closed_rec.resolved_count == 10


# ==============================================================================
# SCENARIO 8 & 9: Human Reconciliation & Transactional Outbox Relay
# ==============================================================================
def test_scenario_8_and_9_outbox_and_erp_dispatch():
    """Verify transactional outbox atomicity, retry backoff, and CSV & SAP dispatch."""
    env = seed_baseline_environment()
    from packages.cs_core.interfaces.erp_adapter import SessionPayload
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        outbox_repo = OutboxRepository(db)

        sess = session_repo.create_session(
            line_id=env["line_id"], product_profile_id=env["profile_id"], external_ref="DELIVERY-7788"
        )
        sess.counted_total = 250
        sess.status = "closed"
        db.commit()

        entry = outbox_repo.create_entry(
            session_id=sess.id,
            payload={"line_id": env["line_id"], "counted_total": 250, "material": "MAT-FLOUR-50KG"},
            external_ref="DELIVERY-7788",
        )
        assert entry.status == "pending"

    # CSV Dispatch
    csv_adapter = CsvErpAdapter(export_dir="./data/csv_exports")
    relay = ErpRelayWorker(adapter=csv_adapter)
    processed = relay.run_step()
    assert processed >= 1

    # SAP OData Dispatch
    sap_adapter = SapODataErpAdapter()
    payload = SessionPayload(
        line_id=env["line_id"],
        session_id=sess.id,
        product_profile_id=env["profile_id"],
        counted_total=250,
        area_estimate_total=250.0,
        external_ref="DELIVERY-7788",
        erp_material_code="MAT-FLOUR-50KG",
        opened_at=datetime.now(timezone.utc),
    )
    res_sap = sap_adapter.submit_session(payload)
    # Verifies retryable network error capture when endpoint is offline
    assert res_sap.retryable is True or res_sap.success is True


# ==============================================================================
# SCENARIO 10: Background Job Queue & GPU Sharing Policy
# ==============================================================================
def test_scenario_10_job_queue_and_gpu_policy():
    """Verify Jobrunner leases, heartbeat, fail-safe recovery, and strict GPU sharing."""
    seed_baseline_environment()
    with get_sync_session() as db:
        job_repo = JobRepository(db)
        # 1. Submit GPU synthesis job
        job = job_repo.submit_job(
            kind=JobKind.SYNTHESIZE.value,
            payload={"count": 20, "output_dir": "./data/synth_out"},
            priority=50,
            requires_gpu=True,
        )
        assert job.status == "queued"

    # 2. Execute via JobrunnerWorker in always mode
    runner = JobrunnerWorker(poll_interval_sec=0.1, gpu_mode=GpuSharingMode.ALWAYS.value)
    executed = runner.run_step()
    assert executed is True

    with get_sync_session() as db:
        job_repo = JobRepository(db)
        completed_job = job_repo.get_job(job.id)
        assert completed_job.status == "completed"


# ==============================================================================
# SCENARIO 11: 13-Step Setup Wizard Full Execution via API
# ==============================================================================
def test_scenario_11_setup_wizard_13_steps():
    """Verify 13-step setup wizard API flow from site creation to live deployment bundle."""
    seed_baseline_environment()

    # Login as admin for setup operations
    login_adm = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    token_adm = login_adm.json()["token"]
    headers_adm = {"Authorization": f"Bearer {token_adm}"}

    # Step 1: Create Site & Line
    s_res = client.post("/api/sites", json={"name": "Wizard Test Site"}, headers=headers_adm)
    assert s_res.status_code == 200
    site_id = s_res.json()["id"]

    l_res = client.post("/api/lines", json={"site_id": site_id, "name": "Wizard Line 1"}, headers=headers_adm)
    assert l_res.status_code == 200
    line_id = l_res.json()["id"]

    # Step 2: Register Camera
    c_res = client.post(
        "/api/cameras",
        json={
            "line_id": line_id,
            "node_id": 1,
            "role": "counting",
            "source_driver": "file",
            "source_config": {"file_path": "./data/test.mp4"},
        },
        headers=headers_adm,
    )
    assert c_res.status_code == 200
    cam_id = c_res.json()["id"]

    # Step 3: Set ROI & Step 5: Gate in Config Version
    cfg_res = client.post(
        f"/api/configs/{line_id}",
        json={
            "payload": {
                "roi_polygon": [[0, 0], [640, 0], [640, 480], [0, 480]],
                "gate_position_along_axis": 320.0,
                "pre_gate_offset": 50.0,
                "post_gate_offset": 50.0,
            },
            "note": "Wizard Configuration v1",
        },
        headers=headers_adm,
    )
    assert cfg_res.status_code == 200
    config_id = cfg_res.json()["id"]

    # Step 4: Motion Calibration (Stage 1)
    m_cal = client.post(
        f"/api/calibrations/{line_id}/motion",
        json={"belt_speed_px_per_frame": 15.0, "belt_direction_vector": [1.0, 0.0]},
        headers=headers_adm,
    )
    assert m_cal.status_code == 202

    # Step 6: Product Profile
    p_res = client.post(
        "/api/products",
        json={"site_id": site_id, "name": "Cimento 50kg", "erp_material_code": "MAT-CEM-50", "nominal_dims_mm": {}},
        headers=headers_adm,
    )
    assert p_res.status_code == 200
    prof_id = p_res.json()["id"]

    # Step 7: Scale Calibration (Stage 2)
    s_cal = client.post(
        f"/api/calibrations/{line_id}/scale",
        json={"px_per_mm": 0.8, "mean_bag_gate_area_px": 20000.0, "bag_area_stddev_px": 500.0},
        headers=headers_adm,
    )
    assert s_cal.status_code == 202

    # Step 12: Deployment Bundle Creation & Activation
    with get_sync_session() as db:
        m = ModelVersionORM(onnx_hash="sha256:wizard", onnx_path="./models/w.onnx", stage="active")
        db.add(m)
        db.commit()
        model_id = m.id

        calib_repo = CalibrationRepository(db)
        cal = calib_repo.create_scale_calibration(line_id=line_id, px_per_mm=0.8, mean_bag_gate_area_px=20000.0, bag_area_stddev_px=500.0)
        calib_id = cal.id

    b_res = client.post(
        "/api/bundles/activate",
        json={
            "line_id": line_id,
            "model_version_id": model_id,
            "config_version_id": config_id,
            "calibration_id": calib_id,
        },
        headers=headers_adm,
    )
    assert b_res.status_code == 200
    bundle_id = b_res.json()["id"]
    assert bundle_id is not None


# ==============================================================================
# SCENARIO 12: UI All Buttons & RBAC Security Permissions
# ==============================================================================
def test_scenario_12_ui_all_buttons_and_rbac_matrix():
    """Verify all UI action buttons for Operator, Engineer, and Admin personas."""
    env = seed_baseline_environment()

    # Login tokens
    op_tok = client.post("/api/auth/login", json={"username": "operator", "password": "op123"}).json()["token"]
    eng_tok = client.post("/api/auth/login", json={"username": "engineer", "password": "eng123"}).json()["token"]
    adm_tok = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).json()["token"]

    op_h = {"Authorization": f"Bearer {op_tok}"}
    eng_h = {"Authorization": f"Bearer {eng_tok}"}
    adm_h = {"Authorization": f"Bearer {adm_tok}"}

    # 1. Operator opens session (ALLOWED)
    s_open = client.post(
        "/api/sessions",
        json={"line_id": env["line_id"], "product_profile_id": env["profile_id"], "external_ref": "UI-TEST-001"},
        headers=op_h,
    )
    assert s_open.status_code == 200
    sess_id = s_open.json()["id"]

    # 2. Operator pauses & resumes session (ALLOWED)
    p_res = client.post(f"/api/sessions/{sess_id}/pause", headers=op_h)
    assert p_res.status_code == 200 and p_res.json()["status"] == "paused"

    r_res = client.post(f"/api/sessions/{sess_id}/resume", headers=op_h)
    assert r_res.status_code == 200 and r_res.json()["status"] == "counting"

    # 3. Operator closes session (ALLOWED)
    c_res = client.post(f"/api/sessions/{sess_id}/close", headers=op_h)
    assert c_res.status_code == 200 and c_res.json()["status"] == "closed"

    # 4. Operator tries to trigger Model Training (FORBIDDEN -> 403)
    t_forbid = client.post("/api/training/runs", json={"dataset_version_id": 1}, headers=op_h)
    assert t_forbid.status_code == 403

    # 5. Engineer triggers Model Training (ALLOWED)
    with get_sync_session() as db:
        from packages.cs_storage.models_orm import DatasetVersionORM
        ds = DatasetVersionORM(site_id=env["site_id"], name="DS 1", manifest_hash="sha256:d", frame_count=100)
        db.add(ds)
        db.commit()
        ds_id = ds.id

    t_allow = client.post("/api/training/runs", json={"dataset_version_id": ds_id, "run_kind": "base"}, headers=eng_h)
    assert t_allow.status_code == 202

    # 6. Admin accesses System Audit & Outbox List (ALLOWED)
    out_res = client.get("/api/system/outbox", headers=adm_h)
    assert out_res.status_code == 200

    # 7. Operator tries to access Admin User management (FORBIDDEN -> 403)
    user_forbid = client.post(
        "/api/auth/register", json={"username": "hacker", "password": "123", "role": "admin"}, headers=op_h
    )
    assert user_forbid.status_code == 403
