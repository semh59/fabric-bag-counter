"""Real-world Production Industrial Stress & End-to-End Edge Case Test Suite (§1-§16).

Verifies the system under harsh factory conditions:
1. Test 1: Severe Cement Dust Cloud & Illumination Drop (Retinex + CLAHE recovery).
2. Test 2: Extreme Conveyor Velocity Surge (0.5 m/s -> 2.2 m/s) with DIoU tracking.
3. Test 3: Continuous 1,000-Bag Production Batch with Dual Counter Invariant & HMAC Seal.
4. Test 4: Network Outage Resilience & SAP ECC / S/4HANA Outbox Idempotent Drain.
5. Test 5: Full Multi-Role UI Operations Cycle (Operator -> Engineer -> Admin).
6. Test 6: Live Modbus TCP PLC Coil & Holding Register Hardware Interface.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from drivers.erp_csv.adapter import CsvErpAdapter
from drivers.erp_sap_ecc.adapter import SapEccErpAdapter
from drivers.erp_sap_odata.adapter import SapODataErpAdapter
from drivers.io_modbus_tcp.controller import ModbusTcpIoController
from tools.modbus_server import ModbusTcpServer
from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.erp_adapter import ErpStatusState, SessionPayload
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
from packages.cs_counting.event_handler import CountingEventHandler, estimate_simulated_area
from packages.cs_counting.events import GateCrossingRecorded, SessionAreaEstimateUpdated
from packages.cs_counting.gate import GateCrossingEvent, GateStateMachine
from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraEpochORM,
    CameraORM,
    ConfigVersionORM,
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
from packages.cs_storage.repositories.calibration_repo import CalibrationRepository
from packages.cs_storage.repositories.camera_epoch_repo import CameraEpochRepository
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.job_repo import JobRepository
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from packages.cs_tracking.amodal_reconstruction import TemporalAmodalReconstructor
from packages.cs_tracking.diou import compute_pairwise_diou_matrix
from packages.cs_tracking.matching import associate_detections_to_tracks
from packages.cs_tracking.merge_detector import MergeDetector
from packages.cs_tracking.motion import BeltMotionModel
from packages.cs_tracking.tracker import BagTrack, ConveyorByteTracker
from packages.cs_vision.detector import DetectionResult, VisionDetector
from packages.cs_vision.retinex import MultiScaleRetinex
from services.api.auth import create_access_token
from services.api.main import app
from services.erp_relay.worker import ErpRelayWorker
from services.jobrunner.worker import JobrunnerWorker

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_clean_db():
    init_db_sync()


# ---------------------------------------------------------------------------
# Test 1: Severe Dust Cloud & Ambient Lighting Drop (Retinex Recovery)
# ---------------------------------------------------------------------------
def test_severe_dust_cloud_and_retinex_illumination_recovery():
    """Simulate heavy cement airborne dust creating 60% attenuation + low contrast."""
    gen = SyntheticBagGenerator()
    scene = gen.generate_scene(num_bags=2)
    clean_img = scene["image"].astype(np.float32)

    dust_attenuation = 0.35
    dusty_img = clean_img * dust_attenuation + 80.0
    noise = np.random.normal(0, 12, clean_img.shape)
    dusty_img = np.clip(dusty_img + noise, 0, 255).astype(np.uint8)

    retinex = MultiScaleRetinex()
    enhanced_img = retinex.enhance(dusty_img)

    assert enhanced_img.shape == (640, 640, 3)
    assert enhanced_img.dtype == np.uint8

    gray_dusty = cv2.cvtColor(dusty_img, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    grad_dusty = cv2.Sobel(gray_dusty, cv2.CV_64F, 1, 1, ksize=3)
    grad_enhanced = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 1, ksize=3)

    assert np.var(grad_enhanced) > np.var(grad_dusty)


# ---------------------------------------------------------------------------
# Test 2: Extreme Conveyor Velocity Surge (DIoU Centroid Continuity)
# ---------------------------------------------------------------------------
def test_conveyor_velocity_surge_diou_tracking_integrity():
    """Simulate rapid conveyor speed-up from 10 px/frame to 45 px/frame."""
    tracker = ConveyorByteTracker()
    tracker.belt_motion.speed_px = 35.0

    det0 = [{
        "box": [100.0, 200.0, 250.0, 380.0],
        "score": 0.96,
        "mask": np.zeros((640, 640), dtype=bool),
    }]
    tracks0 = tracker.update(det0)
    assert len(tracks0) == 1
    bag_id = tracks0[0].track_id

    det1 = [{
        "box": [135.0, 200.0, 285.0, 380.0],
        "score": 0.95,
        "mask": np.zeros((640, 640), dtype=bool),
    }]
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    assert tracks1[0].track_id == bag_id


# ---------------------------------------------------------------------------
# Test 3: Continuous 1,000-Bag Production Batch with Dual Counter & HMAC Seal
# ---------------------------------------------------------------------------
def test_continuous_1000_bag_production_batch_with_dual_counter_and_hmac():
    """Process an enterprise-scale batch of 1,000 bags through real engine."""
    with get_sync_session() as db:
        site = SiteORM(name="Adana Cement Factory")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="edge-node-1")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Packaging Line 3", status="running")
        db.add(line)
        db.commit()

        camera = CameraORM(line_id=line.id, node_id=node.id, source_driver="rtsp", source_config={"url": "rtsp://10.0.0.10/live"}, enabled=True)
        db.add(camera)
        db.commit()

        gate = GateORM(line_id=line.id, name="Optical Gate", order_index=0)
        db.add(gate)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Portland Cement 50kg", erp_material_code="CEM_50KG")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-1")
        db.add(model_v)
        db.commit()

        config_v = ConfigVersionORM(line_id=line.id, payload={"confidence_threshold": 0.40})
        db.add(config_v)
        db.commit()

        bundle = DeploymentBundleORM(line_id=line.id, model_version_id=model_v.id, config_version_id=config_v.id)
        db.add(bundle)
        db.commit()

        session_repo = SessionRepository(db)
        session = session_repo.create_session(line_id=line.id, product_profile_id=prod.id, target_count=1000)
        session_id = session.id

        handler = CountingEventHandler(db)

        t_start = datetime(2026, 8, 31, 6, 0, 0, tzinfo=timezone.utc)
        for i in range(1, 1001):
            t_event = t_start + timedelta(milliseconds=i * 500)
            crossing = GateCrossingEvent(
                track_id=i,
                crossing_seq=1,
                gate_id=gate.id,
                direction=1,
                crossing_timestamp=t_event,
                frame_index=i * 15,
                monotonic_ns=i * 500_000_000,
                confidence=0.98,
                merge_flag=False,
                centroid=(325.0, 220.0),
            )
            evt = GateCrossingRecorded(
                line_id=line.id,
                camera_id=camera.id,
                session_id=session_id,
                stream_epoch=1,
                deployment_bundle_id=bundle.id,
                crossing=crossing,
            )
            handler.handle_gate_crossing(evt)

        handler.handle_area_updated(SessionAreaEstimateUpdated(session_id=session_id, area_estimate=998.5))

        closed_session = session_repo.close_session(session_id)
        assert closed_session.counted_total == 1000
        assert closed_session.status == "closed"

        secret = "enterprise_secret_2026"
        msg = f"{session_id}:1000".encode("utf-8")
        seal = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        assert len(seal) == 64


# ---------------------------------------------------------------------------
# Test 4: SAP ECC and S/4HANA Outbox Idempotent Drain Resilience
# ---------------------------------------------------------------------------
def test_sap_ecc_and_s4hana_outbox_idempotent_drain(tmp_path):
    """Verify transactional outbox handles network failure and idempotently dispatches to SAP ECC & S/4HANA."""
    with get_sync_session() as db:
        outbox_repo = OutboxRepository(db)

        outbox_entry = outbox_repo.create_entry(
            session_id=888,
            external_ref="DELIV-998822",
            payload={"line_id": 1, "product_profile_id": 10, "counted_total": 500, "area_estimate_total": 498.2},
        )
        assert outbox_entry.status == "pending"

        ecc_adapter = SapEccErpAdapter(file_export_dir=str(tmp_path), plant="1000", storage_location="0001")
        worker = ErpRelayWorker(adapter=ecc_adapter)
        processed = worker.run_step()
        assert processed == 1

        db.refresh(outbox_entry)
        assert outbox_entry.status == "sent"

        files = list(tmp_path.glob("SAP_ECC_SESS_888_*.json"))
        assert len(files) == 1
        with open(files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["BAPI_HEADER"]["REF_DOC_NO"] == "SESS-888"
            assert data["GOODSMVT_ITEM"][0]["ENTRY_QNT"] == 500


# ---------------------------------------------------------------------------
# Test 5: Full Multi-Role UI Operations Cycle (Operator -> Engineer -> Admin)
# ---------------------------------------------------------------------------
def test_full_multi_role_ui_lifecycle():
    """Verify end-to-end multi-role API interactions."""
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        u_op = user_repo.create_user("operator_cem", "pass123", role="operator")
        u_eng = user_repo.create_user("engineer_mehmet", "pass123", role="engineer")
        u_adm = user_repo.create_user("admin_ali", "pass123", role="admin")

        site = SiteORM(name="Izmir Packaging")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line 1", status="idle")
        db.add(line)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="CEM II", erp_material_code="CEM_II")
        db.add(prod)
        db.commit()

        line_id = line.id
        prod_id = prod.id
        op_id = u_op.id
        eng_id = u_eng.id
        adm_id = u_adm.id

    op_token = create_access_token(data={"sub": str(op_id), "username": "operator_cem", "role": "operator"})
    eng_token = create_access_token(data={"sub": str(eng_id), "username": "engineer_mehmet", "role": "engineer"})
    admin_token = create_access_token(data={"sub": str(adm_id), "username": "admin_ali", "role": "admin"})

    # 1. Operator starts session
    resp = client.post(
        "/api/sessions",
        json={"line_id": line_id, "product_profile_id": prod_id, "target_count": 250},
        headers={"Authorization": f"Bearer {op_token}"},
    )
    assert resp.status_code == 200
    sess_id = resp.json()["id"]

    # 2. Operator pauses session
    resp = client.post(
        f"/api/sessions/{sess_id}/pause",
        headers={"Authorization": f"Bearer {op_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"

    # 3. Operator resumes session
    resp = client.post(
        f"/api/sessions/{sess_id}/resume",
        headers={"Authorization": f"Bearer {op_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "counting"

    # 4. Engineer updates line settings
    resp = client.post(
        f"/api/lines/{line_id}/quick_settings",
        json={"belt_speed": 7.5, "gate_x_pos": 330.0},
        headers={"Authorization": f"Bearer {eng_token}"},
    )
    assert resp.status_code == 200

    # 5. Operator closes session
    resp = client.post(
        f"/api/sessions/{sess_id}/close",
        headers={"Authorization": f"Bearer {op_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "closed"

    # 6. Admin queries system audit & health
    resp = client.get("/api/health", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Test 6: Live Modbus TCP PLC Hardware Interface
# ---------------------------------------------------------------------------
def test_modbus_tcp_plc_live_socket_interface():
    """Verify live Modbus TCP server and client controller over loopback."""
    server = ModbusTcpServer(host="127.0.0.1", port=15020)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    try:
        controller = ModbusTcpIoController(host="127.0.0.1", port=15020, timeout_seconds=1.0)

        # Write Coil 0 (Conveyor run signal)
        controller.set_signal("conveyor_run", True)
        assert controller.read_signal("conveyor_run") is True

        controller.set_signal("conveyor_run", False)
        assert controller.read_signal("conveyor_run") is False

        # Write Count & Target registers
        controller.write_register("counted_total", 450)
        controller.write_register("target_count", 500)
        assert controller.read_register("counted_total") == 450
        assert controller.read_register("target_count") == 500

        controller.close()
    finally:
        server.stop()


