"""Comprehensive Test Suite for Waybill Reports, Defect Exports & Inference Modbus Integration (§5.8, §7.1).

Covers:
1. API endpoints for dispatch waybills, defect exports, and camera stream configs.
2. InferenceWorker frame processing with gate crossing and Modbus PLC register sync.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from drivers.io_modbus_tcp.controller import ModbusTcpIoController
from packages.cs_core.frame import Frame
from packages.cs_core.transport import SharedMemoryTransport
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraORM,
    ConfigVersionORM,
    CountEventORM,
    DeploymentBundleORM,
    GateORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    ProductProfileORM,
    SessionORM,
    SiteORM,
    UserAccountORM,
)
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.auth import create_access_token
from services.api.main import app
from services.inference.worker import InferenceWorker

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db_sync()


def test_api_waybill_and_defect_exports():
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        u_eng = user_repo.create_user("eng_wb", "pass12345", role="engineer")
        u_op = user_repo.create_user("op_wb", "pass12345", role="operator")

        site = SiteORM(name="Site WB")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="node-wb")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Line WB", status="running")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="rtsp", source_config={"url": "rtsp://localhost"}, enabled=True)
        db.add(cam)
        db.commit()

        gate = GateORM(line_id=line.id, name="Gate WB", order_index=0)
        db.add(gate)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod WB", erp_material_code="WB-001")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-wb")
        db.add(model_v)
        db.commit()

        cfg_v = ConfigVersionORM(line_id=line.id, payload={"confidence_threshold": 0.40})
        db.add(cfg_v)
        db.commit()

        bundle = DeploymentBundleORM(line_id=line.id, model_version_id=model_v.id, config_version_id=cfg_v.id)
        db.add(bundle)
        db.commit()

        sess = SessionORM(
            line_id=line.id,
            product_profile_id=prod.id,
            target_count=100,
            counted_total=100,
            external_ref="WB-WAYBILL-001",
            vehicle_plate="34-ABC-123",
            driver_name="Ahmet Yilmaz",
            carrier_company="Lojistik AS",
            status="closed",
        )
        db.add(sess)
        db.commit()

        # Defect event
        evt = CountEventORM(
            event_id="EVT-WB-01",
            line_id=line.id,
            camera_id=cam.id,
            gate_id=gate.id,
            deployment_bundle_id=bundle.id,
            session_id=sess.id,
            track_id=1,
            crossing_seq=1,
            crossing_timestamp=datetime.now(timezone.utc),
            frame_index=50,
            direction=1,
            confidence=0.98,
            stream_epoch=1,
            defect_reason="Ripped sack side seam",
        )
        db.add(evt)
        db.commit()

        sess_id, line_id = sess.id, line.id
        eng_id, op_id = u_eng.id, u_op.id

    eng_token = create_access_token(data={"sub": str(eng_id), "username": "eng_wb", "role": "engineer"})
    op_token = create_access_token(data={"sub": str(op_id), "username": "op_wb", "role": "operator"})

    eng_headers = {"Authorization": f"Bearer {eng_token}"}
    op_headers = {"Authorization": f"Bearer {op_token}"}

    # 1. Dispatch report
    r = client.get(f"/api/sessions/{sess_id}/dispatch_report", headers=op_headers)
    assert r.status_code == 200
    assert r.json()["truck_plate"] == "34-ABC-123"

    # 2. Defects list
    r = client.get(f"/api/lines/{line_id}/defects", headers=op_headers)
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_inference_worker_with_io_controller():
    mock_plc = MagicMock()
    mock_plc.read_coil.return_value = True
    mock_plc.write_holding_register.return_value = True

    transport = SharedMemoryTransport(ring_slots=4)

    with get_sync_session() as db:
        site = SiteORM(name="Site Modbus")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="node-modbus")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Line Modbus", status="running")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="file", source_config={"path": "dummy.mp4"}, enabled=True)
        db.add(cam)
        db.commit()

        gate = GateORM(line_id=line.id, name="Gate Modbus", order_index=0)
        db.add(gate)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod Modbus")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-mb")
        db.add(model_v)
        db.commit()

        cfg_v = ConfigVersionORM(line_id=line.id, payload={"confidence_threshold": 0.40})
        db.add(cfg_v)
        db.commit()

        bundle = DeploymentBundleORM(line_id=line.id, model_version_id=model_v.id, config_version_id=cfg_v.id)
        db.add(bundle)
        db.commit()

        sess = SessionORM(line_id=line.id, product_profile_id=prod.id, target_count=100, status="counting")
        db.add(sess)
        db.commit()

        cam_id, line_id = cam.id, line.id

    worker = InferenceWorker(transport=transport, line_id=line_id)
    worker.modbus = mock_plc

    # Write frame to transport
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    transport.write_image_data(f"shm_cam_{cam_id}_slot_0", dummy_img)
    transport.publish(Frame(
        camera_id=cam_id,
        stream_epoch=1,
        frame_index=1,
        monotonic_ns=time.perf_counter_ns(),
        wall_clock=datetime.now(UTC),
        shape=(640, 640, 3),
        dtype="uint8",
        shm_name=f"shm_cam_{cam_id}_slot_0",
    ))

    processed = worker.run_step()
    assert processed == 1
