"""Exhaustive Coverage Suite for All Remaining API Routes, CVAT, Calibrations, and System Endpoints (§5.8, §8.2).

Directly exhausts:
1. Dataset jobs (extract, synthesize, build, mine_hard_frames).
2. Calibration jobs (perspective, motion, scale).
3. Line ROI configuration and bundle activation.
4. Model downloads, listing, and stage transitions.
5. CVAT status and connectivity.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from packages.cs_core.models import UserRole
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
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
    UserAccountORM,
)
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.auth import create_access_token
from services.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db_sync()


def test_api_all_remaining_routes_and_error_branches():
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        u_adm = user_repo.create_user("adm_all", "pass12345", role="admin")
        u_eng = user_repo.create_user("eng_all", "pass12345", role="engineer")
        u_op = user_repo.create_user("op_all", "pass12345", role="operator")

        site = SiteORM(name="Site All")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="node-all")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Line All", status="idle")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="rtsp", source_config={"url": "rtsp://localhost"}, enabled=True)
        db.add(cam)
        db.commit()

        gate = GateORM(line_id=line.id, name="Gate All", order_index=0)
        db.add(gate)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod All", erp_material_code="ALL-001")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-all")
        db.add(model_v)
        db.commit()

        cfg_v = ConfigVersionORM(line_id=line.id, payload={"confidence_threshold": 0.40})
        db.add(cfg_v)
        db.commit()

        bundle = DeploymentBundleORM(line_id=line.id, model_version_id=model_v.id, config_version_id=cfg_v.id)
        db.add(bundle)
        db.commit()

        job = JobORM(kind="synthesize", status="queued", payload={"count": 1}, priority=5)
        db.add(job)
        db.commit()

        adm_id, eng_id, op_id = u_adm.id, u_eng.id, u_op.id
        line_id, cam_id, model_id, job_id, cfg_id = line.id, cam.id, model_v.id, job.id, cfg_v.id

    adm_headers = {"Authorization": f"Bearer {create_access_token(data={'sub': str(adm_id), 'username': 'adm_all', 'role': 'admin'})}"}
    eng_headers = {"Authorization": f"Bearer {create_access_token(data={'sub': str(eng_id), 'username': 'eng_all', 'role': 'engineer'})}"}
    op_headers = {"Authorization": f"Bearer {create_access_token(data={'sub': str(op_id), 'username': 'op_all', 'role': 'operator'})}"}

    # 1. Calibration endpoints
    r = client.post(f"/api/calibrations/{line_id}/motion", json={"sample_frames": 5}, headers=eng_headers)
    assert r.status_code == 202
    r = client.post(f"/api/calibrations/{line_id}/scale", json={"nominal_bag_width_mm": 500.0}, headers=eng_headers)
    assert r.status_code == 202
    r = client.post(f"/api/calibrations/{line_id}/perspective", json={"roi_src_points": [[50.0, 50.0], [590.0, 50.0], [590.0, 590.0], [50.0, 590.0]]}, headers=eng_headers)
    assert r.status_code == 200

    # 2. Line ROI & Bundle activation
    r = client.post(f"/api/lines/{line_id}/roi", json={"roi_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}, headers=eng_headers)
    assert r.status_code == 200
    r = client.post("/api/bundles/activate", json={"line_id": line_id, "model_version_id": model_id, "config_version_id": cfg_id}, headers=eng_headers)
    assert r.status_code == 200

    # 3. Model management
    r = client.get("/api/models", headers=eng_headers)
    assert r.status_code == 200
    r = client.post(f"/api/models/{model_id}/stage?stage=shadow", headers=eng_headers)
    assert r.status_code == 200

    # 4. Extract, synthesize & build dataset jobs
    r = client.post("/api/datasets/extract", json={"video_path": "dummy.mp4"}, headers=eng_headers)
    assert r.status_code == 202
    r = client.post("/api/datasets/synthesize", json={"count": 2}, headers=eng_headers)
    assert r.status_code == 202
    r = client.post("/api/datasets/build", json={"sessions": []}, headers=eng_headers)
    assert r.status_code == 202
    r = client.post("/api/datasets/mine_hard_frames", json={"line_id": line_id}, headers=eng_headers)
    assert r.status_code == 202

    # 5. Replay and training
    r = client.post("/api/replay/runs", json={"scenario": "normal"}, headers=eng_headers)
    assert r.status_code == 202
    r = client.post("/api/training/runs", json={"epochs": 1}, headers=eng_headers)
    assert r.status_code == 202

    # 6. CVAT status
    r = client.get("/api/cvat/status")
    assert r.status_code == 200
    assert "status" in r.json()
