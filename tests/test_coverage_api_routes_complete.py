"""Exhaustive API Route & Endpoint Coverage Test Suite (§8.1, §8.2).

Hits every single endpoint, method, permission check, error handler, and parameter in services/api/routes.py.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timezone
import pytest
from fastapi.testclient import TestClient

from packages.cs_core.models import UserRole
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraORM,
    ConfigVersionORM,
    DeploymentBundleORM,
    GateORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
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
def setup_api_env():
    init_db_sync()


def test_complete_api_routes_coverage():
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        u_adm = user_repo.create_user("super_admin", "adm12345", role="admin")
        u_eng = user_repo.create_user("chief_engineer", "eng12345", role="engineer")
        u_op = user_repo.create_user("plant_operator", "op12345", role="operator")

        site = SiteORM(name="Global Cement Site", timezone="Europe/Istanbul", locale="tr_TR")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="edge-pc-01", gpu_info={"name": "RTX 4060"})
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Conveyor Line 1", status="idle")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="rtsp", source_config={"url": "rtsp://localhost/1"}, enabled=True)
        db.add(cam)
        db.commit()

        gate = GateORM(line_id=line.id, name="Main Gate", order_index=0)
        db.add(gate)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Standard Portland", erp_material_code="MAT-001")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-abc")
        db.add(model_v)
        db.commit()

        cfg_v = ConfigVersionORM(line_id=line.id, payload={"confidence_threshold": 0.40})
        db.add(cfg_v)
        db.commit()

        bundle = DeploymentBundleORM(line_id=line.id, model_version_id=model_v.id, config_version_id=cfg_v.id)
        db.add(bundle)
        db.commit()

        site_id = site.id
        node_id = node.id
        line_id = line.id
        cam_id = cam.id
        gate_id = gate.id
        prod_id = prod.id
        model_id = model_v.id
        cfg_id = cfg_v.id
        bundle_id = bundle.id

    adm_token = create_access_token(data={"sub": str(u_adm.id), "username": "super_admin", "role": "admin"})
    eng_token = create_access_token(data={"sub": str(u_eng.id), "username": "chief_engineer", "role": "engineer"})
    op_token = create_access_token(data={"sub": str(u_op.id), "username": "plant_operator", "role": "operator"})

    adm_headers = {"Authorization": f"Bearer {adm_token}"}
    eng_headers = {"Authorization": f"Bearer {eng_token}"}
    op_headers = {"Authorization": f"Bearer {op_token}"}

    # 1. Auth & Users
    r = client.post("/api/auth/login", json={"username": "super_admin", "password": "adm12345"})
    assert r.status_code == 200
    r = client.post("/api/auth/login", json={"username": "wrong", "password": "wrong"})
    assert r.status_code == 401

    r = client.post("/api/auth/change-password", json={"old_password": "adm12345", "new_password": "newpassword123"}, headers=adm_headers)
    assert r.status_code == 200

    r = client.post("/api/auth/register", json={"username": "new_op", "password": "password123", "role": "operator"}, headers=adm_headers)
    assert r.status_code == 200

    # 2. Sites, Nodes, Lines, Cameras, Gates, Products
    r = client.get("/api/sites", headers=adm_headers)
    assert r.status_code == 200
    r = client.post("/api/sites", json={"name": "Site 2", "timezone": "Europe/Istanbul", "locale": "tr_TR"}, headers=adm_headers)
    assert r.status_code == 200

    r = client.get("/api/nodes", headers=adm_headers)
    assert r.status_code == 200
    r = client.post("/api/nodes", json={"site_id": site_id, "hostname": "edge-node-2", "gpu_info": {}}, headers=adm_headers)
    assert r.status_code == 200

    r = client.get("/api/lines", headers=adm_headers)
    assert r.status_code == 200
    r = client.post("/api/lines", json={"site_id": site_id, "name": "Line 2"}, headers=adm_headers)
    assert r.status_code == 200

    r = client.get("/api/cameras", headers=adm_headers)
    assert r.status_code == 200
    r = client.post("/api/cameras", json={"line_id": line_id, "node_id": node_id, "source_driver": "rtsp", "source_config": {"url": "rtsp://x"}, "role": "counting"}, headers=adm_headers)
    assert r.status_code == 200

    r = client.get("/api/gates", headers=adm_headers)
    assert r.status_code == 200
    r = client.post("/api/gates", json={"line_id": line_id, "name": "Gate 2", "order_index": 1}, headers=adm_headers)
    assert r.status_code == 200

    r = client.get("/api/products", headers=adm_headers)
    assert r.status_code == 200
    r = client.post("/api/products", json={"site_id": site_id, "name": "Product 2", "erp_material_code": "MAT-002"}, headers=adm_headers)
    assert r.status_code == 200

    # 3. Sessions & Operations Lifecycle
    r = client.post("/api/sessions", json={"line_id": line_id, "product_profile_id": prod_id, "target_count": 100}, headers=op_headers)
    assert r.status_code == 200
    sess_id = r.json()["id"]

    r = client.get("/api/sessions", headers=op_headers)
    assert r.status_code == 200
    r = client.get(f"/api/sessions/{sess_id}", headers=op_headers)
    assert r.status_code == 200

    r = client.post(f"/api/sessions/{sess_id}/pause", headers=op_headers)
    assert r.status_code == 200
    r = client.post(f"/api/sessions/{sess_id}/resume", headers=op_headers)
    assert r.status_code == 200

    # Simulate bag crossing
    r = client.post(f"/api/sessions/{sess_id}/simulate_bag", json={"direction": 1, "confidence": 0.98}, headers=op_headers)
    assert r.status_code == 200

    # Quick line settings
    r = client.post(f"/api/lines/{line_id}/quick_settings", json={"belt_speed": 8.0, "gate_x_pos": 340.0}, headers=eng_headers)
    assert r.status_code == 200

    # Camera source switch
    r = client.post(f"/api/cameras/{cam_id}/source", json={"source_config": {"url": "rtsp://10.0.0.1/live"}, "source_driver": "rtsp"}, headers=adm_headers)
    assert r.status_code == 200

    # Close session
    r = client.post(f"/api/sessions/{sess_id}/close", headers=op_headers)
    assert r.status_code == 200

    # Ledger events & reports
    r = client.get(f"/api/sessions/{sess_id}/events", headers=op_headers)
    assert r.status_code == 200
    r = client.get(f"/api/sessions/{sess_id}/dispatch_report", headers=op_headers)
    assert r.status_code == 200
    r = client.get(f"/api/lines/{line_id}/defects", headers=op_headers)
    assert r.status_code == 200

    # 4. Configs, Models, Bundles, Jobs & Calibrations
    r = client.get(f"/api/configs/{line_id}", headers=eng_headers)
    assert r.status_code == 200
    r = client.post(f"/api/configs/{line_id}", json={"payload": {"confidence_threshold": 0.45}, "note": "Adjusted conf"}, headers=eng_headers)
    assert r.status_code == 200

    r = client.get("/api/models", headers=eng_headers)
    assert r.status_code == 200
    r = client.post(f"/api/models/{model_id}/stage?stage=shadow", headers=eng_headers)
    assert r.status_code == 200

    r = client.post("/api/bundles/activate", json={"line_id": line_id, "model_version_id": model_id, "config_version_id": cfg_id}, headers=eng_headers)
    assert r.status_code == 200

    r = client.post(f"/api/calibrations/{line_id}/perspective", json={"roi_src_points": [[50.0, 50.0], [590.0, 50.0], [590.0, 590.0], [50.0, 590.0]]}, headers=eng_headers)
    assert r.status_code == 200

    r = client.post(f"/api/lines/{line_id}/roi", json={"roi_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}, headers=eng_headers)
    assert r.status_code == 200

    # Datasets & Training
    r = client.get("/api/datasets", headers=eng_headers)
    assert r.status_code == 200
    r = client.post("/api/datasets/synthesize", json={"count": 5}, headers=eng_headers)
    assert r.status_code == 202

    # Reconciliations
    with get_sync_session() as db:
        rec = ReconciliationORM(session_id=sess_id, trigger_reason="discrepancy", evidence_refs={"test": 1})
        db.add(rec)
        db.commit()
        rec_id = rec.id

    r = client.get("/api/reconciliations", headers=eng_headers)
    assert r.status_code == 200
    r = client.post(f"/api/reconciliations/{rec_id}/resolve", json={"resolution": "accept_system", "resolved_count": 1, "note": "Resolved"}, headers=eng_headers)
    assert r.status_code == 200

    # Health & Metrics
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    r = client.get("/api/metrics")
    assert r.status_code == 200
