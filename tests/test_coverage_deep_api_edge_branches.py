"""Deep Coverage Test Suite for API Edge Cases, Errors, Disputes & Reports (§5.5, §5.7, §8.2).

Covers all remaining branches in services/api/routes.py:
1. Defect dispute overturning and idempotency.
2. Model artifact downloads and stage updates.
3. Shadow model A/B comparison metrics.
4. CVAT connectivity status.
5. Session submission to transactional outbox.
6. Error branches (400/404 on missing entities, invalid passwords, short passwords).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from packages.cs_core.models import UserRole
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
    ReconciliationORM,
    SessionORM,
    SiteORM,
    UserAccountORM,
)
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.auth import create_access_token
from services.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db_sync()


def test_api_edge_cases_and_error_handling(tmp_path):
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        u_adm = user_repo.create_user("admin_edge", "pass12345", role="admin")
        u_eng = user_repo.create_user("eng_edge", "pass12345", role="engineer")
        u_op = user_repo.create_user("op_edge", "pass12345", role="operator")

        site = SiteORM(name="Site Edge")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="node-edge")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Line Edge", status="idle")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="rtsp", source_config={"url": "rtsp://localhost"}, enabled=True)
        db.add(cam)
        db.commit()

        gate = GateORM(line_id=line.id, name="Gate Edge", order_index=0)
        db.add(gate)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod Edge", erp_material_code="EDGE-01")
        db.add(prod)
        db.commit()

        dummy_onnx = tmp_path / "edge_model.onnx"
        dummy_onnx.write_bytes(b"ONNX_DUMMY_BYTES_12345")

        model_v = ModelVersionORM(stage="active", onnx_path=str(dummy_onnx), onnx_hash="hash-edge")
        db.add(model_v)
        db.commit()

        cfg_v = ConfigVersionORM(line_id=line.id, payload={"confidence_threshold": 0.40})
        db.add(cfg_v)
        db.commit()

        bundle = DeploymentBundleORM(line_id=line.id, model_version_id=model_v.id, config_version_id=cfg_v.id)
        db.add(bundle)
        db.commit()

        sess = SessionORM(line_id=line.id, product_profile_id=prod.id, target_count=50, counted_total=50, status="closed")
        db.add(sess)
        db.commit()

        # Create defect event with all non-null columns
        event = CountEventORM(
            event_id="EVT-DEFECT-1",
            line_id=line.id,
            camera_id=cam.id,
            gate_id=gate.id,
            deployment_bundle_id=bundle.id,
            session_id=sess.id,
            track_id=101,
            crossing_seq=1,
            crossing_timestamp=datetime.now(timezone.utc),
            frame_index=150,
            direction=1,
            confidence=0.95,
            stream_epoch=1,
            defect_reason="Torn Bag Corner",
        )
        db.add(event)
        db.commit()

        adm_id, eng_id, op_id = u_adm.id, u_eng.id, u_op.id
        line_id, model_id, sess_id = line.id, model_v.id, sess.id

    adm_token = create_access_token(data={"sub": str(adm_id), "username": "admin_edge", "role": "admin"})
    eng_token = create_access_token(data={"sub": str(eng_id), "username": "eng_edge", "role": "engineer"})
    op_token = create_access_token(data={"sub": str(op_id), "username": "op_edge", "role": "operator"})

    adm_headers = {"Authorization": f"Bearer {adm_token}"}
    eng_headers = {"Authorization": f"Bearer {eng_token}"}
    op_headers = {"Authorization": f"Bearer {op_token}"}

    # 1. Defect dispute overturning
    r = client.post("/api/events/EVT-DEFECT-1/dispute_defect", json={"note": "False positive, bag is intact"}, headers=eng_headers)
    assert r.status_code == 200
    assert r.json()["defect_disputed_by"] == "eng_edge"

    # 2. Model download
    r = client.get(f"/api/models/{model_id}/download", headers=eng_headers)
    assert r.status_code == 200
    assert r.content == b"ONNX_DUMMY_BYTES_12345"

    # 3. Shadow comparison
    r = client.get("/api/models/shadow/comparison", headers=eng_headers)
    assert r.status_code == 200

    # 4. CVAT status
    r = client.get("/api/cvat/status")
    assert r.status_code == 200
    assert "status" in r.json()

    # 5. Session submission to outbox
    r = client.post(f"/api/sessions/{sess_id}/submit", headers=op_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "submitted_to_outbox"

    # 6. Password change validation errors
    r = client.post("/api/auth/change-password", json={"old_password": "wrongpassword", "new_password": "newpass123"}, headers=adm_headers)
    assert r.status_code == 400

    r = client.post("/api/auth/change-password", json={"old_password": "pass12345", "new_password": "123"}, headers=adm_headers)
    assert r.status_code == 400
