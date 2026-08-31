"""Unit tests for FastAPI endpoints, authentication, and RBAC (§8.1, §8.2)."""

import pytest
from fastapi.testclient import TestClient

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraORM,
    GateORM,
    LineORM,
    ModelVersionORM,
    ProductProfileORM,
    SiteORM,
)
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _seed_default_users():
    # login() no longer auto-seeds (removed with the rest of demo-data
    # behavior); this file logs in with the fixed operator/engineer/admin
    # test accounts, so seed them explicitly here instead.
    init_db_sync()
    with get_sync_session() as db:
        UserRepository(db).seed_default_users()


def setup_api_data():
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="API Test Site")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()

        prof = ProductProfileORM(site_id=site.id, name="Bag 50kg", nominal_dims_mm={})
        db.add(prof)
        db.commit()

        return site.id, line.id, prof.id


def test_auth_login_and_roles():
    setup_api_data()

    # 1. Login as operator
    res_op = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    assert res_op.status_code == 200
    op_token = res_op.json()["token"]

    # 2. Login as admin
    res_admin = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res_admin.status_code == 200
    admin_token = res_admin.json()["token"]

    # 3. Operator tries to access Admin route (e.g. POST /sites) -> 403 Forbidden
    headers_op = {"Authorization": f"Bearer {op_token}"}
    res_forbidden = client.post("/api/sites", json={"name": "Forbidden Site"}, headers=headers_op)
    assert res_forbidden.status_code == 403

    # 4. Admin accesses Admin route -> 200 OK
    headers_admin = {"Authorization": f"Bearer {admin_token}"}
    res_allowed = client.post("/api/sites", json={"name": "Allowed Site"}, headers=headers_admin)
    assert res_allowed.status_code == 200


def test_session_lifecycle_routes():
    _, line_id, prof_id = setup_api_data()
    res_login = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    token = res_login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Open session
    res_open = client.post(
        "/api/sessions",
        json={"line_id": line_id, "product_profile_id": prof_id, "external_ref": "IRS-2026-001", "target_count": 100},
        headers=headers,
    )
    assert res_open.status_code == 200
    sess_id = res_open.json()["id"]

    # 2. Get session detail
    res_get = client.get(f"/api/sessions/{sess_id}", headers=headers)
    assert res_get.status_code == 200
    assert res_get.json()["status"] == "open"

    # 3. Close session
    res_close = client.post(f"/api/sessions/{sess_id}/close", headers=headers)
    assert res_close.status_code == 200
    assert res_close.json()["status"] == "closed"

    # 4. Submit to ERP outbox
    res_submit = client.post(f"/api/sessions/{sess_id}/submit", headers=headers)
    assert res_submit.status_code == 200
    assert res_submit.json()["status"] == "submitted_to_outbox"


def test_dispute_defect_event_route():
    _, line_id, prof_id = setup_api_data()
    # simulate_bag_crossing now requires a real camera/gate/active deployment
    # bundle for the line (previously fell back to a hardcoded id=1 for
    # whichever was missing -- a real FK violation on any line whose real
    # rows didn't happen to have id 1, see stream_renderer.py's
    # reload_camera_context() for the same fix on the live-stream paths).
    with get_sync_session() as db:
        cam = CameraORM(line_id=line_id, node_id=1, source_driver="rtsp")
        db.add(cam)
        gate = GateORM(line_id=line_id, name="Gate 1")
        db.add(gate)
        mv = ModelVersionORM(onnx_hash="test-hash", onnx_path="models/test.onnx")
        db.add(mv)
        db.commit()
        config_repo = ConfigRepository(db)
        cfg = config_repo.create_config_version(line_id=line_id, payload={})
        config_repo.create_and_activate_bundle(line_id=line_id, model_version_id=mv.id, config_version_id=cfg.id)

    res_login = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    headers = {"Authorization": f"Bearer {res_login.json()['token']}"}

    sess_id = client.post(
        "/api/sessions",
        json={"line_id": line_id, "product_profile_id": prof_id, "target_count": 10},
        headers=headers,
    ).json()["id"]
    client.post(f"/api/sessions/{sess_id}/simulate_bag", json={"direction": 1, "defect_reason": "torn"}, headers=headers)

    events = client.get(f"/api/sessions/{sess_id}/events", headers=headers).json()
    defect_event = next(e for e in events if e["defect_reason"] == "torn")
    assert defect_event["defect_disputed"] is False

    res = client.post(
        f"/api/events/{defect_event['event_id']}/dispute_defect",
        json={"note": "Bag was intact"},
        headers=headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["defect_disputed"] is True
    assert body["defect_disputed_by"] == "operator"
    assert body["defect_disputed_note"] == "Bag was intact"
    assert body["defect_reason"] == "torn"  # original detection preserved


def test_dispute_nonexistent_event_returns_404():
    res_login = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    headers = {"Authorization": f"Bearer {res_login.json()['token']}"}
    res = client.post("/api/events/not-a-real-id/dispute_defect", json={}, headers=headers)
    assert res.status_code == 404


def test_simulate_bag_400_without_real_camera_gate_bundle():
    """No camera/gate/active bundle configured for the line -> a real,
    clear 400 -- previously fell back to a hardcoded camera_id=1/gate_id=1/
    deployment_bundle_id=1 for whichever was missing, which either silently
    recorded the crossing against the wrong real row or raised an
    unhandled foreign-key violation for any line whose real rows didn't
    happen to have id 1."""
    _, line_id, prof_id = setup_api_data()
    res_login = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    headers = {"Authorization": f"Bearer {res_login.json()['token']}"}

    sess_id = client.post(
        "/api/sessions",
        json={"line_id": line_id, "product_profile_id": prof_id, "target_count": 10},
        headers=headers,
    ).json()["id"]

    res = client.post(f"/api/sessions/{sess_id}/simulate_bag", json={"direction": 1}, headers=headers)
    assert res.status_code == 400
    assert "camera" in res.json()["detail"].lower() or "gate" in res.json()["detail"].lower() or "deployment" in res.json()["detail"].lower()


def test_cvat_status_endpoint():
    res = client.get("/api/cvat/status?cvat_url=http://127.0.0.1:9999/api")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "offline"
    assert "ui_url" in body


def test_cvat_sync_dataset_endpoint():
    res_login = client.post("/api/auth/login", json={"username": "engineer", "password": "eng123"})
    headers = {"Authorization": f"Bearer {res_login.json()['token']}"}

    res = client.post("/api/cvat/sync_dataset", json={}, headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert "status" in body
    assert "real_bags_path" in body
    assert "image_count" in body


def test_auth_via_query_param_token_for_sse():
    """Verify that get_current_user resolves tokens passed in query string (?token=...) for browser EventSource."""
    res_login = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    token = res_login.json()["token"]

    # Request without Authorization header, passing token query param only
    res = client.get(f"/api/sessions?token={token}")
    assert res.status_code == 200


