"""Unit tests for FastAPI endpoints, authentication, and RBAC (§8.1, §8.2)."""

from fastapi.testclient import TestClient
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import LineORM, ProductProfileORM, SiteORM
from services.api.main import app

client = TestClient(app)


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
