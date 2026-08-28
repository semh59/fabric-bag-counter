"""Tests for POST /lines/{line_id}/roi (§5.4, §6.1).

Before this endpoint existed, the frontend's ROI drawing tool
(toggleRoiDraw()) called no API at all and showed a fake "kaydedildi"
success toast regardless of what was drawn -- this is the real endpoint
that closes that gap, and CountingEngine.configure()/process_frame()
(packages/cs_counting/engine.py) is the real consumer on the other end.
"""

import pytest
from fastapi.testclient import TestClient

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import LineORM, ModelVersionORM, SiteORM
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _seed_default_users():
    init_db_sync()
    with get_sync_session() as db:
        UserRepository(db).seed_default_users()


def _setup_line() -> tuple[int, int]:
    with get_sync_session() as db:
        site = SiteORM(name="ROI Endpoint Test Site")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()
        mv = ModelVersionORM(onnx_hash="test-hash", onnx_path="models/test.onnx")
        db.add(mv)
        db.commit()
        return line.id, mv.id


def _login() -> str:
    res = client.post("/api/auth/login", json={"username": "engineer", "password": "eng123"})
    assert res.status_code == 200
    return res.json()["token"]


def test_roi_endpoint_400_when_no_active_bundle():
    line_id, _ = _setup_line()
    token = _login()
    res = client.post(
        f"/api/lines/{line_id}/roi",
        json={"roi_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400


def test_roi_endpoint_rejects_fewer_than_3_points():
    line_id, _ = _setup_line()
    token = _login()
    res = client.post(
        f"/api/lines/{line_id}/roi",
        json={"roi_polygon": [[0.1, 0.1], [0.9, 0.9]]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400


def test_roi_endpoint_creates_and_activates_bundle_preserving_other_fields():
    line_id, model_id = _setup_line()
    token = _login()
    headers = {"Authorization": f"Bearer {token}"}

    with get_sync_session() as db:
        repo = ConfigRepository(db)
        cfg = repo.create_config_version(line_id=line_id, payload={"confidence_threshold": 0.77})
        bundle = repo.create_and_activate_bundle(line_id=line_id, model_version_id=model_id, config_version_id=cfg.id)
        original_bundle_id = bundle.id

    polygon = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    res = client.post(f"/api/lines/{line_id}/roi", json={"roi_polygon": polygon}, headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["roi_polygon"] == polygon
    assert body["bundle_id"] != original_bundle_id

    with get_sync_session() as db:
        repo = ConfigRepository(db)
        active = repo.get_active_bundle(line_id)
        assert active.id == body["bundle_id"]
        assert active.model_version_id == model_id  # carried forward, not clobbered
        payload = repo.get_effective_config_payload(active.config_version)
        assert payload["roi_polygon"] == polygon
        assert payload["confidence_threshold"] == 0.77  # preserved from the previous config


def test_roi_endpoint_forbidden_for_operator():
    line_id, model_id = _setup_line()
    with get_sync_session() as db:
        repo = ConfigRepository(db)
        cfg = repo.create_config_version(line_id=line_id, payload={})
        repo.create_and_activate_bundle(line_id=line_id, model_version_id=model_id, config_version_id=cfg.id)

    res_op = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    token = res_op.json()["token"]
    res = client.post(
        f"/api/lines/{line_id}/roi",
        json={"roi_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403
