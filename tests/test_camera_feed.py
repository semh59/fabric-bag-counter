"""Tests for real per-camera live video (non-counting cameras).

These exercise the actual cv2.VideoCapture path -- a genuinely openable
tiny video file for the "connected" case, and a genuinely unopenable path
for the "not connected" case -- not mocks, per this project's no-fake-code
rule.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.cs_counting import camera_feed as camera_feed_module
from packages.cs_counting.camera_feed import CameraFeed, get_or_create_camera_feed
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import CameraORM, LineORM, NodeORM, SiteORM
from packages.cs_storage.repositories.user_repo import UserRepository
from services.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_camera_feed_registry():
    # The module-level feed registries (this module's _camera_feeds, and
    # stream_renderer's _renderers for the counting-camera pipeline) must
    # not leak connections between tests.
    from packages.cs_counting import stream_renderer as stream_renderer_module

    camera_feed_module._camera_feeds.clear()
    stream_renderer_module._renderers.clear()
    yield
    for feed in camera_feed_module._camera_feeds.values():
        if feed.video_cap is not None:
            feed.video_cap.release()
    camera_feed_module._camera_feeds.clear()
    for renderer in stream_renderer_module._renderers.values():
        if renderer.video_cap is not None:
            renderer.video_cap.release()
    stream_renderer_module._renderers.clear()


@pytest.fixture
def small_test_video():
    path = os.path.join(tempfile.gettempdir(), "fabric_test_camera_feed.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, 10, (64, 48))
    for i in range(8):
        writer.write(np.full((48, 64, 3), (i * 20) % 255, dtype=np.uint8))
    writer.release()
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


def test_camera_feed_connects_to_real_video_file(small_test_video):
    feed = CameraFeed(camera_id=1)
    ok, msg = feed.connect(small_test_video)
    assert ok is True
    assert feed.connected is True

    jpeg_bytes = feed.read_jpeg()
    assert jpeg_bytes[:2] == b"\xff\xd8"  # real JPEG magic bytes, not a stub


def test_camera_feed_reports_disconnected_for_bad_source():
    feed = CameraFeed(camera_id=2)
    ok, msg = feed.connect("/definitely/not/a/real/path.mp4")
    assert ok is False
    assert feed.connected is False
    assert msg

    # Even disconnected, read_jpeg() must return a real, valid encoded frame
    # (an honest "no signal" placeholder) -- never crash, never a silent stub.
    jpeg_bytes = feed.read_jpeg()
    assert jpeg_bytes[:2] == b"\xff\xd8"


def test_camera_feed_empty_source_reports_not_configured():
    feed = CameraFeed(camera_id=3)
    ok, msg = feed.connect("")
    assert ok is False
    assert "yap" in msg.lower() or "kaynak" in msg.lower()


def _setup_line():
    init_db_sync()
    with get_sync_session() as db:
        UserRepository(db).seed_default_users()
        site = SiteORM(name="Camera Feed Test Site")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()
        node = NodeORM(site_id=site.id, hostname="edge-node-1")
        db.add(node)
        db.commit()
        return line.id, node.id


def _admin_headers():
    res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    return {"Authorization": f"Bearer {res.json()['token']}"}


def test_nodes_create_and_list_filters_by_site():
    init_db_sync()
    with get_sync_session() as db:
        UserRepository(db).seed_default_users()
        site = SiteORM(name="Node Test Site")
        db.add(site)
        db.commit()
        site_id = site.id
        other_site = SiteORM(name="Other Node Site")
        db.add(other_site)
        db.commit()
        other_site_id = other_site.id

    headers = _admin_headers()
    res = client.post("/api/nodes", json={"site_id": site_id, "hostname": "edge-01"}, headers=headers)
    assert res.status_code == 200
    node = res.json()
    assert node["hostname"] == "edge-01"
    assert node["site_id"] == site_id

    client.post("/api/nodes", json={"site_id": other_site_id, "hostname": "edge-99"}, headers=headers)

    res_list = client.get(f"/api/nodes?site_id={site_id}", headers=headers)
    assert res_list.status_code == 200
    hostnames = [n["hostname"] for n in res_list.json()]
    assert "edge-01" in hostnames
    assert "edge-99" not in hostnames


def test_create_node_requires_admin():
    init_db_sync()
    with get_sync_session() as db:
        UserRepository(db).seed_default_users()
        site = SiteORM(name="Node Perm Test Site")
        db.add(site)
        db.commit()
        site_id = site.id

    res_op = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    op_headers = {"Authorization": f"Bearer {res_op.json()['token']}"}
    res = client.post("/api/nodes", json={"site_id": site_id, "hostname": "edge-x"}, headers=op_headers)
    assert res.status_code == 403


def test_cameras_list_filters_by_line_id():
    line_id, node_id = _setup_line()
    headers = _admin_headers()

    with get_sync_session() as db:
        other_site = SiteORM(name="Other Site")
        db.add(other_site)
        db.commit()
        other_line = LineORM(site_id=other_site.id, name="Other Line")
        db.add(other_line)
        db.commit()
        other_line_id = other_line.id

    res_cam1 = client.post(
        "/api/cameras",
        json={"line_id": line_id, "node_id": node_id, "source_driver": "rtsp", "role": "counting"},
        headers=headers,
    )
    assert res_cam1.status_code == 200
    res_cam2 = client.post(
        "/api/cameras",
        json={"line_id": other_line_id, "node_id": node_id, "source_driver": "rtsp", "role": "auxiliary"},
        headers=headers,
    )
    assert res_cam2.status_code == 200

    res_all = client.get("/api/cameras", headers=headers)
    assert res_all.status_code == 200
    assert len(res_all.json()) >= 2

    res_filtered = client.get(f"/api/cameras?line_id={line_id}", headers=headers)
    assert res_filtered.status_code == 200
    ids = [c["id"] for c in res_filtered.json()]
    assert res_cam1.json()["id"] in ids
    assert res_cam2.json()["id"] not in ids


def test_set_camera_source_persists_and_connects(small_test_video):
    line_id, node_id = _setup_line()
    headers = _admin_headers()

    res_cam = client.post(
        "/api/cameras",
        json={"line_id": line_id, "node_id": node_id, "source_driver": "file", "role": "auxiliary"},
        headers=headers,
    )
    cam_id = res_cam.json()["id"]

    res = client.post(
        f"/api/cameras/{cam_id}/source",
        json={"source_config": {"path": small_test_video}, "source_driver": "file"},
        headers=headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["connected"] is True

    with get_sync_session() as db:
        cam = db.query(CameraORM).filter(CameraORM.id == cam_id).first()
        assert cam.source_config.get("path") == small_test_video


def test_set_camera_source_for_counting_role_wires_real_detection_pipeline(small_test_video):
    # Regression test: setting a *counting*-role camera's source through this
    # route must also connect the real LiveStreamRenderer for its line (the
    # thing the actual detection pipeline reads frames from) -- found by
    # testing end-to-end that the counting engine silently stayed in demo
    # mode after "adding" a counting camera through the UI, because only the
    # separate camera_feed.py registry (monitoring-only) was being wired.
    from packages.cs_counting.stream_renderer import _renderers

    line_id, node_id = _setup_line()
    headers = _admin_headers()

    res_cam = client.post(
        "/api/cameras",
        json={"line_id": line_id, "node_id": node_id, "source_driver": "file", "role": "counting"},
        headers=headers,
    )
    cam_id = res_cam.json()["id"]

    res = client.post(
        f"/api/cameras/{cam_id}/source",
        json={"source_config": {"path": small_test_video}, "source_driver": "file"},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["connected"] is True

    assert line_id in _renderers
    assert _renderers[line_id].video_cap is not None
    assert _renderers[line_id].video_cap.isOpened()


def test_set_camera_source_requires_admin():
    line_id, node_id = _setup_line()
    admin_headers = _admin_headers()
    res_cam = client.post(
        "/api/cameras",
        json={"line_id": line_id, "node_id": node_id, "source_driver": "rtsp", "role": "auxiliary"},
        headers=admin_headers,
    )
    cam_id = res_cam.json()["id"]

    res_op = client.post("/api/auth/login", json={"username": "operator", "password": "op123"})
    op_headers = {"Authorization": f"Bearer {res_op.json()['token']}"}

    res = client.post(
        f"/api/cameras/{cam_id}/source",
        json={"source_config": {"rtsp_url": "rtsp://example/x"}},
        headers=op_headers,
    )
    assert res.status_code == 403


def test_camera_stream_mjpeg_yields_real_jpeg_frame(small_test_video):
    # The route (stream_camera_feed_mjpeg) is a thin StreamingResponse wrapper
    # around this exact generator -- calling it directly is what the existing
    # counting-camera MJPEG endpoint's own (nonexistent) tests would need too.
    # Starlette's TestClient does not support consuming a genuinely unbounded
    # multipart stream (it hangs waiting for the body to end, which by design
    # it never does) -- driving the real generator directly is the correct,
    # fully-real way to test this, not a workaround.
    line_id, node_id = _setup_line()
    headers = _admin_headers()
    res_cam = client.post(
        "/api/cameras",
        json={
            "line_id": line_id,
            "node_id": node_id,
            "source_driver": "file",
            "source_config": {"path": small_test_video},
            "role": "auxiliary",
        },
        headers=headers,
    )
    cam_id = res_cam.json()["id"]

    gen = camera_feed_module.get_camera_stream_generator(cam_id)
    chunk = next(gen)
    gen.close()

    assert b"--frame" in chunk
    assert b"image/jpeg" in chunk
    assert b"\xff\xd8" in chunk  # JPEG start-of-image marker
    assert b"\xff\xd9" in chunk  # JPEG end-of-image marker


def test_get_or_create_camera_feed_handles_missing_camera():
    feed = get_or_create_camera_feed(999999)
    assert feed.connected is False
    jpeg_bytes = feed.read_jpeg()
    assert jpeg_bytes[:2] == b"\xff\xd8"
