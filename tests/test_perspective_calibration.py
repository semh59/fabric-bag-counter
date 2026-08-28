"""Tests for Stage 3 (perspective/ROI-warp) camera calibration (§5.3, §6.2).

Covers: the real cv2 homography math (packages/cs_vision/calibration.py),
the repository round-trip, the API endpoint, and that
LiveStreamRenderer._process_real_frame actually applies the active
calibration before detection rather than just storing it unused.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import LineORM, SiteORM
from packages.cs_storage.repositories.calibration_repo import CalibrationRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from packages.cs_vision.calibration import CANVAS_SIZE, apply_perspective_warp, compute_homography
from services.api.main import app

client = TestClient(app)


def _setup_line() -> int:
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Perspective Calib Test Site")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()
        return line.id


# --- compute_homography / apply_perspective_warp ---------------------------

def test_compute_homography_requires_exactly_4_points():
    with pytest.raises(ValueError):
        compute_homography([[0, 0], [10, 0], [10, 10]])


def test_compute_homography_rejects_degenerate_points():
    # All 4 points identical -> the linear system OpenCV solves is singular.
    with pytest.raises(ValueError):
        compute_homography([[5, 5], [5, 5], [5, 5], [5, 5]])


def test_compute_homography_identity_rectangle_is_near_identity():
    w, h = CANVAS_SIZE
    src = [[0, 0], [w, 0], [w, h], [0, h]]
    matrix = compute_homography(src)
    assert np.allclose(np.array(matrix), np.eye(3), atol=1e-3)


def test_apply_perspective_warp_outputs_canvas_size():
    w, h = CANVAS_SIZE
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    src = [[50, 50], [600, 40], [610, 430], [40, 440]]
    matrix = compute_homography(src)
    warped = apply_perspective_warp(frame, matrix)
    assert warped.shape[:2] == (h, w)


def test_calibration_canvas_size_matches_train_rfdetr():
    # calibration.py deliberately duplicates this literal instead of importing
    # train_rfdetr.py (which pulls in torch/onnx, unavailable in cs-api's
    # lightweight container) -- guard against the two silently drifting apart.
    from packages.cs_vision.train_rfdetr import CANVAS_SIZE as TRAIN_CANVAS_SIZE
    assert CANVAS_SIZE == TRAIN_CANVAS_SIZE


# --- CalibrationRepository ---------------------------------------------------

def test_repo_create_and_get_active_perspective_calibration():
    line_id = _setup_line()
    src_points = [[10.0, 10.0], [600.0, 5.0], [610.0, 590.0], [5.0, 600.0]]
    homography = compute_homography(src_points)

    with get_sync_session() as db:
        repo = CalibrationRepository(db)
        calib = repo.create_perspective_calibration(
            line_id=line_id, roi_src_points=src_points, homography_matrix=homography,
        )
        assert calib.stage == "perspective"
        assert calib.is_active is True

        active = repo.get_active_calibration(line_id, stage="perspective")
        assert active is not None
        assert active.id == calib.id
        assert active.roi_src_points == src_points


def test_repo_new_perspective_calibration_deactivates_previous():
    line_id = _setup_line()
    src_a = [[0.0, 0.0], [640.0, 0.0], [640.0, 640.0], [0.0, 640.0]]
    src_b = [[20.0, 20.0], [620.0, 10.0], [630.0, 610.0], [10.0, 630.0]]

    with get_sync_session() as db:
        repo = CalibrationRepository(db)
        first = repo.create_perspective_calibration(
            line_id=line_id, roi_src_points=src_a, homography_matrix=compute_homography(src_a),
        )
        second = repo.create_perspective_calibration(
            line_id=line_id, roi_src_points=src_b, homography_matrix=compute_homography(src_b),
        )

        db.refresh(first)
        assert first.is_active is False
        assert second.is_active is True

        active = repo.get_active_calibration(line_id, stage="perspective")
        assert active.id == second.id


# --- API endpoint -------------------------------------------------------------

def _login(username: str, password: str) -> str:
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200
    return res.json()["token"]


@pytest.fixture(autouse=True)
def _seed_users():
    init_db_sync()
    with get_sync_session() as db:
        UserRepository(db).seed_default_users()


def test_create_perspective_calibration_endpoint_engineer_ok():
    line_id = _setup_line()
    token = _login("engineer", "eng123")
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post(
        f"/api/calibrations/{line_id}/perspective",
        json={"roi_src_points": [[10.0, 10.0], [600.0, 5.0], [610.0, 590.0], [5.0, 600.0]]},
        headers=headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["stage"] == "perspective"
    assert len(body["homography_matrix"]) == 3

    with get_sync_session() as db:
        active = CalibrationRepository(db).get_active_calibration(line_id, stage="perspective")
        assert active is not None
        assert active.id == body["calibration_id"]


def test_create_perspective_calibration_endpoint_rejects_bad_points():
    line_id = _setup_line()
    token = _login("engineer", "eng123")
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post(
        f"/api/calibrations/{line_id}/perspective",
        json={"roi_src_points": [[0.0, 0.0], [1.0, 0.0]]},
        headers=headers,
    )
    assert res.status_code == 400


def test_create_perspective_calibration_endpoint_forbidden_for_operator():
    line_id = _setup_line()
    token = _login("operator", "op123")
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post(
        f"/api/calibrations/{line_id}/perspective",
        json={"roi_src_points": [[10.0, 10.0], [600.0, 5.0], [610.0, 590.0], [5.0, 600.0]]},
        headers=headers,
    )
    assert res.status_code == 403


# --- LiveStreamRenderer wiring -------------------------------------------------

def test_renderer_loads_no_calibration_by_default():
    from packages.cs_counting.stream_renderer import LiveStreamRenderer

    line_id = _setup_line()
    renderer = LiveStreamRenderer(line_id=line_id)
    assert renderer.homography_matrix is None


def test_renderer_reload_picks_up_active_calibration():
    from packages.cs_counting.stream_renderer import LiveStreamRenderer

    line_id = _setup_line()
    renderer = LiveStreamRenderer(line_id=line_id)
    assert renderer.homography_matrix is None

    src_points = [[10.0, 10.0], [600.0, 5.0], [610.0, 590.0], [5.0, 600.0]]
    homography = compute_homography(src_points)
    with get_sync_session() as db:
        CalibrationRepository(db).create_perspective_calibration(
            line_id=line_id, roi_src_points=src_points, homography_matrix=homography,
        )

    renderer.reload_perspective_calibration()
    assert renderer.homography_matrix is not None
    assert np.allclose(np.array(renderer.homography_matrix), np.array(homography))


def test_process_real_frame_applies_warp_before_detection(monkeypatch):
    """The frame handed to CountingEngine.process_frame must be the warped
    one when a perspective calibration is active -- not the raw frame."""
    from packages.cs_counting import stream_renderer as sr_module

    line_id = _setup_line()
    renderer = sr_module.LiveStreamRenderer(line_id=line_id)

    src_points = [[10.0, 10.0], [600.0, 5.0], [610.0, 590.0], [5.0, 600.0]]
    homography = compute_homography(src_points)
    renderer.homography_matrix = homography

    # Fake video_cap so process_and_annotate_frame takes the _process_real_frame path.
    class _FakeCap:
        def isOpened(self):
            return True

    renderer.video_cap = _FakeCap()

    captured = {}

    class _FakeOutput:
        gate_crossings = []
        active_tracks = []

    def _fake_process_frame(image, frame_index, monotonic_ns, wall_clock):
        captured["image"] = image
        return _FakeOutput()

    monkeypatch.setattr(renderer.engine, "process_frame", _fake_process_frame)

    # A uniform-color frame would look identical before/after warping (only
    # border pixels would differ) -- use a gradient so a genuinely applied
    # warp is provably distinguishable from a no-op.
    raw_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    raw_frame[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)[None, :]
    raw_frame[:, :, 1] = np.linspace(0, 255, 640, dtype=np.uint8)[:, None]
    expected_warp = apply_perspective_warp(raw_frame, homography)

    renderer.process_and_annotate_frame(raw_frame, session_id=None)

    assert "image" in captured
    assert np.array_equal(captured["image"], expected_warp)
    assert not np.array_equal(captured["image"], raw_frame)
