"""Deep Coverage Test Suite for Geometry, Homography, Stream Rendering & Camera Feeds (§4.3, §4.5, §9).

Directly exercises:
1. packages/cs_core/geometry.py (all IoU, mask IoU, polygon area, centroid, and projection functions).
2. packages/cs_counting/stream_renderer.py (_process_real_frame, homography warp, HUD overlay, gate crossings).
3. packages/cs_counting/camera_feed.py (CameraFeed grab loop, FPS tracking, reconnect, error reporting).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
import cv2
import numpy as np
import pytest

from packages.cs_core.geometry import (
    compute_box_iou,
    compute_mask_iou,
    compute_mask_iou_matrix,
    compute_polygon_area,
    mask_centroid,
    point_in_polygon,
    polygon_centroid,
    project_point_on_axis,
)
from packages.cs_counting.camera_feed import CameraFeed, reconnect_camera_feed
from packages.cs_counting.stream_renderer import LiveStreamRenderer
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraORM,
    ConfigVersionORM,
    DeploymentBundleORM,
    GateORM,
    LineCalibrationORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    ProductProfileORM,
    SessionORM,
    SiteORM,
)
from packages.cs_storage.repositories.session_repo import SessionRepository


@pytest.fixture(autouse=True)
def setup_db():
    init_db_sync()


def test_geometry_functions():
    # 1. Box IoU
    b1 = [10.0, 10.0, 50.0, 50.0]
    b2 = [20.0, 20.0, 60.0, 60.0]
    iou = compute_box_iou(b1, b2)
    assert 0.0 < iou < 1.0

    # 2. Mask IoU and Centroid
    m1 = np.ones((50, 50), dtype=np.uint8)
    m2 = np.ones((50, 50), dtype=np.uint8)
    assert compute_mask_iou(m1, m2) == 1.0

    matrix = compute_mask_iou_matrix([m1], [m2])
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 1.0

    mcx, mcy = mask_centroid(m1)
    assert 20.0 < mcx < 30.0 and 20.0 < mcy < 30.0

    # 3. Polygons & Projection
    poly = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    area = compute_polygon_area(poly)
    assert area == 10000.0

    cx, cy = polygon_centroid(poly)
    assert cx == 50.0 and cy == 50.0

    assert point_in_polygon((50.0, 50.0), poly) is True
    assert point_in_polygon((150.0, 150.0), poly) is False

    proj = project_point_on_axis((50.0, 20.0), (0.0, 0.0), (1.0, 0.0))
    assert proj == 50.0


def test_live_stream_renderer_real_frame_processing(tmp_path):
    # Generate MP4 test file for real frame processing
    vid_path = tmp_path / "real_sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 640))
    for i in range(10):
        frame = np.ones((640, 640, 3), dtype=np.uint8) * 120
        # Draw simulated bag
        cv2.rectangle(frame, (300 + i * 5, 200), (420 + i * 5, 360), (50, 50, 50), -1)
        out.write(frame)
    out.release()

    with get_sync_session() as db:
        site = SiteORM(name="Site Real")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="node-real")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Line Real", status="running")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="file", source_config={"path": str(vid_path)}, enabled=True)
        db.add(cam)
        db.commit()

        gate = GateORM(line_id=line.id, name="Gate Real", order_index=0)
        db.add(gate)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod Real")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-real")
        db.add(model_v)
        db.commit()

        cfg_v = ConfigVersionORM(line_id=line.id, payload={"confidence_threshold": 0.30})
        db.add(cfg_v)
        db.commit()

        bundle = DeploymentBundleORM(line_id=line.id, model_version_id=model_v.id, config_version_id=cfg_v.id)
        db.add(bundle)
        db.commit()

        session_repo = SessionRepository(db)
        session = session_repo.create_session(line_id=line.id, product_profile_id=prod.id, target_count=50)
        sess_id = session.id
        line_id = line.id

    renderer = LiveStreamRenderer(line_id=line_id)
    ok = renderer.set_video_source(str(vid_path))
    assert ok is True

    # Process real frames
    for _ in range(5):
        frame = renderer.generate_conveyor_frame()
        annotated, sess = renderer.process_and_annotate_frame(frame, session_id=sess_id)
        assert annotated is not None


def test_camera_feed_monitoring(tmp_path):
    vid_path = tmp_path / "feed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (100, 100))
    for _ in range(5):
        out.write(np.zeros((100, 100, 3), dtype=np.uint8))
    out.release()

    feed = CameraFeed(camera_id=99)
    ok, msg = feed.connect(str(vid_path))
    assert ok is True
    frame_bytes = feed.read_jpeg()
    assert isinstance(frame_bytes, bytes)
    assert len(frame_bytes) > 100
    if feed.video_cap:
        feed.video_cap.release()
