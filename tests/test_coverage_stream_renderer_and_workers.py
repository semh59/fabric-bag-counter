"""Exhaustive Stream Renderer, Ingest & Inference Workers Coverage Test Suite (§4.5, §9.6).

Directly exercises LiveStreamRenderer frame processing, HUD overlays, manual bag spawning,
and Ingest/Inference worker step functions.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
import numpy as np
import pytest

from packages.cs_core.frame import Frame
from packages.cs_core.transport import SharedMemoryTransport
from packages.cs_counting.stream_renderer import LiveStreamRenderer, get_stream_generator
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
    SessionORM,
    SiteORM,
)
from packages.cs_storage.repositories.session_repo import SessionRepository
from services.inference.worker import InferenceWorker
from services.ingest.worker import IngestWorker


@pytest.fixture(autouse=True)
def setup_renderer_env():
    init_db_sync()


def test_live_stream_renderer_all_features():
    with get_sync_session() as db:
        site = SiteORM(name="Site R")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="node-r")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Line R", status="running")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="rtsp", source_config={"url": "rtsp://localhost"}, enabled=True)
        db.add(cam)
        db.commit()

        gate = GateORM(line_id=line.id, name="Gate R", order_index=0)
        db.add(gate)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod R")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-r")
        db.add(model_v)
        db.commit()

        cfg_v = ConfigVersionORM(line_id=line.id, payload={"confidence_threshold": 0.40})
        db.add(cfg_v)
        db.commit()

        bundle = DeploymentBundleORM(line_id=line.id, model_version_id=model_v.id, config_version_id=cfg_v.id)
        db.add(bundle)
        db.commit()

        session_repo = SessionRepository(db)
        session = session_repo.create_session(line_id=line.id, product_profile_id=prod.id, target_count=100)
        sess_id = session.id
        line_id = line.id

    renderer = LiveStreamRenderer(line_id=line_id)

    # 1. Spawn manual bags with defects and multi-count
    renderer.spawn_manual_bag(defective=True, bag_count_estimate=1, label="Torn Bag")
    renderer.spawn_manual_bag(defective=False, bag_count_estimate=2, label="Merged")

    # 2. Process conveyor frames
    frame = renderer.generate_conveyor_frame()
    annotated, active_sess = renderer.process_and_annotate_frame(frame, session_id=sess_id)
    assert annotated.ndim == 3
    assert active_sess == sess_id

    # 3. Source switching
    ok, _ = renderer.set_camera_source("demo")
    assert ok is True

    # 4. Stream generator iterator
    gen = get_stream_generator(line_id=line_id)
    chunk = next(gen)
    assert chunk.startswith(b"--frame\r\n")


def test_ingest_and_inference_worker_step():
    transport = SharedMemoryTransport(ring_slots=4)

    with get_sync_session() as db:
        site = SiteORM(name="Site W")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="node-w")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Line W", status="running")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="file", source_config={"path": "dummy.mp4"}, enabled=True)
        db.add(cam)
        db.commit()

        cam_id = cam.id
        line_id = line.id

    ingest = IngestWorker(camera_id=cam_id, source_driver="file", source_config={"path": "dummy.mp4"}, transport=transport)
    inference = InferenceWorker(line_id=line_id, transport=transport)

    # Write dummy frame to transport
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    transport.write_image_data(f"shm_test_cam_{cam_id}_slot_0", dummy_img)
    transport.publish(Frame(
        camera_id=cam_id,
        stream_epoch=1,
        frame_index=1,
        monotonic_ns=time.perf_counter_ns(),
        wall_clock=datetime.now(UTC),
        shape=(480, 640, 3),
        dtype="uint8",
        shm_name=f"shm_test_cam_{cam_id}_slot_0",
    ))

    # Run single inference worker step
    processed = inference.run_step()
    assert processed == 1

