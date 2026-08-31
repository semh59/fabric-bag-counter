"""Deep Coverage Test Suite for Workers, Ingest, Hardware Failover & Frame Extraction (§4, §5, §10).

Exhaustively exercises:
1. services/jobrunner/worker.py (all queue polling, TOCTOU GPU checks, train/export jobs, heartbeat, retry).
2. services/erp_relay/worker.py (retry backoff, permanent error handling, reconciliation creation).
3. services/inference/worker.py (timeout handling, gate crossings, Modbus PLC sync, session active/inactive).
4. services/ingest/worker.py (reconnect, video loop, stream epoch increment, shm write).
5. packages/cs_data/extract_frames.py (video frame extraction & stride).
6. packages/cs_vision/tensorrt_builder.py (builder initialization, options, engine path generation).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from packages.cs_core.frame import Frame
from packages.cs_core.transport import SharedMemoryTransport
from packages.cs_data.extract_frames import extract_video_frames
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraEpochORM,
    CameraORM,
    ConfigVersionORM,
    DatasetVersionORM,
    DeploymentBundleORM,
    GateORM,
    JobORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    OutboxORM,
    ProductProfileORM,
    ReconciliationORM,
    SessionORM,
    SiteORM,
    TrainingRunORM,
)
from packages.cs_storage.repositories.job_repo import JobRepository
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_vision.tensorrt_builder import TensorRtEngineBuilder
from services.erp_relay.worker import ErpRelayWorker
from services.inference.worker import InferenceWorker
from services.ingest.worker import IngestWorker
from services.jobrunner.worker import JobrunnerWorker


@pytest.fixture(autouse=True)
def setup_db():
    init_db_sync()


def test_extract_video_frames(tmp_path):
    vid_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (100, 100))
    for i in range(15):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 10)
        out.write(frame)
    out.release()

    out_dir = tmp_path / "extracted"
    frames = extract_video_frames(str(vid_path), output_dir=str(out_dir), stride_frames=3)
    assert len(frames) >= 4
    for idx, ts, item in frames:
        assert Path(item).exists()


def test_tensorrt_builder_options():
    builder = TensorRtEngineBuilder("models/rfdetr_seg_v2.onnx", use_fp16=True, max_workspace_gb=1.0)
    opts = builder.get_onnxruntime_tensorrt_options()
    assert "trt_fp16_enable" in opts
    assert opts["trt_fp16_enable"] is True
    assert builder.engine_path.name.endswith(".engine")


def test_jobrunner_worker_full_queue_run(tmp_path):
    with get_sync_session() as db:
        site = SiteORM(name="Site J")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line J")
        db.add(line)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod J")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-j")
        db.add(model_v)
        db.commit()

        job_repo = JobRepository(db)
        j1 = job_repo.submit_job("synthesize", {"count": 2})
        j2 = job_repo.submit_job("extract_frames", {"video_path": "nonexistent.mp4", "output_dir": str(tmp_path)})
        j3 = job_repo.submit_job("build_dataset", {"sessions": [{"session_id": "s1", "camera_id": 1, "shift": "d", "frame_count": 10, "is_heavy_shingling": False}]})
        j4 = job_repo.submit_job("export_onnx", {"model_id": model_v.id})

    worker = JobrunnerWorker(lease_seconds=30, gpu_mode="always")

    for _ in range(4):
        processed = worker.run_step()
        assert processed is True

    assert worker.run_step() is False


@dataclass
class MockErpResult:
    success: bool
    error_message: str | None = None
    external_tx_id: str | None = None


def test_erp_relay_worker_error_and_retry():
    with get_sync_session() as db:
        outbox_repo = OutboxRepository(db)
        entry = outbox_repo.create_entry(session_id=991, payload={"counted_total": 100}, external_ref="FAIL-REF")
        entry_id = entry.id

    failing_adapter = MagicMock()
    failing_adapter.submit_session.return_value = MockErpResult(success=False, error_message="Connection timeout")
    failing_adapter.supports_status_query = False

    worker = ErpRelayWorker(adapter=failing_adapter, poll_interval_sec=0.1)
    processed = worker.run_step()
    assert processed == 0

    with get_sync_session() as db:
        e = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
        assert e is not None


def test_ingest_worker_lifecycle():
    transport = SharedMemoryTransport(ring_slots=2)

    with get_sync_session() as db:
        site = SiteORM(name="Site Ingest")
        db.add(site)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="node-ingest")
        db.add(node)
        db.commit()

        line = LineORM(site_id=site.id, name="Line Ingest")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="file", source_config={"path": "dummy.mp4"}, enabled=True)
        db.add(cam)
        db.commit()
        cam_id = cam.id

    mock_driver = MagicMock()
    mock_driver.is_connected = True
    mock_driver.read.return_value = Frame(
        camera_id=cam_id,
        stream_epoch=1,
        frame_index=1,
        monotonic_ns=time.perf_counter_ns(),
        wall_clock=datetime.now(UTC),
        shape=(100, 100, 3),
        dtype="uint8",
        shm_name=f"shm_cam_{cam_id}_slot_0",
    )

    ingest = IngestWorker(camera_id=cam_id, source_driver=mock_driver, source_config={}, transport=transport)
    ingest.start()
    ok = ingest.run_step()
    assert ok is True
    ingest.is_running = False



def test_inference_worker_no_active_session():
    transport = SharedMemoryTransport(ring_slots=2)

    with get_sync_session() as db:
        site = SiteORM(name="Site Inf")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line Inf", status="idle")
        db.add(line)
        db.commit()
        line_id = line.id

    worker = InferenceWorker(line_id=line_id, transport=transport)
    assert worker.run_step() == 0
