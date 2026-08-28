"""Tests for InferenceWorker.run_step()'s active_bundle_id handling (§4.2, §4.5, §5.5).

active_bundle_id previously defaulted to a hardcoded 1 and only got
updated when a real active bundle existed -- on a line with no bundle
activated yet (or whose real bundle didn't happen to have id 1), this
meant ledger writes either recorded a real crossing against the wrong
bundle, or raised a real deployment_bundle_id foreign-key violation
(caught and logged, the crossing silently never recorded). Verifies the
fix: no active bundle means the write is honestly skipped (a clear
warning, no fabricated id), and a real one lets it proceed.
"""

from datetime import UTC, datetime

import numpy as np

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraORM,
    CountEventORM,
    LineORM,
    ModelVersionORM,
    ProductProfileORM,
    SiteORM,
)
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from services.inference.worker import InferenceWorker


class _FakeTransport:
    def __init__(self, frames):
        self._frames = frames
        self.released = []

    def consume(self, timeout_ms):
        frames, self._frames = self._frames, []
        return frames

    def get_image_data(self, shm_name):
        return np.zeros((640, 640, 3), dtype=np.uint8)

    def get_stats(self):
        return {"consecutive_drops": {}}

    def release(self, frame):
        self.released.append(frame)


def _setup_line_and_session() -> tuple[int, int, int]:
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Inference Bundle Context Test Site")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()
        cam = CameraORM(line_id=line.id, node_id=1, source_driver="rtsp")
        db.add(cam)
        db.commit()
        prof = ProductProfileORM(site_id=site.id, name="Bag", nominal_dims_mm={})
        db.add(prof)
        db.commit()
        sess = SessionRepository(db).create_session(line_id=line.id, product_profile_id=prof.id)
        return line.id, cam.id, sess.id


def _make_frame(camera_id: int):
    from packages.cs_core.frame import Frame
    return Frame(
        camera_id=camera_id, stream_epoch=1, frame_index=1,
        monotonic_ns=1, wall_clock=datetime.now(UTC),
        shm_name="shm_test", shape=(640, 640, 3),
    )


def test_run_step_skips_write_without_active_bundle(monkeypatch):
    line_id, cam_id, sess_id = _setup_line_and_session()
    transport = _FakeTransport([_make_frame(cam_id)])
    worker = InferenceWorker(transport=transport, line_id=line_id)
    assert worker.active_bundle_id is None

    # Real detector call would need a real model + real detections; skip
    # that entirely by making process_frame return a real, empty
    # FrameProcessingOutput so this test isolates active_bundle_id gating.
    from packages.cs_counting.engine import FrameProcessingOutput
    from packages.cs_vision.detector import DetectionResult

    def fake_process_frame(image, frame_index, monotonic_ns, wall_clock):
        return FrameProcessingOutput(
            frame_index=frame_index, monotonic_ns=monotonic_ns, wall_clock=wall_clock,
            detections=DetectionResult(), active_tracks=[], gate_crossings=[],
            running_net_count=0, area_estimate=0.0, discrepancy_flag=False,
        )

    monkeypatch.setattr(worker.engine, "process_frame", fake_process_frame)

    processed = worker.run_step()
    assert processed == 1

    with get_sync_session() as db:
        assert db.query(CountEventORM).filter(CountEventORM.session_id == sess_id).count() == 0


def test_run_step_applies_config_and_resolves_bundle_when_active(monkeypatch):
    line_id, cam_id, sess_id = _setup_line_and_session()
    with get_sync_session() as db:
        mv = ModelVersionORM(onnx_hash="test-hash", onnx_path="models/test.onnx")
        db.add(mv)
        db.commit()
        config_repo = ConfigRepository(db)
        cfg = config_repo.create_config_version(line_id=line_id, payload={"confidence_threshold": 0.66})
        bundle = config_repo.create_and_activate_bundle(line_id=line_id, model_version_id=mv.id, config_version_id=cfg.id)
        bundle_id = bundle.id

    transport = _FakeTransport([_make_frame(cam_id)])
    worker = InferenceWorker(transport=transport, line_id=line_id)
    assert worker.active_bundle_id is None  # not resolved until run_step() actually processes a batch

    from packages.cs_counting.engine import FrameProcessingOutput
    from packages.cs_vision.detector import DetectionResult

    def fake_process_frame(image, frame_index, monotonic_ns, wall_clock):
        return FrameProcessingOutput(
            frame_index=frame_index, monotonic_ns=monotonic_ns, wall_clock=wall_clock,
            detections=DetectionResult(), active_tracks=[], gate_crossings=[],
            running_net_count=0, area_estimate=0.0, discrepancy_flag=False,
        )

    monkeypatch.setattr(worker.engine, "process_frame", fake_process_frame)

    processed = worker.run_step()
    assert processed == 1
    assert worker.active_bundle_id == bundle_id
    assert worker.engine.detector.conf_threshold == 0.66
