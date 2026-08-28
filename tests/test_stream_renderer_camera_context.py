"""Tests for LiveStreamRenderer.reload_camera_context() (§5.2, §5.5) --
previously zero coverage for this module.

Both frame-processing paths used to hardcode camera_id=1, gate_id=1, and
stream_epoch=4 regardless of what was actually configured for the line --
harmless only by coincidence when the real rows happened to have id 1,
and a real ForeignKeyViolation (silently caught and logged, the crossing
just never recorded) on any other line. These tests cover the real fix:
resolving actual camera/gate/bundle rows, and honestly skipping the ledger
write (not fabricating an id) when they don't exist yet.
"""

import numpy as np

from packages.cs_counting.stream_renderer import LiveStreamRenderer
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraORM,
    CountEventORM,
    GateORM,
    LineORM,
    ModelVersionORM,
    ProductProfileORM,
    SiteORM,
)
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.session_repo import SessionRepository


def _setup_line() -> int:
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Renderer Camera Context Test Site")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()
        return line.id


def _add_real_camera_gate_bundle(line_id: int) -> tuple[int, int]:
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
        return cam.id, gate.id


def _open_session(line_id: int) -> int:
    with get_sync_session() as db:
        prof = ProductProfileORM(site_id=1, name="Bag", nominal_dims_mm={})
        db.add(prof)
        db.commit()
        sess = SessionRepository(db).create_session(line_id=line_id, product_profile_id=prof.id)
        return sess.id


def test_reload_camera_context_none_when_nothing_configured():
    line_id = _setup_line()
    renderer = LiveStreamRenderer(line_id=line_id)
    assert renderer.camera_id is None
    assert renderer.gate_id is None
    assert renderer.stream_epoch == 1


def test_reload_camera_context_resolves_real_ids():
    line_id = _setup_line()
    cam_id, gate_id = _add_real_camera_gate_bundle(line_id)

    renderer = LiveStreamRenderer(line_id=line_id)
    assert renderer.camera_id == cam_id
    assert renderer.gate_id == gate_id
    assert renderer.active_bundle_id is not None


def test_reload_camera_context_syncs_gate_state_machine_gate_id():
    """GateStateMachine.gate_id defaults to 1 and update_geometry() never
    touches it -- reload_camera_context() is the only place it gets synced
    to a real GateORM row, so every real crossing carries the right id."""
    line_id = _setup_line()
    _, gate_id = _add_real_camera_gate_bundle(line_id)

    renderer = LiveStreamRenderer(line_id=line_id)
    assert renderer.engine.gate_state_machine.gate_id == gate_id


def test_simulated_frame_skips_ledger_write_without_real_camera_gate_bundle():
    """No camera/gate/bundle configured yet -> the crossing is honestly
    dropped (logged), not recorded with a fabricated id -- verify no row
    is written at all, and nothing raises."""
    line_id = _setup_line()
    sess_id = _open_session(line_id)
    renderer = LiveStreamRenderer(line_id=line_id)

    # Force an immediate crossing: park a bag right at the gate line.
    renderer.bags[0]["x"] = float(renderer.gate_x)

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    renderer.process_and_annotate_frame(frame, session_id=sess_id)

    with get_sync_session() as db:
        count = db.query(CountEventORM).filter(CountEventORM.session_id == sess_id).count()
        assert count == 0


def test_simulated_frame_records_ledger_write_with_real_camera_gate_bundle():
    line_id = _setup_line()
    cam_id, gate_id = _add_real_camera_gate_bundle(line_id)
    sess_id = _open_session(line_id)
    renderer = LiveStreamRenderer(line_id=line_id)

    renderer.bags[0]["x"] = float(renderer.gate_x)

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    renderer.process_and_annotate_frame(frame, session_id=sess_id)

    with get_sync_session() as db:
        events = db.query(CountEventORM).filter(CountEventORM.session_id == sess_id).all()
        assert len(events) == 1
        assert events[0].camera_id == cam_id
        assert events[0].gate_id == gate_id
        assert events[0].deployment_bundle_id == renderer.active_bundle_id
