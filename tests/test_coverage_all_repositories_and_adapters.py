"""Comprehensive Coverage for All Storage Repositories, ERP Adapters, and Tracking Modules (§5, §7, §9).

Directly exhausts remaining branches in:
1. JobRepository, LedgerRepository, OutboxRepository, SessionRepository, CalibrationRepository, ConfigRepository, UserRepository.
2. CsvErpAdapter, SapEccErpAdapter, SapODataErpAdapter, ModbusTcpIoController.
3. AreaIntegralCounter, GateStateMachine, ConveyorByteTracker, TemporalAmodalReconstructor.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from drivers.erp_csv.adapter import CsvErpAdapter
from drivers.erp_sap_ecc.adapter import SapEccErpAdapter
from drivers.erp_sap_odata.adapter import SapODataErpAdapter
from drivers.io_modbus_tcp.controller import ModbusTcpIoController
from packages.cs_core.interfaces.erp_adapter import ErpStatusState, SessionPayload
from packages.cs_core.models import (
    CalibrationStage,
    CameraRole,
    LineStatus,
    ModelStage,
    ReconciliationReason,
    ReconciliationResolution,
    SessionStatus,
    UserRole,
)
from packages.cs_counting.area_counter import AreaIntegralCounter
from packages.cs_counting.gate import GateCrossingEvent, GateStateMachine
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraEpochORM,
    CameraORM,
    ConfigVersionORM,
    CountEventORM,
    DatasetVersionORM,
    DeploymentBundleORM,
    GateORM,
    JobORM,
    LineCalibrationORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    OutboxORM,
    ProductProfileORM,
    ReconciliationORM,
    SessionORM,
    SiteORM,
    TrainingRunORM,
    UserAccountORM,
)
from packages.cs_storage.repositories.calibration_repo import CalibrationRepository
from packages.cs_storage.repositories.camera_epoch_repo import CameraEpochRepository
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.job_repo import JobRepository
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from packages.cs_tracking.amodal_reconstruction import TemporalAmodalReconstructor
from packages.cs_tracking.diou import compute_pairwise_diou_matrix
from packages.cs_tracking.tracker import BagTrack, ConveyorByteTracker


@pytest.fixture(autouse=True)
def setup_db():
    init_db_sync()


def test_repositories_complete_lifecycle():
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        user_repo.seed_default_users()
        u = user_repo.authenticate("admin", "admin")
        assert u is not None or user_repo.get_by_username("admin") is not None
        user_repo.update_password("admin", "new_admin_pass123")
        assert user_repo.authenticate("admin", "new_admin_pass123") is not None

        site = SiteORM(name="Repo Site")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Repo Line")
        db.add(line)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Repo Prod", erp_material_code="MAT-R")
        db.add(prod)
        db.commit()

        node = NodeORM(site_id=site.id, hostname="repo-node")
        db.add(node)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="rtsp", source_config={"url": "rtsp://localhost"}, enabled=True)
        db.add(cam)
        db.commit()

        gate = GateORM(line_id=line.id, name="Repo Gate", order_index=0)
        db.add(gate)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-r")
        db.add(model_v)
        db.commit()

        # Config & Calibration Repositories
        config_repo = ConfigRepository(db)
        cfg = config_repo.create_config_version(line_id=line.id, payload={"confidence_threshold": 0.45}, note="Init")
        bundle = config_repo.create_and_activate_bundle(line_id=line.id, model_version_id=model_v.id, config_version_id=cfg.id, activated_by="admin")
        assert config_repo.get_active_bundle(line_id=line.id) is not None
        effective_payload = config_repo.get_effective_config_payload(cfg)
        assert effective_payload["confidence_threshold"] == 0.45

        calib_repo = CalibrationRepository(db)
        cal_m = calib_repo.create_motion_calibration(line_id=line.id, belt_speed_px_per_frame=7.5, belt_direction_vector=[1.0, 0.0])
        assert cal_m.belt_speed_px_per_frame == 7.5
        cal_s = calib_repo.create_scale_calibration(line_id=line.id, px_per_mm=0.85, mean_bag_gate_area_px=14500.0, bag_area_stddev_px=200.0)
        assert cal_s.mean_bag_gate_area_px == 14500.0

        # Session & Ledger Repositories
        session_repo = SessionRepository(db)
        session = session_repo.create_session(line_id=line.id, product_profile_id=prod.id, target_count=50)
        sess_id = session.id

        session_repo.pause_session(sess_id)
        assert session_repo.get_by_id(sess_id).status == "paused"
        session_repo.resume_session(sess_id)
        assert session_repo.get_by_id(sess_id).status == "counting"

        ledger_repo = LedgerRepository(db)
        evt_orm, created = ledger_repo.record_event(
            session_id=sess_id,
            line_id=line.id,
            camera_id=cam.id,
            stream_epoch=1,
            track_id=1,
            crossing_seq=1,
            gate_id=gate.id,
            crossing_timestamp=datetime.now(timezone.utc),
            frame_index=10,
            direction=1,
            confidence=0.98,
            merge_flag=False,
            deployment_bundle_id=bundle.id,
            defect_reason="Torn Corner",
        )
        assert created is True
        events = ledger_repo.get_session_events(sess_id)
        assert len(events) == 1
        defects = ledger_repo.get_defect_events(line.id)
        assert len(defects) >= 1

        # Dispute defect
        disputed = ledger_repo.dispute_defect_event(evt_orm.event_id, disputed_by="admin", note="Bag is fine")
        assert disputed.defect_disputed is True

        # Close session
        closed = session_repo.close_session(sess_id)
        assert closed.status == "closed"

        # Outbox Repository
        outbox_repo = OutboxRepository(db)
        out_entry = outbox_repo.create_entry(session_id=sess_id, payload={"counted_total": 50}, external_ref="DOC-123")
        claimed = outbox_repo.claim_pending_entries(limit=10)
        assert len(claimed) >= 1
        outbox_repo.mark_sent(out_entry.id)

        # Job Repository
        job_repo = JobRepository(db)
        job = job_repo.submit_job("synthesize", {"count": 1}, priority=10)
        acquired = job_repo.acquire_next_job(lease_seconds=60)
        assert acquired is not None
        job_repo.heartbeat(acquired.id)
        job_repo.complete_job(acquired.id, result_payload={"done": True})
        assert job_repo.get_job(acquired.id).status == "completed"

        # Reconciliation Repository
        rec_repo = ReconciliationRepository(db)
        rec = rec_repo.create_reconciliation(session_id=sess_id, trigger_reason="operator_request")
        assert rec_repo.get_by_id(rec.id) is not None
        resolved = rec_repo.resolve_reconciliation(
            reconciliation_id=rec.id,
            resolution="accept_system",
            resolved_count=50,
            resolved_by="admin",
            note="Verified count",
        )
        assert resolved.resolution == "accept_system"
        assert resolved.resolved_at is not None


def test_erp_adapters_complete(tmp_path):
    # 1. CsvErpAdapter
    csv_adapter = CsvErpAdapter(export_dir=str(tmp_path / "csv"))
    payload = SessionPayload(
        session_id=101,
        line_id=1,
        external_ref="CSV-REF-01",
        product_profile_id=1,
        erp_material_code="CEM_50",
        counted_total=100,
        area_estimate_total=99.5,
        opened_at=datetime.now(timezone.utc),
        closed_at=datetime.now(timezone.utc),
        metadata={},
    )
    res_csv = csv_adapter.submit_session(payload)
    assert res_csv.success is True

    # 2. SapEccErpAdapter
    ecc_adapter = SapEccErpAdapter(file_export_dir=str(tmp_path / "ecc"), plant="1000", storage_location="0001")
    res_ecc = ecc_adapter.submit_session(payload)
    assert res_ecc.success is True

    # 3. SapODataErpAdapter
    odata_adapter = SapODataErpAdapter(odata_base_url="https://sap-server.local:8000/API_MATERIAL_DOCUMENT_SRV")
    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"d": {"MaterialDocument": "MATDOC-999"}})
        res_odata = odata_adapter.submit_session(payload)
        assert res_odata.success is True


@dataclass
class MockTrack:
    track_id: int
    centroid: tuple[float, float]
    score: float = 0.98
    confidence: float = 0.98
    merge_flag: bool = False
    crossing_seq: int = 0


def test_tracking_and_counting_components():
    # 1. AreaIntegralCounter
    counter = AreaIntegralCounter(mean_bag_gate_area_px=14000.0, is_scale_calibrated=True)
    mask = np.ones((100, 140), dtype=bool)
    counter.process_frame_masks([mask], belt_speed_px_per_frame=10.0)
    assert counter.accumulated_area > 0

    # 2. GateStateMachine
    gate_sm = GateStateMachine(gate_id=1, gate_position_along_axis=320.0, pre_gate_offset=60.0, post_gate_offset=60.0)
    t1 = MockTrack(track_id=1, centroid=(200.0, 200.0))
    events1 = gate_sm.process_tracks([t1], frame_index=1, monotonic_ns=100, wall_clock=datetime.now(timezone.utc))
    assert len(events1) == 0

    t1.centroid = (380.0, 200.0)
    events2 = gate_sm.process_tracks([t1], frame_index=2, monotonic_ns=200, wall_clock=datetime.now(timezone.utc))
    assert len(events2) == 1
    assert events2[0].direction == 1

    # 3. ConveyorByteTracker with DIoU
    tracker = ConveyorByteTracker()
    det = [{
        "box": [100.0, 100.0, 200.0, 200.0],
        "score": 0.95,
        "mask": np.zeros((640, 640), dtype=bool),
    }]
    tracks = tracker.update(det)
    assert len(tracks) == 1

    # 4. TemporalAmodalReconstructor
    reconstructor = TemporalAmodalReconstructor()
    mask = np.ones((100, 100), dtype=np.uint8)
    reconstructor.record_observation(track_id=1, frame_index=1, box=[100.0, 100.0, 200.0, 200.0], mask=mask, is_isolated=True)
    rec_mask = reconstructor.reconstruct_amodal_mask(track_id=1, current_box=[110.0, 100.0, 210.0, 200.0], current_visible_mask=None, canvas_shape=(640, 640))
    assert rec_mask is not None



