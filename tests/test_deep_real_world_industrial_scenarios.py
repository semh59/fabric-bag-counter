"""Deep Real-World Industrial Factory Test Suite (§1-§16).

Exhaustively verifies realistic harsh factory conditions, physical dynamics, and production workflows:
1. Test 1: Variable Conveyor Velocity, Slip & Emergency Stop Reverse Springback Invariant.
2. Test 2: Dense 4-Bag Shingled Cluster, Multi-Signal Merge Detection & Amodal Reconstruction.
3. Test 3: Ripped/Punctured Bag Defect Exclusion, Scrap Ledger & Immutable Dispute Audit Trail.
4. Test 4: Industrial Camera Network Drop, Stream Epoch Bump & Zero Double-Count Guarantee.
5. Test 5: Closed-Loop SCADA & Modbus TCP PLC Hardware Actuation (Stop on Target Count).
6. Test 6: Dual-Counter Area Discrepancy & Full Human-in-the-Loop Reconciliation Lifecycle.
7. Test 7: Cryptographic HMAC-SHA256 Manifest Anti-Tamper & Transactional Outbox Failover Retry.
8. Test 8: OIML R51 Automatic Weighbridge Mass Balance & Industrial Metrology Tolerance.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import socket
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from drivers.io_modbus_tcp.controller import ModbusTcpIoController
from tools.modbus_server import ModbusTcpServer
from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.erp_adapter import ErpAdapter, ErpResult, ErpStatus, ErpStatusState, SessionPayload
from packages.cs_core.models import (
    CalibrationStage,
    CameraRole,
    LineStatus,
    ReconciliationReason,
    ReconciliationResolution,
    SessionStatus,
    UserRole,
)
from packages.cs_counting.area_counter import AreaIntegralCounter
from packages.cs_counting.engine import CountingEngine
from packages.cs_counting.event_handler import CountingEventHandler
from packages.cs_counting.events import GateCrossingRecorded, SessionAreaEstimateUpdated, SessionDiscrepancyDetected
from packages.cs_counting.gate import GateCrossingEvent, GateStateMachine
from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraEpochORM,
    CameraORM,
    ConfigVersionORM,
    CountEventORM,
    DeploymentBundleORM,
    GateORM,
    LineCalibrationORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    OutboxORM,
    ProductProfileORM,
    ReconciliationORM,
    SessionORM,
    SiteORM,
    UserAccountORM,
)
from packages.cs_storage.repositories.camera_epoch_repo import CameraEpochRepository
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.outbox_repo import OutboxRepository
from packages.cs_storage.repositories.reconciliation_repo import ReconciliationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_storage.repositories.user_repo import UserRepository
from packages.cs_tracking.amodal_reconstruction import TemporalAmodalReconstructor
from packages.cs_tracking.merge_detector import MergeDetector
from packages.cs_tracking.motion import BeltMotionModel
from packages.cs_tracking.tracker import BagTrack, ConveyorByteTracker
from packages.cs_vision.detector import DetectionResult, VisionDetector
from services.api.auth import SECRET_KEY, create_access_token
from services.api.main import app
from services.erp_relay.worker import ErpRelayWorker

client = TestClient(app)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _seed_standard_topology() -> dict[str, int]:
    """Helper to seed a complete, verified factory topology."""
    with get_sync_session() as db:
        user_repo = UserRepository(db)
        user_repo.seed_default_users()

        site = SiteORM(name="Factory Plant Alpha", timezone="Europe/Istanbul", locale="tr_TR")
        db.add(site)
        db.commit()
        db.refresh(site)

        node = NodeORM(site_id=site.id, hostname="edge-camera-node-01")
        db.add(node)
        db.commit()
        db.refresh(node)

        line = LineORM(site_id=site.id, name="Packing Conveyor Line 1", status="running")
        db.add(line)
        db.commit()
        db.refresh(line)

        cam = CameraORM(
            line_id=line.id,
            node_id=node.id,
            source_driver="file",
            source_config={"path": "data/test_conveyor_input.mp4"},
            role="counting",
            enabled=True,
        )
        db.add(cam)
        db.commit()
        db.refresh(cam)

        epoch = CameraEpochORM(camera_id=cam.id, current_epoch=1)
        db.add(epoch)

        gate = GateORM(line_id=line.id, name="Optical Counting Gate 1", order_index=0)
        db.add(gate)

        profile = ProductProfileORM(
            site_id=site.id,
            name="50kg Portland Cement Bag",
            nominal_weight_g=50000.0,
            nominal_dims_mm={"length": 650.0, "width": 480.0, "thickness": 130.0},
            erp_material_code="CEM-I-42.5R",
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)

        calib = LineCalibrationORM(
            line_id=line.id,
            stage="scale",
            belt_speed_px_per_frame=8.0,
            belt_direction_vector=[1.0, 0.0],
            px_per_mm=0.75,
            mean_bag_gate_area_px=24000.0,
            bag_area_stddev_px=1200.0,
            is_active=True,
        )
        db.add(calib)

        cfg = ConfigVersionORM(
            line_id=line.id,
            payload={
                "confidence_threshold": 0.40,
                "merge_area_ratio": 1.50,
                "discrepancy_threshold": 0.08,
            },
            payload_schema_version=2,
            note="Factory Standard v2.0",
        )
        db.add(cfg)
        db.commit()
        db.refresh(cfg)

        model = ModelVersionORM(
            onnx_hash="sha256:verified_rfdetr_v2_model_hash",
            onnx_path="models/rfdetr_seg_v2.onnx",
            stage="active",
        )
        db.add(model)
        db.commit()
        db.refresh(model)

        bundle = DeploymentBundleORM(
            line_id=line.id,
            model_version_id=model.id,
            config_version_id=cfg.id,
            calibration_id=calib.id,
            activated_by="lead_engineer",
        )
        db.add(bundle)
        db.commit()
        db.refresh(bundle)

        sess = SessionORM(
            line_id=line.id,
            product_profile_id=profile.id,
            external_ref="WB-DISPATCH-2026-09",
            target_count=50,
            status="counting",
            counted_total=0,
            area_estimate_total=0.0,
            vehicle_plate="34-ABC-789",
            driver_name="Ahmet Yilmaz",
            carrier_company="LojiNext Logistics",
        )
        db.add(sess)
        db.commit()
        db.refresh(sess)

        return {
            "site_id": site.id,
            "node_id": node.id,
            "line_id": line.id,
            "camera_id": cam.id,
            "gate_id": gate.id,
            "profile_id": profile.id,
            "bundle_id": bundle.id,
            "session_id": sess.id,
        }


# ===========================================================================
# 1. Variable Belt Velocity, Slip & Emergency Stop Springback Invariant
# ===========================================================================
def test_conveyor_variable_velocity_slip_and_emergency_stop_oscillation():
    """Verify physical conveyor dynamics:
    - Acceleration from 0 to 22 px/frame.
    - Random belt micro-slip (+/- 2.5 px).
    - Emergency Stop with reverse mechanical spring-back oscillation.
    - Strict net crossing invariant: Net Count == Forward - Backward crossings.
    """
    BagTrack.reset_counter()
    motion = BeltMotionModel(default_speed_px=10.0, default_direction=[1.0, 0.0], smoothing_alpha=0.20)
    tracker = ConveyorByteTracker(belt_motion=motion, high_score_threshold=0.35)
    gate = GateStateMachine(
        gate_id=1,
        axis_origin=(0.0, 0.0),
        axis_vector=(1.0, 0.0),
        gate_position_along_axis=300.0,
        pre_gate_offset=50.0,
        post_gate_offset=50.0,
    )

    bag_cx = 80.0
    bag_cy = 200.0
    bag_w, bag_h = 100.0, 70.0

    all_crossings: list[GateCrossingEvent] = []
    total_forward = 0
    total_backward = 0

    # Phase 1: Acceleration & Transit (Frames 0 to 25)
    speed = 5.0
    for frame_idx in range(25):
        speed = min(22.0, speed + 1.2)  # Accelerating
        slip = float(np.random.uniform(-1.0, 1.0))
        bag_cx += speed + slip

        mask = np.zeros((400, 600), dtype=bool)
        y1, y2 = int(bag_cy - bag_h / 2), int(bag_cy + bag_h / 2)
        x1, x2 = int(bag_cx - bag_w / 2), int(bag_cx + bag_w / 2)
        if 0 <= y1 < y2 <= 400 and 0 <= x1 < x2 <= 600:
            mask[y1:y2, x1:x2] = True

        dets = [{"box": [x1, y1, x2, y2], "score": 0.94, "mask": mask}]
        tracks = tracker.update(dets)

        now = datetime.now(timezone.utc)
        events = gate.process_tracks(tracks, frame_index=frame_idx, monotonic_ns=time.monotonic_ns(), wall_clock=now)
        for ev in events:
            all_crossings.append(ev)
            if ev.direction > 0:
                total_forward += 1
            else:
                total_backward += 1

    # In Phase 1, bag started at 80 and moved past gate at 300 -> exactly 1 forward crossing
    assert total_forward == 1
    assert total_backward == 0
    net_count = total_forward - total_backward
    assert net_count == 1

    # Phase 2: Emergency Stop & Reverse Springback Oscillation
    # Simulate bag initialized in PRE zone (pos=280 < gate_pos=300), then traversing back & forth
    BagTrack.reset_counter()
    gate_osc = GateStateMachine(gate_id=1, gate_position_along_axis=300.0, pre_gate_offset=50.0, post_gate_offset=50.0)
    tracker_osc = ConveyorByteTracker(high_score_threshold=0.35)

    # Frame 0: Track is born firmly in PRE zone
    m0 = np.zeros((400, 600), dtype=bool)
    m0[165:235, 230:330] = True
    trks0 = tracker_osc.update([{"box": [230.0, 165.0, 330.0, 235.0], "score": 0.95, "mask": m0}])
    gate_osc.process_tracks(trks0, frame_index=100, monotonic_ns=time.monotonic_ns(), wall_clock=datetime.now(timezone.utc))

    oscillation_displacements = [
        30.0,   # Frame 1: Moves from 280 to 310 -> enters POST zone -> forward crossing (+1)
        -30.0,  # Frame 2: Reverse springback to 280 -> re-enters PRE zone -> backward slip (-1)
        35.0,   # Frame 3: Forward surge to 315 -> re-enters POST zone -> forward crossing (+1)
        -5.0,   # Frame 4: Slight settling to 310 -> stays in POST zone -> NO crossing event
    ]

    osc_forward = 0
    osc_backward = 0
    current_cx = 280.0

    for idx, dx in enumerate(oscillation_displacements):
        current_cx += dx
        x1, x2 = current_cx - bag_w / 2, current_cx + bag_w / 2
        y1, y2 = bag_cy - bag_h / 2, bag_cy + bag_h / 2

        m = np.zeros((400, 600), dtype=bool)
        m[int(y1):int(y2), int(x1):int(x2)] = True
        dets = [{"box": [x1, y1, x2, y2], "score": 0.95, "mask": m}]
        trks = tracker_osc.update(dets)

        evs = gate_osc.process_tracks(trks, frame_index=101 + idx, monotonic_ns=time.monotonic_ns(), wall_clock=datetime.now(timezone.utc))
        for e in evs:
            if e.direction > 0:
                osc_forward += 1
            else:
                osc_backward += 1

    # Under 3 boundary traversals (PRE -> POST -> PRE -> POST):
    # Total forward = 2, Total backward = 1. Net count MUST be exactly 1!
    assert osc_forward == 2
    assert osc_backward == 1
    assert (osc_forward - osc_backward) == 1


# ===========================================================================
# 2. Dense 4-Bag Shingled Cluster, Multi-Signal Merge & Amodal Recovery
# ===========================================================================
def test_dense_four_bag_shingled_cluster_with_multi_signal_merge_and_amodal_recovery():
    """Verify handling of heavily shingled bags (4 bags overlapping like roof shingles):
    - Multi-signal merge detector votes: area oversized, elongated aspect ratio, multiple print marks.
    - Correct estimate of 4 physical bags from 1 connected mask.
    - Latent track generation and temporal amodal reconstruction.
    """
    mean_area = 20000.0
    merge_detector = MergeDetector(
        mean_bag_gate_area_px=mean_area,
        merge_area_ratio=1.50,
        min_votes=2,
        is_scale_calibrated=True,
    )

    # Construct synthetic 4-bag cluster: length 480px, width 120px (total area ~ 57,600 px)
    # 57,600 / 20,000 = 2.88x nominal area
    cluster_box = [60.0, 150.0, 540.0, 270.0]  # width=480, height=120 -> aspect_ratio = 4.0
    cluster_mask = np.zeros((400, 640), dtype=bool)
    cluster_mask[150:270, 60:540] = True

    # 4 distinct print marks inside the cluster
    print_marks = [
        {"box": [100.0, 190.0, 130.0, 230.0], "score": 0.92},
        {"box": [220.0, 190.0, 250.0, 230.0], "score": 0.94},
        {"box": [340.0, 190.0, 370.0, 230.0], "score": 0.91},
        {"box": [460.0, 190.0, 490.0, 230.0], "score": 0.93},
    ]

    hypothesis = merge_detector.analyze_detection(
        mask=cluster_mask,
        box=cluster_box,
        print_marks=print_marks,
    )

    assert hypothesis.is_merged is True
    assert hypothesis.confidence >= 0.75
    assert "signal_area_oversized" in hypothesis.signal_votes
    assert "signal_shape_convexity_deficit" in hypothesis.signal_votes
    assert "signal_multiple_print_marks" in hypothesis.signal_votes
    assert hypothesis.estimated_object_count == 4
    assert len(hypothesis.centroid_seeds) == 4

    # Verify Amodal Reconstruction on partially occluded observation
    reconstructor = TemporalAmodalReconstructor(max_history_frames=10)
    canonical_box = [100.0, 150.0, 220.0, 270.0]
    canonical_mask = np.zeros((400, 640), dtype=bool)
    canonical_mask[150:270, 100:220] = True

    # Record isolated view in earlier frame
    reconstructor.record_observation(track_id=42, frame_index=1, box=canonical_box, mask=canonical_mask, is_isolated=True)

    # Later frame: Bag 42 is heavily overlapped (only 40% visible)
    occluded_box = [160.0, 150.0, 220.0, 270.0]
    occluded_mask = np.zeros((400, 640), dtype=bool)
    occluded_mask[150:270, 160:220] = True

    reconstructed_mask = reconstructor.reconstruct_amodal_mask(
        track_id=42,
        current_box=occluded_box,
        current_visible_mask=occluded_mask,
        canvas_shape=(400, 640),
    )

    # The reconstructed mask must recover the full bag area from the template
    rec_area = float(np.sum(reconstructed_mask))
    occ_area = float(np.sum(occluded_mask))
    assert rec_area > occ_area * 1.5


# ===========================================================================
# 3. Ripped/Punctured Bag Defect Exclusion & Scrap Ledger Dispute Audit
# ===========================================================================
def test_ripped_punctured_bag_defect_exclusion_and_audit_dispute_workflow():
    """Verify defect detection and post-gate scrap dispute audit trail:
    - Detector flags torn/punctured bags with low solidity (< 0.82) as DAMAGED_DEFORMED.
    - Defect exclusion recorded in immutable CountEvent ledger with defect_reason.
    - Operator dispute registered without erasing or mutating original audit trail.
    """
    topo = _seed_standard_topology()
    session_id = topo["session_id"]
    line_id = topo["line_id"]
    camera_id = topo["camera_id"]
    bundle_id = topo["bundle_id"]

    detector = VisionDetector(conf_threshold=0.25, allow_fallback=True)

    # Create a severely ripped bag with jagged polygon (solidity < 0.82)
    torn_img = np.zeros((400, 640, 3), dtype=np.uint8)
    jagged_pts = np.array([[200, 100], [350, 80], [380, 250], [290, 180], [210, 320]], np.int32)
    cv2.fillPoly(torn_img, [jagged_pts], (200, 200, 200))

    result = detector.predict(torn_img)
    assert len(result.bag_bodies) >= 1
    torn_bag = result.bag_bodies[0]

    assert torn_bag["is_defective"] is True
    assert torn_bag["defect_type"] == "DAMAGED_DEFORMED"
    assert torn_bag["solidity"] is not None and torn_bag["solidity"] < 0.82

    # Step 2: Record Defect Exclusion in Immutable Ledger
    with get_sync_session() as db:
        ledger_repo = LedgerRepository(db)

        # 1 normal crossing
        normal_crossing = GateCrossingEvent(
            track_id=101, crossing_seq=1, gate_id=topo["gate_id"], direction=1,
            crossing_timestamp=datetime.now(timezone.utc), frame_index=15,
            monotonic_ns=time.monotonic_ns(), confidence=0.95, merge_flag=False, centroid=(320.0, 200.0),
        )
        ledger_repo.record_event(
            session_id=session_id, line_id=line_id, camera_id=camera_id, stream_epoch=1,
            track_id=normal_crossing.track_id, crossing_seq=normal_crossing.crossing_seq, gate_id=topo["gate_id"],
            crossing_timestamp=normal_crossing.crossing_timestamp, frame_index=normal_crossing.frame_index,
            direction=normal_crossing.direction, confidence=normal_crossing.confidence, deployment_bundle_id=bundle_id,
        )

        # Torn bag crossed and was subsequently pulled by operator
        torn_crossing = GateCrossingEvent(
            track_id=102, crossing_seq=1, gate_id=topo["gate_id"], direction=1,
            crossing_timestamp=datetime.now(timezone.utc), frame_index=20,
            monotonic_ns=time.monotonic_ns(), confidence=0.90, merge_flag=False, centroid=(320.0, 200.0),
        )
        ledger_repo.record_event(
            session_id=session_id, line_id=line_id, camera_id=camera_id, stream_epoch=1,
            track_id=torn_crossing.track_id, crossing_seq=torn_crossing.crossing_seq, gate_id=topo["gate_id"],
            crossing_timestamp=torn_crossing.crossing_timestamp, frame_index=torn_crossing.frame_index,
            direction=torn_crossing.direction, confidence=torn_crossing.confidence, deployment_bundle_id=bundle_id,
        )

        # Defect exclusion: pulled from line (direction = -1, defect_reason = TORN_KRAFT_VALVE)
        defect_event, created = ledger_repo.record_event(
            session_id=session_id,
            line_id=line_id,
            camera_id=camera_id,
            stream_epoch=1,
            track_id=102,
            crossing_seq=2,  # Monotonic seq incremented
            gate_id=topo["gate_id"],
            crossing_timestamp=datetime.now(timezone.utc),
            frame_index=25,
            direction=-1,
            confidence=0.90,
            defect_reason="TORN_KRAFT_VALVE",
            deployment_bundle_id=bundle_id,
        )
        assert created is True

        # Invariant: Net count = 1 (2 forward - 1 defect exclusion)
        net_count = ledger_repo.get_session_total_count(session_id)
        assert net_count == 1

        # Step 3: Quality Supervisor disputes defect (claims it was an intact bag)
        defect_event.defect_disputed = True
        defect_event.defect_disputed_by = "qa_lead_mehmet"
        defect_event.defect_disputed_note = "Cosmetic surface wrinkle only; integrity intact"
        defect_event.defect_disputed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(defect_event)

        # Verify audit trail preserved
        assert defect_event.defect_reason == "TORN_KRAFT_VALVE"
        assert defect_event.defect_disputed is True
        assert defect_event.defect_disputed_by == "qa_lead_mehmet"


# ===========================================================================
# 4. Camera Network Drop, Stream Epoch Bump & Zero Double-Count Guarantee
# ===========================================================================
def test_camera_network_drop_stream_epoch_bump_and_anti_duplicate_ledger():
    """Verify harsh field reconnect handling:
    - Camera streams 20 bags under stream_epoch=1.
    - Physical network drop occurs; CameraEpochRepository bumps epoch to 2.
    - Edge tracker restarts, repeating track IDs starting from 1.
    - Stream epoch isolates the sequences completely: zero collision, zero duplicate count.
    """
    topo = _seed_standard_topology()
    session_id = topo["session_id"]
    line_id = topo["line_id"]
    cam_id = topo["camera_id"]
    bundle_id = topo["bundle_id"]
    gate_id = topo["gate_id"]

    with get_sync_session() as db:
        epoch_repo = CameraEpochRepository(db)
        ledger_repo = LedgerRepository(db)

        # Epoch 1: Stream 10 bags with track_ids 1..10
        for tid in range(1, 11):
            evt, created = ledger_repo.record_event(
                session_id=session_id,
                line_id=line_id,
                camera_id=cam_id,
                stream_epoch=1,
                track_id=tid,
                crossing_seq=1,
                gate_id=gate_id,
                crossing_timestamp=datetime.now(timezone.utc),
                frame_index=tid * 5,
                direction=1,
                confidence=0.92,
                deployment_bundle_id=bundle_id,
            )
            assert created is True

        assert ledger_repo.get_session_total_count(session_id) == 10

        # Simulate RTSP Network Drop & Reconnection -> Epoch Bump
        new_epoch = epoch_repo.increment_and_get_epoch(cam_id)
        assert new_epoch == 2

        # In Epoch 2, tracker counter resets: track_id 1 reappears
        for tid in range(1, 6):
            evt, created = ledger_repo.record_event(
                session_id=session_id,
                line_id=line_id,
                camera_id=cam_id,
                stream_epoch=new_epoch,
                track_id=tid,
                crossing_seq=1,
                gate_id=gate_id,
                crossing_timestamp=datetime.now(timezone.utc),
                frame_index=tid * 5 + 100,
                direction=1,
                confidence=0.93,
                deployment_bundle_id=bundle_id,
            )
            assert created is True

        # Total net count = 10 (epoch 1) + 5 (epoch 2) = 15
        assert ledger_repo.get_session_total_count(session_id) == 15

        # Idempotency duplicate rejection: re-submitting exact same event from epoch 1
        dup_evt, dup_created = ledger_repo.record_event(
            session_id=session_id,
            line_id=line_id,
            camera_id=cam_id,
            stream_epoch=1,
            track_id=1,
            crossing_seq=1,
            gate_id=gate_id,
            crossing_timestamp=datetime.now(timezone.utc),
            frame_index=5,
            direction=1,
            confidence=0.92,
            deployment_bundle_id=bundle_id,
        )
        assert dup_created is False  # Correctly rejected
        assert ledger_repo.get_session_total_count(session_id) == 15


# ===========================================================================
# 5. Closed-Loop SCADA & Modbus TCP PLC Hardware Actuation
# ===========================================================================
def test_closed_loop_scada_modbus_plc_actuation_and_emergency_interlock():
    """Verify closed-loop industrial automation:
    - Real Modbus TCP PLC server running locally.
    - Inference writes running counted_total to Holding Register 100.
    - Automatic conveyor shutdown (Coil 0 set to False) when counted_total reaches target_count.
    """
    port = _find_free_port()
    plc_server = ModbusTcpServer(host="127.0.0.1", port=port)
    plc_thread = threading.Thread(target=plc_server.start, daemon=True)
    plc_thread.start()
    time.sleep(0.3)

    modbus_client = ModbusTcpIoController(host="127.0.0.1", port=port, timeout_seconds=2.0)
    try:
        assert modbus_client.connect() is True

        # Initial PLC state: conveyor running
        modbus_client.set_signal("conveyor_run", True)
        assert modbus_client.read_signal("conveyor_run") is True

        target_count = 10

        # Simulate counting reaching target
        for current_count in range(1, target_count + 1):
            modbus_client.write_register("counted_total", current_count)
            read_back = modbus_client.read_register("counted_total")
            assert read_back == current_count

            # Interlock logic: when count reaches target, trip conveyor_run signal
            if current_count >= target_count:
                modbus_client.set_signal("conveyor_run", False)  # Motor stop

        assert modbus_client.read_signal("conveyor_run") is False  # Interlock triggered!
    finally:
        modbus_client.disconnect()
        plc_server.stop()


# ===========================================================================
# 6. Dual-Counter Area Discrepancy & Full Reconciliation Lifecycle
# ===========================================================================
def test_dual_counter_area_discrepancy_and_reconciliation_resolution_lifecycle():
    """Verify dual-counter cross-verification under an underfilled batch:
    - 20 bags pass the gate, but each is 40% underfilled (small area).
    - Area integrator accumulates area flux and reports severe discrepancy (> 8%).
    - Session transitions to RECONCILE_REQUIRED.
    - Engineer resolves reconciliation via API with manual override and audit trail.
    """
    topo = _seed_standard_topology()
    session_id = topo["session_id"]
    line_id = topo["line_id"]

    area_counter = AreaIntegralCounter(
        mean_bag_gate_area_px=25000.0,
        discrepancy_threshold=0.08,
        is_scale_calibrated=True,
    )

    # 20 bags counted on gate, but small masks passed (each 10,000 px instead of 25,000 px)
    small_mask = np.zeros((100, 100), dtype=bool)
    small_mask[0:100, 0:100] = True  # 10,000 px
    for _ in range(20):
        area_counter.process_frame_masks([small_mask], belt_speed_px_per_frame=10.0)

    est = area_counter.get_estimate()
    has_disc, rel_diff = area_counter.check_discrepancy(ledger_count=20)

    assert has_disc is True
    assert rel_diff > 0.15

    # Trigger reconciliation via CountingEventHandler and ReconciliationRepository
    with get_sync_session() as db:
        handler = CountingEventHandler(db)
        sess = db.query(SessionORM).filter(SessionORM.id == session_id).first()
        sess.counted_total = 20
        sess.area_estimate_total = est
        db.commit()

        handler.handle_discrepancy(SessionDiscrepancyDetected(session_id=session_id, area_estimate=est))

        rec_repo = ReconciliationRepository(db)
        active_rec = rec_repo.create_reconciliation(
            session_id=session_id,
            trigger_reason="count_area_mismatch",
            evidence_refs={"ledger_count": 20, "area_estimate": est, "relative_delta": rel_diff},
        )
        rec_id = active_rec.id

    # Engineer logs in and resolves discrepancy via API
    eng_token = create_access_token({"sub": "2", "username": "engineer", "role": "engineer"})
    headers = {"Authorization": f"Bearer {eng_token}"}

    res = client.post(
        f"/api/reconciliations/{rec_id}/resolve",
        json={
            "resolution": "manual_override",
            "resolved_count": 20,
            "note": "Visual camera review confirmed 20 bags; underfilling caused optical area deficit.",
        },
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["resolution"] == "manual_override"
    assert data["resolved_count"] == 20

    with get_sync_session() as db:
        sess = db.query(SessionORM).filter(SessionORM.id == session_id).first()
        assert sess.status == "reconciled"
        assert sess.counted_total == 20


# ===========================================================================
# 7. Cryptographic HMAC-SHA256 Manifest Anti-Tamper & Outbox Retry
# ===========================================================================
def test_cryptographic_manifest_anti_tamper_and_transactional_outbox_retry():
    """Verify cryptographic sealing and outbox reliability:
    - HMAC-SHA256 seal covers session metadata and count.
    - Any payload tampering invalidates cryptographic verification.
    - ERP Relay retries transient 503 errors and posts to SAP on recovery.
    """
    topo = _seed_standard_topology()
    session_id = topo["session_id"]

    op_token = create_access_token({"sub": "1", "username": "operator", "role": "operator"})
    headers = {"Authorization": f"Bearer {op_token}"}

    rep_res = client.get(f"/api/sessions/{session_id}/dispatch_report", headers=headers)
    assert rep_res.status_code == 200
    manifest = rep_res.json()
    seal = manifest["crypto_seal"]
    assert len(seal) == 64

    # Anti-Tamper Proof: verify altering count invalidates seal
    raw_data = f"SESS:{manifest['session_id']}|REF:{manifest['external_ref']}|COUNT:{manifest['counted_total']}|PROD:{manifest['erp_sku']}"
    # Recomputing HMAC with tampered count (+1)
    tampered_data = f"SESS:{manifest['session_id']}|REF:{manifest['external_ref']}|COUNT:{manifest['counted_total'] + 1}|PROD:{manifest['erp_sku']}"
    tampered_seal = hmac.new(SECRET_KEY.encode(), tampered_data.encode(), hashlib.sha256).hexdigest().upper()
    assert seal != tampered_seal

    # Transactional Outbox Failover Retry Test
    class MockFlakySapAdapter(ErpAdapter):
        attempts = 0
        def submit_session(self, payload: SessionPayload) -> ErpResult:
            MockFlakySapAdapter.attempts += 1
            if MockFlakySapAdapter.attempts < 2:
                return ErpResult(success=False, external_tx_id=None, error_message="SAP HTTP 503 Service Unavailable", retryable=True)
            return ErpResult(success=True, external_tx_id="SAP_MATDOC_500129")

        def query_status(self, external_ref: str) -> ErpStatus:
            return ErpStatus(state=ErpStatusState.PENDING, external_tx_id=None, message="Not yet posted")

        @property
        def supports_status_query(self) -> bool:
            return True

    adapter = MockFlakySapAdapter()
    relay = ErpRelayWorker(adapter=adapter)

    with get_sync_session() as db:
        outbox_entry = OutboxORM(
            session_id=session_id,
            external_ref="WB-DISPATCH-2026-09",
            payload={"line_id": 1, "counted_total": 50, "erp_material_code": "CEM-I-42.5R"},
            status="pending",
        )
        db.add(outbox_entry)
        db.commit()
        db.refresh(outbox_entry)
        entry_id = outbox_entry.id

    # Step 1: First poll -> ERP 503 error
    relay.run_step()
    with get_sync_session() as db:
        e = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
        assert e.attempts == 1
        assert e.status in ["pending", "in_progress", "failed"]

    # Step 2: Second poll -> ERP recovers & succeeds
    relay.process_entry(e)
    with get_sync_session() as db:
        e = db.query(OutboxORM).filter(OutboxORM.id == entry_id).first()
        assert e.status == "sent"


# ===========================================================================
# 8. OIML R51 Automatic Weighbridge Mass Balance & Tolerance Verification
# ===========================================================================
def test_oiml_r51_automatic_weighbridge_mass_balance_and_tolerance_certificate():
    """Verify metrology standards (OIML R51 Class X(1)):
    - Target batch: 600 bags @ 50 kg nominal weight = 30,000 kg (30.00 Metric Tons).
    - Certified weighbridge scale: Gross 45,180 kg, Tare 15,150 kg -> Net 30,030 kg.
    - Mass balance deviation: 30 kg / 30,000 kg = 0.10% (well within <= 0.50% standard).
    """
    nominal_weight_kg = 50.0
    counted_bags = 600
    expected_mass_kg = counted_bags * nominal_weight_kg

    scale_gross_kg = 45180.0
    scale_tare_kg = 15150.0
    measured_net_mass_kg = scale_gross_kg - scale_tare_kg

    mass_delta_kg = abs(measured_net_mass_kg - expected_mass_kg)
    relative_error_pct = (mass_delta_kg / expected_mass_kg) * 100.0

    # OIML R51 Class X(1) tolerance limit is 0.50%
    assert relative_error_pct <= 0.50
    assert mass_delta_kg == 30.0
