"""Comprehensive demo seeder for out-of-the-box system demo (§9.2, §9.4)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session

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
from packages.cs_storage.repositories.user_repo import UserRepository


def seed_demo_data(db: Session, force_reset: bool = False) -> None:
    """Seed comprehensive demo records if database is fresh."""
    user_repo = UserRepository(db)
    user_repo.seed_default_users()

    existing_site = db.query(SiteORM).first()
    if existing_site and not force_reset:
        return

    if force_reset:
        for model in [CountEventORM, ReconciliationORM, OutboxORM, JobORM,
                      DeploymentBundleORM, ModelVersionORM, TrainingRunORM, DatasetVersionORM,
                      LineCalibrationORM, ConfigVersionORM, ProductProfileORM, SessionORM,
                      CameraEpochORM, CameraORM, GateORM, LineORM, NodeORM, SiteORM]:
            try:
                db.query(model).delete()
            except Exception:
                db.rollback()
        db.commit()

    # 1. Site
    site = SiteORM(
        name="Gebze Cimento & Yapi Kimyasallari Fabrikasi",
        timezone="Europe/Istanbul",
        locale="tr_TR",
        created_at=datetime.now(timezone.utc),
    )
    db.add(site)
    db.commit()
    db.refresh(site)

    # 2. Node (Edge GPU Worker)
    node = NodeORM(
        site_id=site.id,
        hostname="edge-gpu-worker-01",
        gpu_info={
            "device": "NVIDIA GeForce RTX 4090",
            "vram_mb": 24576,
            "cuda": "12.4",
            "driver_version": "550.54.14",
            "utilization_pct": 34,
            "temp_c": 52,
        },
        status="online",
        last_heartbeat=datetime.now(timezone.utc),
    )
    db.add(node)
    db.commit()
    db.refresh(node)

    # 3. Line
    line = LineORM(
        site_id=site.id,
        name="Hat-1 (Ana Dolum & Yukleme Hatti)",
        status="counting",
        maintenance_window={"planned_window": "02:00-04:00", "status": "nominal"},
    )
    db.add(line)
    db.commit()
    db.refresh(line)

    # 4. Gate
    gate = GateORM(
        line_id=line.id,
        name="GATE_01 (Orta Kapi)",
        order_index=0,
    )
    db.add(gate)
    db.commit()
    db.refresh(gate)

    # 5. Camera
    camera = CameraORM(
        line_id=line.id,
        node_id=node.id,
        source_driver="rtsp",
        source_config={
            "url": "rtsp://192.168.1.120:554/live/stream1",
            "fps": 25,
            "resolution": "1920x1080",
            "codec": "h264",
            "lens": "8mm_low_distortion",
        },
        role="counting",
        enabled=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    # Camera epoch
    epoch = CameraEpochORM(camera_id=camera.id, current_epoch=4)
    db.add(epoch)

    # 6. Product Profiles
    p1 = ProductProfileORM(
        site_id=site.id,
        name="50kg Polipropilen Cimento Cuvali",
        nominal_weight_g=50000.0,
        nominal_dims_mm={"width": 500, "length": 800, "height": 150, "mean_area_px": 42500},
        erp_material_code="MAT-CIM-50KG",
        template_images=["/assets/templates/cement_50kg_poly.png"],
    )
    p2 = ProductProfileORM(
        site_id=site.id,
        name="25kg Kraft Alci Torbasi",
        nominal_weight_g=25000.0,
        nominal_dims_mm={"width": 400, "length": 600, "height": 120, "mean_area_px": 28000},
        erp_material_code="MAT-ALC-25KG",
        template_images=["/assets/templates/kraft_25kg.png"],
    )
    p3 = ProductProfileORM(
        site_id=site.id,
        name="40kg Yapi Kimyasali Cuvali",
        nominal_weight_g=40000.0,
        nominal_dims_mm={"width": 450, "length": 750, "height": 140, "mean_area_px": 36000},
        erp_material_code="MAT-YAP-40KG",
        template_images=["/assets/templates/chem_40kg.png"],
    )
    db.add_all([p1, p2, p3])
    db.commit()
    db.refresh(p1)
    db.refresh(p2)
    db.refresh(p3)

    # 7. Config Version
    cfg = ConfigVersionORM(
        line_id=line.id,
        payload_schema_version=2,
        payload={
            "belt_speed_mps": 0.75,
            "optical_flow_window": 15,
            "gate_y_ratio": 0.50,
            "gate_hysteresis_px": 35,
            "amodal_seg_threshold": 0.72,
            "min_bag_area_px": 12000,
            "max_consecutive_drops": 3,
            "discrepancy_tolerance_pct": 5.0,
            "roi_polygon": [[100, 100], [1820, 100], [1820, 980], [100, 980]],
        },
        note="Standart fabrika konveyor bant konfigurasyonu",
        created_by="admin",
        created_at=datetime.now(timezone.utc),
    )
    db.add(cfg)
    db.commit()
    db.refresh(cfg)

    # 8. Calibration
    calib = LineCalibrationORM(
        line_id=line.id,
        stage="motion",
        belt_speed_px_per_frame=12.4,
        belt_direction_vector=[1.0, 0.0],
        px_per_mm=1.002,
        mean_bag_gate_area_px=42500.0,
        bag_area_stddev_px=1200.0,
        is_active=True,
        created_by="admin",
        created_at=datetime.now(timezone.utc),
    )
    db.add(calib)
    db.commit()
    db.refresh(calib)

    # 9. Dataset Version
    ds = DatasetVersionORM(
        site_id=site.id,
        name="DS-2026-Gebze-Synthetic-Mix-v2.1",
        manifest_hash="sha256:d41d8cd98f00b204e9800998ecf8427e",
        frame_count=5400,
        synthetic_count=1380,
        split_spec={"train": 0.8, "val": 0.15, "test": 0.05},
        annotation_guide_version="2.0",
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    # 10. Training Run
    tr = TrainingRunORM(
        dataset_version_id=ds.id,
        run_kind="fine_tune",
        status="completed",
        hyperparams={"batch_size": 16, "learning_rate": 1e-4, "optimizer": "AdamW"},
        started_at=datetime.now(timezone.utc) - timedelta(hours=12),
        finished_at=datetime.now(timezone.utc) - timedelta(hours=10),
        metrics={"loss": 0.038, "mAP_50": 0.968, "epochs": 50},
    )
    db.add(tr)
    db.commit()
    db.refresh(tr)

    # 11. Model Version
    mv = ModelVersionORM(
        training_run_id=tr.id,
        onnx_hash="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        onnx_path="/app/data/models/rf_detr_v2_1.onnx",
        stage="active",
        eval_scores={
            "mAP_50": 0.968,
            "mAP_50_95": 0.884,
            "f1_score": 0.972,
            "fps": 52.4,
            "latency_ms": 19.1,
        },
        created_at=datetime.now(timezone.utc),
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)

    # 12. Deployment Bundle
    bundle = DeploymentBundleORM(
        line_id=line.id,
        model_version_id=mv.id,
        config_version_id=cfg.id,
        calibration_id=calib.id,
        git_commit="c29f8a4",
        activated_at=datetime.now(timezone.utc),
        activated_by="admin",
    )
    db.add(bundle)
    db.commit()
    db.refresh(bundle)

    # 13. Past Closed Sessions
    past_session1 = SessionORM(
        line_id=line.id,
        product_profile_id=p1.id,
        external_ref="IRS-2026-0840",
        target_count=200,
        status="closed",
        opened_at=datetime.now(timezone.utc) - timedelta(hours=6),
        closed_at=datetime.now(timezone.utc) - timedelta(hours=4),
        counted_total=200,
        area_estimate_total=199.6,
        discrepancy_flag=False,
    )
    past_session2 = SessionORM(
        line_id=line.id,
        product_profile_id=p2.id,
        external_ref="IRS-2026-0841",
        target_count=150,
        status="reconcile_required",
        opened_at=datetime.now(timezone.utc) - timedelta(hours=4),
        closed_at=datetime.now(timezone.utc) - timedelta(hours=2),
        counted_total=148,
        area_estimate_total=132.4,
        discrepancy_flag=True,
    )
    db.add_all([past_session1, past_session2])
    db.commit()
    db.refresh(past_session1)
    db.refresh(past_session2)

    # 14. Active Counting Session
    active_session = SessionORM(
        line_id=line.id,
        product_profile_id=p1.id,
        external_ref="IRS-2026-0842",
        target_count=200,
        status="counting",
        opened_at=datetime.now(timezone.utc) - timedelta(minutes=45),
        counted_total=142,
        area_estimate_total=141.8,
        discrepancy_flag=False,
    )
    db.add(active_session)
    db.commit()
    db.refresh(active_session)

    # 15. Seed 142 Ledger Count Events for active session
    events = []
    base_time = active_session.opened_at
    for i in range(1, 143):
        ev = CountEventORM(
            event_id=f"EVT-DEMO-2026-{i:04d}",
            session_id=active_session.id,
            line_id=line.id,
            camera_id=camera.id,
            stream_epoch=4,
            track_id=1000 + i,
            crossing_seq=1,
            gate_id=gate.id,
            crossing_timestamp=base_time + timedelta(seconds=i * 18),
            frame_index=i * 450,
            direction=1,
            confidence=0.98 + (i % 5) * 0.003,
            merge_flag=(i % 25 == 0),
            deployment_bundle_id=bundle.id,
            evidence_ref=f"/evidence/frames/active_sess_{active_session.id}_trk_{1000+i}.jpg",
            created_at=base_time + timedelta(seconds=i * 18),
        )
        events.append(ev)
    db.add_all(events)
    db.commit()

    # 16. Reconciliation Record for past_session2
    rec = ReconciliationORM(
        session_id=past_session2.id,
        trigger_reason="count_area_mismatch",
        opened_at=datetime.now(timezone.utc) - timedelta(hours=2),
        note="Ledger sayimi: 148 adet, Alan tahmini: 132.4 adet (Fark: %10.5 > %5.0 tolerans). Olasi cuval ust uste binme.",
    )
    db.add(rec)

    # 17. Outbox entries
    ob1 = OutboxORM(
        session_id=past_session1.id,
        payload={
            "session_id": past_session1.id,
            "external_ref": past_session1.external_ref,
            "counted_total": 200,
            "product_code": "MAT-CIM-50KG",
        },
        status="sent",
        attempts=1,
        external_ref=past_session1.external_ref,
        created_at=datetime.now(timezone.utc) - timedelta(hours=4),
    )
    ob2 = OutboxORM(
        session_id=past_session2.id,
        payload={
            "session_id": past_session2.id,
            "external_ref": past_session2.external_ref,
            "counted_total": 148,
            "product_code": "MAT-ALC-25KG",
        },
        status="pending",
        attempts=0,
        external_ref=past_session2.external_ref,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    db.add_all([ob1, ob2])

    # 18. Sample GPU Jobs
    j1 = JobORM(
        kind="train",
        payload={"dataset_id": ds.id, "epochs": 50},
        status="completed",
        priority=5,
        created_at=datetime.now(timezone.utc) - timedelta(hours=12),
        started_at=datetime.now(timezone.utc) - timedelta(hours=12),
        finished_at=datetime.now(timezone.utc) - timedelta(hours=10),
    )
    j2 = JobORM(
        kind="calibrate_motion",
        payload={"line_id": line.id, "algorithm": "lucas_kanade"},
        status="completed",
        priority=3,
        created_at=datetime.now(timezone.utc) - timedelta(hours=8),
        started_at=datetime.now(timezone.utc) - timedelta(hours=8),
        finished_at=datetime.now(timezone.utc) - timedelta(hours=8),
    )
    db.add_all([j1, j2])
    db.commit()
