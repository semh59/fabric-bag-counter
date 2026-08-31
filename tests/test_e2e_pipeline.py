"""End-to-end integration tests using synthetic scenes and CountingEngine (§11 M3, M5)."""

from datetime import UTC, datetime, timedelta

import numpy as np

from packages.cs_counting.engine import CountingEngine
from packages.cs_counting.event_handler import CountingEventHandler
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import CameraORM, LineORM, ProductProfileORM, SiteORM
from packages.cs_storage.repositories.calibration_repo import CalibrationRepository
from packages.cs_storage.repositories.session_repo import SessionRepository


def test_e2e_counting_pipeline():
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Factory 1")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line A")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=1, source_driver="rtsp")
        db.add(cam)
        db.commit()

        prof = ProductProfileORM(site_id=site.id, name="Standard Bag", nominal_dims_mm={})
        db.add(prof)
        db.commit()

        calib_repo = CalibrationRepository(db)
        calib_repo.create_motion_calibration(line_id=line.id, belt_speed_px_per_frame=15.0, belt_direction_vector=[1.0, 0.0])
        calib_repo.create_scale_calibration(line_id=line.id, px_per_mm=0.8, mean_bag_gate_area_px=15000.0, bag_area_stddev_px=500.0)

        session_repo = SessionRepository(db)
        sess = session_repo.create_session(line_id=line.id, product_profile_id=prof.id, target_count=5)
        sess_id = sess.id
        line_id = line.id
        cam_id = cam.id

    engine = CountingEngine()
    engine.belt_motion.update_from_calibration(speed_px=15.0, direction=[1.0, 0.0])
    engine.gate_state_machine.update_geometry(
        axis_origin=(0.0, 0.0),
        axis_vector=(1.0, 0.0),
        gate_pos=300.0,
        pre_offset=50.0,
        post_offset=50.0,
    )
    engine.area_counter.update_calibration(mean_bag_area_px=15000.0, is_active=True)
    engine.merge_detector.update_calibration(mean_bag_area_px=15000.0, is_active=True)

    # Simulate 30 frames of bag moving across conveyor from x=150 to x=500
    base_time = datetime.now(UTC)
    for f_idx in range(30):
        t_frame = base_time + timedelta(milliseconds=40 * f_idx)
        mono_ns = int(f_idx * 40 * 1e6)

        cx = 150.0 + (f_idx * 12.0)  # Bag moves along x axis
        cy = 320.0
        mask = np.zeros((640, 640), dtype=bool)
        x1, y1 = int(cx - 50), int(cy - 60)
        x2, y2 = int(cx + 50), int(cy + 60)
        mask[max(0, y1):min(640, y2), max(0, x1):min(640, x2)] = True

        # Mock single bag detection in frame
        engine.detector.predict = lambda img, _box=[x1, y1, x2, y2], _m=mask: type("DetectionMock", (), {
            "bag_bodies": [{"box": _box, "score": 0.95, "mask": _m}],
            "print_marks": [],
        })()

        out = engine.process_frame(
            image=np.zeros((640, 640, 3), dtype=np.uint8),
            frame_index=f_idx,
            monotonic_ns=mono_ns,
            wall_clock=t_frame,
        )

        with get_sync_session() as db:
            CountingEventHandler(db).handle_frame_output(
                out, line_id=line_id, camera_id=cam_id, session_id=sess_id, stream_epoch=1,
            )

    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        closed_sess = session_repo.close_session(sess_id)
        assert closed_sess.counted_total == 1
        assert closed_sess.status == "closed"


