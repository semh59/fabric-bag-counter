"""Unit tests for LedgerRepository idempotency, persistent stream epoch, and total count derivation (§5.2, §5.5)."""

from datetime import datetime, timezone
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import CameraORM, LineORM, ProductProfileORM, SessionORM, SiteORM
from packages.cs_storage.repositories.camera_epoch_repo import CameraEpochRepository
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.session_repo import SessionRepository


def setup_test_db():
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Test Site")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()

        cam = CameraORM(line_id=line.id, node_id=1, source_driver="rtsp")
        db.add(cam)
        db.commit()

        profile = ProductProfileORM(site_id=site.id, name="Standard 50kg Bag", nominal_dims_mm={})
        db.add(profile)
        db.commit()

        return site.id, line.id, cam.id, profile.id


def test_persistent_camera_epoch():
    _, _, cam_id, _ = setup_test_db()
    with get_sync_session() as db:
        repo = CameraEpochRepository(db)
        epoch1 = repo.increment_and_get_epoch(cam_id)
        assert epoch1 >= 1

        # Simulate camera disconnect and reconnect
        epoch2 = repo.increment_and_get_epoch(cam_id)
        assert epoch2 == epoch1 + 1


def test_ledger_idempotency_and_net_count_derivation():
    _, line_id, cam_id, profile_id = setup_test_db()
    with get_sync_session() as db:
        session_repo = SessionRepository(db)
        ledger_repo = LedgerRepository(db)

        sess = session_repo.create_session(line_id=line_id, product_profile_id=profile_id)
        t_now = datetime.now(timezone.utc)

        # 1. Record first crossing (track 10, seq 1, dir +1)
        ev1, created1 = ledger_repo.record_event(
            session_id=sess.id,
            line_id=line_id,
            camera_id=cam_id,
            stream_epoch=1,
            track_id=10,
            crossing_seq=1,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=100,
            direction=1,
        )
        assert created1 is True

        # 2. Try recording the EXACT same event again (Idempotency test)
        ev1_dup, created1_dup = ledger_repo.record_event(
            session_id=sess.id,
            line_id=line_id,
            camera_id=cam_id,
            stream_epoch=1,
            track_id=10,
            crossing_seq=1,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=100,
            direction=1,
        )
        assert created1_dup is False
        assert ev1_dup.event_id == ev1.event_id

        # 3. Record backward slip (track 10, seq 2, dir -1)
        ledger_repo.record_event(
            session_id=sess.id,
            line_id=line_id,
            camera_id=cam_id,
            stream_epoch=1,
            track_id=10,
            crossing_seq=2,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=120,
            direction=-1,
        )

        # 4. Record second forward crossing (track 10, seq 3, dir +1)
        ledger_repo.record_event(
            session_id=sess.id,
            line_id=line_id,
            camera_id=cam_id,
            stream_epoch=1,
            track_id=10,
            crossing_seq=3,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=140,
            direction=1,
        )

        # 5. Record separate second bag (track 11, seq 1, dir +1)
        ledger_repo.record_event(
            session_id=sess.id,
            line_id=line_id,
            camera_id=cam_id,
            stream_epoch=1,
            track_id=11,
            crossing_seq=1,
            gate_id=1,
            crossing_timestamp=t_now,
            frame_index=150,
            direction=1,
        )

        # Derive net count: +1 - 1 + 1 + 1 = 2
        total_derived = ledger_repo.get_session_total_count(sess.id)
        assert total_derived == 2

        # Close session and verify session.counted_total reflects derived ledger total (§5.5)
        closed_sess = session_repo.close_session(sess.id)
        assert closed_sess.counted_total == 2
        assert closed_sess.status == "closed"
