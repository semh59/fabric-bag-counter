"""Tests for the ingest worker's process entry point (§4.2, §4.5).

cs-ingest was declared as a console script (pyproject.toml) but
services/ingest/worker.py had no main() at all -- the entry point was
broken. These exercise the real dynamic driver resolution (via the actual
installed package's entry_points, not a mock) and the real CAMERA_ID
validation/DB lookup that main() now does.
"""

import os

import pytest

from drivers.video_file.driver import FileVideoSource
from drivers.video_rtsp.driver import RtspVideoSource
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import CameraORM, LineORM, NodeORM, SiteORM
from services.ingest.worker import main, resolve_video_source_driver


def test_resolve_video_source_driver_real_entry_points():
    file_driver = resolve_video_source_driver("file")
    assert isinstance(file_driver, FileVideoSource)

    rtsp_driver = resolve_video_source_driver("rtsp")
    assert isinstance(rtsp_driver, RtspVideoSource)


def test_resolve_video_source_driver_unknown_name_raises_with_available_list():
    with pytest.raises(ValueError, match="available"):
        resolve_video_source_driver("not_a_real_driver")


def test_main_requires_camera_id_env_var(monkeypatch):
    monkeypatch.delenv("CAMERA_ID", raising=False)
    with pytest.raises(RuntimeError, match="CAMERA_ID"):
        main()


def test_main_raises_for_nonexistent_camera(monkeypatch):
    init_db_sync()
    monkeypatch.setenv("CAMERA_ID", "999999")
    with pytest.raises(RuntimeError, match="No camera found"):
        main()


def test_main_accepts_camera_id_cli_flag_matching_real_supervisor_invocation(monkeypatch):
    # SupervisorManager._spawn_worker() literally runs:
    #   python -m services.ingest.worker --camera-id <id>
    # -- this is the real invocation this module must support, not just the
    # CAMERA_ID env var.
    init_db_sync()
    monkeypatch.delenv("CAMERA_ID", raising=False)
    with pytest.raises(RuntimeError, match="No camera found"):
        with monkeypatch.context() as m:
            m.setattr("sys.argv", ["services.ingest.worker", "--camera-id", "999999"])
            main()


def test_main_resolves_real_camera_and_starts_worker(monkeypatch):
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Ingest Test Site")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()
        node = NodeORM(site_id=site.id, hostname="edge-1")
        db.add(node)
        db.commit()
        cam = CameraORM(line_id=line.id, node_id=node.id, source_driver="file", source_config={"path": "/no/such/file.mp4"}, role="counting")
        db.add(cam)
        db.commit()
        cam_id = cam.id

    monkeypatch.setenv("CAMERA_ID", str(cam_id))

    # main() loops forever (start_loop); patch IngestWorker.start_loop so
    # this test only exercises the real setup (env var -> DB lookup ->
    # driver resolution -> worker construction) without actually looping.
    calls = {}

    def fake_start_loop(self):
        calls["camera_id"] = self.camera_id
        calls["driver_type"] = type(self.driver).__name__

    import services.ingest.worker as ingest_module
    monkeypatch.setattr(ingest_module.IngestWorker, "start_loop", fake_start_loop)

    main()

    assert calls["camera_id"] == cam_id
    assert calls["driver_type"] == "FileVideoSource"
