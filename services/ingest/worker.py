"""Ingest Worker: Per-camera video capture and shared memory publisher (§4.2, §4.5, §5.2)."""

from __future__ import annotations

import logging
import os
import time
from importlib.metadata import entry_points
from typing import Any
from packages.cs_core.interfaces.video_source import VideoSource
from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import CameraORM
from packages.cs_storage.repositories.camera_epoch_repo import CameraEpochRepository
from packages.cs_core.transport import SharedMemoryTransport

logger = logging.getLogger(__name__)


class IngestWorker:
    """Worker process capturing frames from a single camera source."""

    def __init__(
        self,
        camera_id: int,
        source_driver: VideoSource,
        source_config: dict[str, Any],
        transport: SharedMemoryTransport,
    ) -> None:
        self.camera_id = camera_id
        self.driver = source_driver
        self.config = source_config
        self.transport = transport
        self.is_running = False
        self.current_epoch = 0

    def start(self) -> None:
        """Start capture loop with persistent stream epoch increment."""
        self.is_running = True

        # Atomically increment persistent camera epoch (§5.2)
        try:
            with get_sync_session() as db:
                epoch_repo = CameraEpochRepository(db)
                self.current_epoch = epoch_repo.increment_and_get_epoch(self.camera_id)
            logger.info(f"[Ingest Cam {self.camera_id}] Starting with persistent epoch {self.current_epoch}")
        except Exception as e:
            logger.error(f"[Ingest Cam {self.camera_id}] Failed to get persistent epoch: {e}")
            self.current_epoch = 1

        self.driver.open(self.config, epoch=self.current_epoch)

    def run_step(self) -> bool:
        """Execute one capture cycle. Returns True if frame captured, False if idle/reconnecting."""
        if not self.is_running:
            return False

        if not self.driver.is_connected:
            # Attempt reconnect with new epoch (§5.2)
            time.sleep(1.0)
            try:
                with get_sync_session() as db:
                    epoch_repo = CameraEpochRepository(db)
                    self.current_epoch = epoch_repo.increment_and_get_epoch(self.camera_id)
            except Exception as e:
                logger.exception(f"[Ingest Cam {self.camera_id}] Failed to get persistent epoch on reconnect: {e}")
                self.current_epoch += 1

            self.driver.open(self.config, epoch=self.current_epoch)
            return False

        frame = self.driver.read(self.transport)
        if frame is None:
            return False

        # Publish metadata into transport ring
        res = self.transport.publish(frame)
        if res.consecutive_drops > 0:
            logger.warning(
                f"[Ingest Cam {self.camera_id}] Frame drop! Consecutive: {res.consecutive_drops}, Total: {res.dropped_frames}"
            )

        return True

    def stop(self) -> None:
        """Stop worker and release camera source."""
        self.is_running = False
        self.driver.close()
        logger.info(f"[Ingest Cam {self.camera_id}] Ingest worker stopped.")

    def start_loop(self) -> None:
        """Run continuous capture, matching the other workers' start_loop() shape."""
        self.start()
        while self.is_running:
            if not self.run_step():
                time.sleep(0.01)


def resolve_video_source_driver(source_driver_name: str) -> VideoSource:
    """Look up a registered "cs.video_source" driver by name and instantiate it.

    Real use of the plugin entry points already declared in pyproject.toml
    (rtsp/file) -- previously nothing in the codebase actually resolved them
    dynamically, so a camera's source_driver string had no real path to a
    running ingest process.
    """
    matches = [ep for ep in entry_points(group="cs.video_source") if ep.name == source_driver_name]
    if not matches:
        available = sorted(ep.name for ep in entry_points(group="cs.video_source"))
        raise ValueError(
            f"No registered cs.video_source driver named '{source_driver_name}' (available: {available})"
        )
    driver_cls = matches[0].load()
    return driver_cls()


def main() -> None:
    """Start one ingest process for exactly one camera (real multi-camera capture is
    one OS process per camera, not one process fanning out across many cameras).

    Which camera is selected via --camera-id -- this is how
    SupervisorManager._spawn_worker() already invokes this module
    (`python -m services.ingest.worker --camera-id <id>`) to dynamically
    spawn/respawn one process per row in the camera table. CAMERA_ID as an
    environment variable is also accepted, for direct docker-compose/manual
    invocation where passing a CLI flag is less convenient; --camera-id
    wins if both are given.
    """
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="Ingest worker: one process per camera.")
    parser.add_argument("--camera-id", type=int, default=None)
    args, _unknown = parser.parse_known_args()

    camera_id = args.camera_id
    if camera_id is None:
        camera_id_str = os.environ.get("CAMERA_ID")
        if not camera_id_str:
            raise RuntimeError(
                "--camera-id (or CAMERA_ID environment variable) is required -- "
                "one ingest worker process serves exactly one camera."
            )
        camera_id = int(camera_id_str)

    init_db_sync()
    with get_sync_session() as db:
        cam = db.query(CameraORM).filter(CameraORM.id == camera_id).first()
        if cam is None:
            raise RuntimeError(f"No camera found with id={camera_id}")
        source_driver_name = cam.source_driver
        source_config = dict(cam.source_config or {})

    driver = resolve_video_source_driver(source_driver_name)
    transport = SharedMemoryTransport()
    worker = IngestWorker(camera_id=camera_id, source_driver=driver, source_config=source_config, transport=transport)
    try:
        worker.start_loop()
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
