"""Ingest Worker: Per-camera video capture and shared memory publisher (§4.2, §4.5, §5.2)."""

from __future__ import annotations

import logging
import time
from typing import Any
from packages.cs_core.interfaces.video_source import VideoSource
from packages.cs_storage.db import get_sync_session
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
            except Exception:
                self.current_epoch += 1

            self.driver.open(self.config, epoch=self.current_epoch)
            return False

        frame = self.driver.read()
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
