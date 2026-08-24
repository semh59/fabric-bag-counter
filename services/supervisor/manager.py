"""Supervisor Manager: Multi-camera worker process lifecycle manager (§4.2, §11 M5)."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from typing import Any
from packages.cs_core.transport import SharedMemoryTransport
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.models_orm import CameraORM

logger = logging.getLogger(__name__)


class SupervisorManager:
    """Manages per-camera ingest workers dynamically based on the database camera table."""

    def __init__(self, transport: SharedMemoryTransport | None = None) -> None:
        self.transport = transport or SharedMemoryTransport()
        self.worker_processes: dict[int, subprocess.Popen[Any]] = {}
        self.is_running = False

    def sync_camera_workers(self) -> None:
        """Query camera table and synchronize active ingest worker processes."""
        try:
            with get_sync_session() as db:
                cameras = db.query(CameraORM).filter(CameraORM.enabled == True).all()  # noqa: E712
                active_ids = {c.id for c in cameras}

                # Start missing worker processes
                for cam in cameras:
                    if cam.id not in self.worker_processes or self.worker_processes[cam.id].poll() is not None:
                        # Spawn independent ingest process
                        cmd = [sys.executable, "-m", "services.ingest.worker", "--camera-id", str(cam.id)]
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        self.worker_processes[cam.id] = proc
                        logger.info(f"[Supervisor] Spawned ingest process (PID {proc.pid}) for camera {cam.id}")

                # Terminate removed worker processes
                for cid in list(self.worker_processes.keys()):
                    if cid not in active_ids:
                        proc = self.worker_processes.pop(cid)
                        proc.terminate()
                        try:
                            proc.wait(timeout=3.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        logger.info(f"[Supervisor] Terminated ingest process for camera {cid}")
        except Exception as e:
            logger.error(f"[Supervisor] Failed to sync camera workers: {e}")

    def start(self) -> None:
        """Start supervisor loop."""
        self.is_running = True
        self.sync_camera_workers()
        logger.info("[Supervisor] Supervisor manager started.")

    def stop(self) -> None:
        """Stop all workers."""
        self.is_running = False
        for cid, proc in self.worker_processes.items():
            proc.terminate()
        self.worker_processes.clear()
        logger.info("[Supervisor] Supervisor manager stopped.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    supervisor = SupervisorManager()
    supervisor.start()
    try:
        while True:
            time.sleep(5)
            supervisor.sync_camera_workers()
    except KeyboardInterrupt:
        supervisor.stop()


if __name__ == "__main__":
    main()
