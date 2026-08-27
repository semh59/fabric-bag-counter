"""Supervisor Manager: Multi-camera worker process lifecycle manager (§4.2, §11 M5)."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from packages.cs_core.transport import SharedMemoryTransport
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.models_orm import CameraORM

logger = logging.getLogger(__name__)

# Respawn backoff: exponential, capped, per-camera. A camera whose ingest
# process keeps crashing immediately (bad source, permissions, etc.) would
# otherwise be respawned every sync_camera_workers() tick (every 5s from
# main()'s loop) forever, burning CPU/log volume for no benefit.
_RESPAWN_BACKOFF_BASE_SECONDS = 2.0
_RESPAWN_BACKOFF_MAX_SECONDS = 60.0


class SupervisorManager:
    """Manages per-camera ingest workers dynamically based on the database camera table."""

    def __init__(self, transport: SharedMemoryTransport | None = None) -> None:
        self.transport = transport or SharedMemoryTransport()
        self.worker_processes: dict[int, subprocess.Popen[Any]] = {}
        self.is_running = False
        # Per-camera consecutive-failed-spawn count and earliest next spawn time.
        self._respawn_attempts: dict[int, int] = {}
        self._next_spawn_at: dict[int, datetime] = {}

    def _drain_pipe_to_logger(self, pipe: Any, camera_id: int, stream_name: str) -> None:
        """Continuously forward a subprocess pipe's output into our logger.

        stdout/stderr=subprocess.PIPE without ever reading them can deadlock
        the child process once its OS pipe buffer fills (the child blocks on
        write()), and eventually starves this supervisor's own polling. Drain
        both pipes on background threads for the lifetime of the process.
        """
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode(errors="replace").rstrip()
                if line:
                    logger.info(f"[Ingest Cam {camera_id}][{stream_name}] {line}")
        except (ValueError, OSError):
            pass  # pipe closed as the process was torn down; nothing to log
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _spawn_worker(self, cam_id: int) -> subprocess.Popen[Any]:
        """Spawn an ingest worker process and start pipe-draining threads for it."""
        cmd = [sys.executable, "-m", "services.ingest.worker", "--camera-id", str(cam_id)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(
            target=self._drain_pipe_to_logger, args=(proc.stdout, cam_id, "stdout"), daemon=True
        ).start()
        threading.Thread(
            target=self._drain_pipe_to_logger, args=(proc.stderr, cam_id, "stderr"), daemon=True
        ).start()
        return proc

    def sync_camera_workers(self) -> None:
        """Query camera table and synchronize active ingest worker processes."""
        try:
            with get_sync_session() as db:
                cameras = db.query(CameraORM).filter(CameraORM.enabled == True).all()  # noqa: E712
                active_ids = {c.id for c in cameras}
                now = datetime.now(timezone.utc)

                for cam in cameras:
                    proc = self.worker_processes.get(cam.id)

                    if proc is not None and proc.poll() is None:
                        # Still running: it survived to this tick, so clear any
                        # backoff state accumulated from earlier crashes.
                        self._respawn_attempts.pop(cam.id, None)
                        self._next_spawn_at.pop(cam.id, None)
                        continue

                    if proc is not None:
                        # Exited since the last tick (crashed or was killed externally).
                        del self.worker_processes[cam.id]
                        self._respawn_attempts[cam.id] = self._respawn_attempts.get(cam.id, 0) + 1

                    next_spawn_at = self._next_spawn_at.get(cam.id)
                    if next_spawn_at is not None and now < next_spawn_at:
                        continue  # still backing off from a recent crash

                    new_proc = self._spawn_worker(cam.id)
                    self.worker_processes[cam.id] = new_proc
                    attempts = self._respawn_attempts.get(cam.id, 0)
                    backoff = min(_RESPAWN_BACKOFF_MAX_SECONDS, _RESPAWN_BACKOFF_BASE_SECONDS * (2 ** attempts))
                    self._next_spawn_at[cam.id] = now + timedelta(seconds=backoff)
                    logger.info(
                        f"[Supervisor] Spawned ingest process (PID {new_proc.pid}) for camera {cam.id} "
                        f"(respawn attempt {attempts}, backoff if it exits again: {backoff:.0f}s)"
                    )

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
                        self._respawn_attempts.pop(cid, None)
                        self._next_spawn_at.pop(cid, None)
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
