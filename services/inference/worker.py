"""Inference Worker: Single GPU batch engine, ledger recorder, and degraded state monitor (§4.2, §4.5, §5.5, §11 M5)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from packages.cs_core.frame import Frame
from packages.cs_counting.engine import CountingEngine
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.ledger_repo import LedgerRepository
from packages.cs_storage.repositories.session_repo import SessionRepository
from packages.cs_core.transport import SharedMemoryTransport

logger = logging.getLogger(__name__)


class InferenceWorker:
    """Consumes frame batches from shared memory, runs counting pipeline, and appends to ledger."""

    def __init__(
        self,
        transport: SharedMemoryTransport,
        line_id: int = 1,
        batch_wait_ms: int = 30,
        max_consecutive_drops: int = 3,
        engine: CountingEngine | None = None,
    ) -> None:
        self.transport = transport
        self.line_id = line_id
        self.batch_wait_ms = batch_wait_ms
        self.max_consecutive_drops = max_consecutive_drops
        self.engine = engine or CountingEngine()
        self.is_running = False
        self.active_bundle_id = 1

    def run_step(self) -> int:
        """Consume pending frames and process through pipeline. Returns count of frames processed."""
        frames: list[Frame] = self.transport.consume(timeout_ms=self.batch_wait_ms)
        if not frames:
            return 0

        # Check for active session
        with get_sync_session() as db:
            session_repo = SessionRepository(db)
            ledger_repo = LedgerRepository(db)
            config_repo = ConfigRepository(db)

            active_session = session_repo.get_active_session(self.line_id)
            active_bundle = config_repo.get_active_bundle(self.line_id)
            if active_bundle:
                self.active_bundle_id = active_bundle.id

            stats = self.transport.get_stats()
            consecutive_drops = stats.get("consecutive_drops", {})

            for frame in frames:
                # 1. Check frame drop degradation policy (§4.5)
                cam_drops = consecutive_drops.get(frame.camera_id, 0)
                if cam_drops >= self.max_consecutive_drops and active_session:
                    logger.warning(
                        f"[Inference] Excessive consecutive frame drops ({cam_drops}) on camera {frame.camera_id}. Marking session {active_session.id} as DEGRADED."
                    )
                    session_repo.mark_degraded(active_session.id)

                # 2. Retrieve image array from shared memory
                img_data = self.transport.get_image_data(frame.shm_name)
                if img_data is None:
                    # No pixel data available for this frame (never written, or the
                    # ring already evicted/released its shared memory block before we
                    # got to it). Never substitute a fabricated black frame here --
                    # that would silently run detection on fake data. Skip this frame
                    # but still release its slot and keep processing the rest.
                    logger.warning(
                        f"[Inference] No shared-memory image data for camera {frame.camera_id} "
                        f"frame_index={frame.frame_index} shm_name={frame.shm_name!r}. Skipping frame."
                    )
                    self.transport.release(frame)
                    continue

                # 3. Process frame through CountingEngine
                output = self.engine.process_frame(
                    image=img_data,
                    frame_index=frame.frame_index,
                    monotonic_ns=frame.monotonic_ns,
                    wall_clock=frame.wall_clock,
                )

                # 4. Record gate crossings into immutable Ledger (§5.5)
                if active_session:
                    for event in output.gate_crossings:
                        event_record, created = ledger_repo.record_event(
                            session_id=active_session.id,
                            line_id=self.line_id,
                            camera_id=frame.camera_id,
                            stream_epoch=frame.stream_epoch,
                            track_id=event.track_id,
                            crossing_seq=event.crossing_seq,
                            gate_id=event.gate_id,
                            crossing_timestamp=event.crossing_timestamp,
                            frame_index=event.frame_index,
                            direction=event.direction,
                            confidence=event.confidence,
                            merge_flag=event.merge_flag,
                            deployment_bundle_id=self.active_bundle_id,
                        )
                        if created:
                            logger.info(
                                f"[Inference] Cross event: Track {event.track_id} (Seq {event.crossing_seq}) Dir {event.direction:+} on Cam {frame.camera_id}"
                            )

                    # Update running area estimate on session
                    session_repo.update_area_estimate(active_session.id, output.area_estimate)

                    # Handle discrepancy flag trigger (§6.9, §5.7)
                    if output.discrepancy_flag:
                        logger.warning(
                            f"[Inference] Discrepancy detected between ledger count ({output.running_net_count}) and area estimate ({output.area_estimate:.1f}). Triggering reconciliation."
                        )
                        session_repo.flag_discrepancy(active_session.id, output.area_estimate)

                # 5. Release shared memory slot
                self.transport.release(frame)

        return len(frames)

    def start_loop(self) -> None:
        """Run continuous inference loop."""
        self.is_running = True
        logger.info(f"[Inference] Worker started on line {self.line_id}")
        while self.is_running:
            self.run_step()

    def stop(self) -> None:
        """Stop worker."""
        self.is_running = False
        logger.info("[Inference] Worker stopped.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    line_id = int(os.environ.get("LINE_ID", "1"))
    transport = SharedMemoryTransport()
    worker = InferenceWorker(transport=transport, line_id=line_id)
    try:
        worker.start_loop()
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
