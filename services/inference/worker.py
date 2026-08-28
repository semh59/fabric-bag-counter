"""Inference Worker: Single GPU batch engine, ledger recorder, and degraded state monitor (§4.2, §4.5, §5.5, §11 M5)."""

from __future__ import annotations

import logging
import os

from packages.cs_core.frame import Frame
from packages.cs_core.transport import SharedMemoryTransport
from packages.cs_counting.engine import CountingEngine
from packages.cs_counting.event_handler import CountingEventHandler
from packages.cs_counting.events import (
    GateCrossingRecorded,
    SessionDegraded,
    SessionDiscrepancyDetected,
)
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.repositories.config_repo import ConfigRepository
from packages.cs_storage.repositories.session_repo import SessionRepository

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
        # Tracks which config_version was last applied to self.engine via
        # CountingEngine.configure(), so run_step() only re-resolves and
        # re-applies the effective payload when the active bundle's config
        # actually changed, not on every single batch.
        self._applied_config_version_id: int | None = None

    def run_step(self) -> int:
        """Consume pending frames and process through pipeline. Returns count of frames processed."""
        frames: list[Frame] = self.transport.consume(timeout_ms=self.batch_wait_ms)
        if not frames:
            return 0

        # Check for active session
        with get_sync_session() as db:
            session_repo = SessionRepository(db)
            event_handler = CountingEventHandler(db)
            config_repo = ConfigRepository(db)

            active_session = session_repo.get_active_session(self.line_id)
            active_bundle = config_repo.get_active_bundle(self.line_id)
            if active_bundle:
                self.active_bundle_id = active_bundle.id
                # Apply the engineer's real config (confidence_threshold,
                # merge_area_ratio, roi_polygon, etc. -- see CountingEngine.
                # configure()) only when it actually changed since the last
                # batch, not on every single one.
                if active_bundle.config_version_id != self._applied_config_version_id:
                    try:
                        payload = config_repo.get_effective_config_payload(active_bundle.config_version)
                        self.engine.configure(payload)
                        self._applied_config_version_id = active_bundle.config_version_id
                    except Exception:
                        logger.exception(
                            f"[Inference] Failed to apply config_version_id={active_bundle.config_version_id} "
                            f"for line_id={self.line_id}"
                        )

            stats = self.transport.get_stats()
            consecutive_drops = stats.get("consecutive_drops", {})

            for frame in frames:
                # 1. Check frame drop degradation policy (§4.5)
                cam_drops = consecutive_drops.get(frame.camera_id, 0)
                if cam_drops >= self.max_consecutive_drops and active_session:
                    logger.warning(
                        f"[Inference] Excessive consecutive frame drops ({cam_drops}) on camera {frame.camera_id}. Marking session {active_session.id} as DEGRADED."
                    )
                    event_handler.handle_degraded(SessionDegraded(
                        session_id=active_session.id, camera_id=frame.camera_id, consecutive_drops=cam_drops,
                    ))

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

                # 4. Record gate crossings into immutable Ledger (§5.5), update
                # session totals, and handle discrepancy -- via
                # CountingEventHandler (packages/cs_counting/event_handler.py),
                # the one shared implementation of this logic also used by
                # LiveStreamRenderer's two frame paths and simulate_bag_crossing.
                # This closes a real gap this call site previously had on its
                # own: it recorded ledger events but never derived/persisted
                # session.counted_total from them (unlike the other three
                # call sites, which all did) -- counted_total would have
                # stayed stale through the whole session on this pipeline.
                if active_session:
                    # A genuine data error here (e.g. self.active_bundle_id
                    # pointing at a deployment_bundle row that no longer
                    # exists) now raises instead of being silently swallowed
                    # as a fake idempotency hit (see ledger_repo.py's
                    # _is_idempotency_duplicate) -- correct, but this loop
                    # processes every subsequent frame/camera on this line
                    # too, so one bad frame must not take the whole worker
                    # process down. Caught and logged here, matching the
                    # same defensive pattern LiveStreamRenderer's two frame
                    # paths already use around this exact call.
                    try:
                        applied = event_handler.handle_frame_output(
                            output,
                            line_id=self.line_id,
                            camera_id=frame.camera_id,
                            session_id=active_session.id,
                            stream_epoch=frame.stream_epoch,
                            deployment_bundle_id=self.active_bundle_id,
                        )
                    except Exception:
                        logger.exception(
                            f"[Inference] Failed to record frame output for session_id={active_session.id}, "
                            f"camera_id={frame.camera_id}, frame_index={frame.frame_index}"
                        )
                        applied = []
                    for evt in applied:
                        if isinstance(evt, GateCrossingRecorded):
                            c = evt.crossing
                            logger.info(
                                f"[Inference] Cross event: Track {c.track_id} (Seq {c.crossing_seq}) Dir {c.direction:+} on Cam {frame.camera_id}"
                            )
                        elif isinstance(evt, SessionDiscrepancyDetected):
                            logger.warning(
                                f"[Inference] Discrepancy detected between ledger count ({output.running_net_count}) and area estimate ({output.area_estimate:.1f}). Triggering reconciliation."
                            )

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
