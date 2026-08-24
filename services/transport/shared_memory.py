"""SharedMemoryTransport: Zero-copy ring buffer IPC frame transport (§4.5)."""

from __future__ import annotations

import collections
import threading
import time
from typing import Any
import numpy as np
from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.frame_transport import FrameTransport, PublishResult


class SharedMemoryTransport:
    """Ring buffer frame transport across ingest workers and inference engine."""

    def __init__(self, ring_slots: int = 8) -> None:
        self.ring_slots = ring_slots
        self._lock = threading.Lock()
        # Per-camera frame ring queues
        self._camera_queues: dict[int, collections.deque[Frame]] = {}
        # Per-camera shared image buffers: shm_name -> np.ndarray
        self._shm_buffers: dict[str, np.ndarray] = {}
        # Dropped frame counters
        self.dropped_frame_counts: dict[int, int] = collections.defaultdict(int)
        self.consecutive_drops: dict[int, int] = collections.defaultdict(int)

    def write_image_data(self, shm_name: str, image: np.ndarray) -> None:
        """Store image array in simulated/shared memory buffer."""
        with self._lock:
            self._shm_buffers[shm_name] = image.copy()

    def get_image_data(self, shm_name: str) -> np.ndarray | None:
        """Read image array from shared memory buffer."""
        with self._lock:
            return self._shm_buffers.get(shm_name)

    def publish(self, frame: Frame) -> PublishResult:
        """Publish frame metadata into camera ring buffer.
        
        If ring buffer is full, drops the oldest frame (§4.5 non-blocking ingest).
        """
        with self._lock:
            queue = self._camera_queues.setdefault(frame.camera_id, collections.deque(maxlen=self.ring_slots))
            was_dropped = False

            if len(queue) >= self.ring_slots:
                # Ring full: discard oldest frame
                dropped = queue.popleft()
                # Free previous shm buffer
                self._shm_buffers.pop(dropped.shm_name, None)
                self.dropped_frame_counts[frame.camera_id] += 1
                self.consecutive_drops[frame.camera_id] += 1
                was_dropped = True
            else:
                self.consecutive_drops[frame.camera_id] = 0

            queue.append(frame)

            return PublishResult(
                success=True,
                dropped_frames=self.dropped_frame_counts[frame.camera_id],
                consecutive_drops=self.consecutive_drops[frame.camera_id],
            )

    def consume(self, timeout_ms: int = 30) -> list[Frame]:
        """Consume available frames across all camera ring queues within timeout."""
        start_t = time.monotonic()
        consumed: list[Frame] = []

        while (time.monotonic() - start_t) * 1000.0 < timeout_ms:
            with self._lock:
                for cid, queue in self._camera_queues.items():
                    if queue:
                        consumed.append(queue.popleft())
            if consumed:
                break
            time.sleep(0.002)

        return consumed

    def release(self, frame: Frame) -> None:
        """Release shared memory block after inference completion."""
        with self._lock:
            self._shm_buffers.pop(frame.shm_name, None)

    def get_stats(self) -> dict[str, Any]:
        """Return frame drop statistics across cameras."""
        with self._lock:
            return {
                "dropped_frame_counts": dict(self.dropped_frame_counts),
                "consecutive_drops": dict(self.consecutive_drops),
            }
