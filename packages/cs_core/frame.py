"""Frame data class representing a single captured frame metadata with shared memory reference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Frame:
    """Immutable frame metadata container.
    
    Image bytes are stored in shared memory for zero-copy IPC.
    Frame index and monotonic timestamp are used for all counting and tracking logic.
    """
    camera_id: int
    stream_epoch: int
    frame_index: int         # Monotonically increasing within epoch, 0-indexed
    monotonic_ns: int        # Monotonic capture timestamp from ingest in nanoseconds
    wall_clock: datetime     # Wall clock time for display / audit
    shm_name: str            # Shared memory block identifier
    shape: tuple[int, int, int]  # (height, width, channels)
    dtype: str = "uint8"
