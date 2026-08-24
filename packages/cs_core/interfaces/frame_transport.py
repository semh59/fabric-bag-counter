"""FrameTransport protocol for shared-memory / IPC frame routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from packages.cs_core.frame import Frame


@dataclass
class PublishResult:
    success: bool
    dropped_frames: int = 0
    consecutive_drops: int = 0


@runtime_checkable
class FrameTransport(Protocol):
    """Protocol for zero-copy IPC transport between ingest workers and inference engine."""

    def publish(self, frame: Frame) -> PublishResult:
        """Publish frame metadata into the queue/ring buffer."""
        ...

    def consume(self, timeout_ms: int) -> list[Frame]:
        """Consume pending frames available across active camera slots."""
        ...

    def release(self, frame: Frame) -> None:
        """Mark frame shared memory slot as released and available for reuse."""
        ...
