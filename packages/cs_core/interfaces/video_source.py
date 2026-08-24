"""VideoSource protocol for camera stream decoders."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from packages.cs_core.frame import Frame


@runtime_checkable
class VideoSource(Protocol):
    """Protocol for reading video frames from RTSP, local files, or test streams."""

    def open(self, config: dict[str, Any], epoch: int) -> None:
        """Initialize and open the video stream for the given epoch."""
        ...

    def read(self) -> Frame | None:
        """Read the next available frame, writing image bytes into shared memory."""
        ...

    def close(self) -> None:
        """Close connection and release native stream resources."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return True if connection is active and healthy."""
        ...
