"""VideoSource protocol for camera stream decoders."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.frame_transport import FrameTransport


@runtime_checkable
class VideoSource(Protocol):
    """Protocol for reading video frames from RTSP, local files, or test streams."""

    def open(self, config: dict[str, Any], epoch: int) -> None:
        """Initialize and open the video stream for the given epoch."""
        ...

    def read(self, transport: FrameTransport) -> Frame | None:
        """Read the next available frame.

        Decodes one frame from the underlying source and writes its pixel
        bytes into shared memory via `transport.write_image_data(shm_name,
        image)` before returning the corresponding `Frame` metadata (whose
        `shm_name` matches what was just written). Returns None if no frame
        is currently available (e.g. not connected, stream ended without
        loop, or a read/decode failure) -- callers must treat None as "no
        frame this step", never fabricate one.
        """
        ...

    def close(self) -> None:
        """Close connection and release native stream resources."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return True if connection is active and healthy."""
        ...
