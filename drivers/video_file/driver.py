"""File and simulated video stream driver (§4.4)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.frame_transport import FrameTransport
from packages.cs_core.interfaces.video_source import VideoSource

logger = logging.getLogger(__name__)


class FileVideoSource:
    """Reads frames from a local video file or directory of images."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.epoch: int = 0
        self.frame_counter: int = 0
        self._is_connected: bool = False
        self.file_path: str = ""
        self.cap: Any = None

    def open(self, config: dict[str, Any], epoch: int) -> None:
        self.config = config
        self.epoch = epoch
        self.frame_counter = 0
        self.file_path = config.get("file_path", "")

        try:
            import cv2
            self.cap = cv2.VideoCapture(self.file_path)
            self._is_connected = bool(self.cap.isOpened())
            if not self._is_connected:
                logger.error(f"[FileVideoSource] Failed to open video source: {self.file_path!r}")
        except Exception as e:
            logger.error(f"[FileVideoSource] Exception while opening video source {self.file_path!r}: {e}")
            self.cap = None
            self._is_connected = False

    def read(self, transport: FrameTransport) -> Frame | None:
        if not self._is_connected or self.cap is None:
            return None

        import cv2

        ret, img = self.cap.read()
        if not ret or img is None:
            if self.config.get("loop", False):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start
                ret, img = self.cap.read()
                if not ret or img is None:
                    self._is_connected = False
                    return None
            else:
                self._is_connected = False
                return None

        self.frame_counter += 1
        mono_ns = time.monotonic_ns()
        wall = datetime.now(timezone.utc)
        camera_id = self.config.get("camera_id", 1)
        shm_name = f"shm_file_cam_{camera_id}_slot_{self.frame_counter % 8}"

        # Write decoded pixel data into shared memory before publishing metadata.
        transport.write_image_data(shm_name, img)

        return Frame(
            camera_id=camera_id,
            stream_epoch=self.epoch,
            frame_index=self.frame_counter,
            monotonic_ns=mono_ns,
            wall_clock=wall,
            shm_name=shm_name,
            shape=img.shape,
            dtype=str(img.dtype),
        )

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected
