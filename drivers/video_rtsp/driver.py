"""RTSP video capture driver (§4.4)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.frame_transport import FrameTransport
from packages.cs_core.interfaces.video_source import VideoSource

logger = logging.getLogger(__name__)


class RtspVideoSource:
    """Decodes RTSP H.264/H.265 video streams using PyAV / OpenCV."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.epoch: int = 0
        self.frame_counter: int = 0
        self._is_connected: bool = False
        self.cap: Any = None

    def open(self, config: dict[str, Any], epoch: int) -> None:
        self.config = config
        self.epoch = epoch
        self.frame_counter = 0
        self.rtsp_url = config.get("url", "")

        try:
            import cv2
            # cv2.VideoCapture(url) with no timeout blocks on the OS's default
            # TCP connect timeout when the camera is unreachable -- commonly
            # tens of seconds, not the ~1s the ingest worker's reconnect loop
            # (IngestWorker.run_step) deliberately sleeps between attempts.
            # Verified directly: against an unreachable address, the plain
            # constructor call hangs far longer than these explicit
            # open/read timeouts (passed via the params-array constructor,
            # the only overload that applies them *before* connecting --
            # setting them via .set() after construction is too late for
            # OPEN_TIMEOUT specifically, since the constructor itself already
            # blocked trying to connect).
            open_timeout_ms = int(config.get("open_timeout_ms", 5000))
            read_timeout_ms = int(config.get("read_timeout_ms", 5000))
            self.cap = cv2.VideoCapture(
                self.rtsp_url, cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms],
            )
            self._is_connected = bool(self.cap.isOpened())
            if not self._is_connected:
                logger.error(f"[RtspVideoSource] Failed to open RTSP stream: {self.rtsp_url!r}")
        except Exception as e:
            logger.error(f"[RtspVideoSource] Exception while opening RTSP stream {self.rtsp_url!r}: {e}")
            self.cap = None
            self._is_connected = False

    def read(self, transport: FrameTransport) -> Frame | None:
        if not self._is_connected or self.cap is None:
            return None

        ret, img = self.cap.read()
        if not ret or img is None:
            self._is_connected = False
            return None

        self.frame_counter += 1
        mono_ns = time.monotonic_ns()
        wall = datetime.now(timezone.utc)
        camera_id = self.config.get("camera_id", 1)
        shm_name = f"shm_cam_{camera_id}_slot_{self.frame_counter % 8}"

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
