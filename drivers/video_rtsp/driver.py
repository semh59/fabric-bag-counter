"""RTSP video capture driver (§4.4)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
import numpy as np
from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.video_source import VideoSource


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
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self._is_connected = bool(self.cap.isOpened())
        except Exception:
            self._is_connected = False

    def read(self) -> Frame | None:
        if not self._is_connected or self.cap is None:
            return None

        ret, img = self.cap.read()
        if not ret or img is None:
            self._is_connected = False
            return None

        self.frame_counter += 1
        mono_ns = time.monotonic_ns()
        wall = datetime.now(timezone.utc)

        return Frame(
            camera_id=self.config.get("camera_id", 1),
            stream_epoch=self.epoch,
            frame_index=self.frame_counter,
            monotonic_ns=mono_ns,
            wall_clock=wall,
            shm_name=f"shm_cam_{self.config.get('camera_id', 1)}_slot_{self.frame_counter % 8}",
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
