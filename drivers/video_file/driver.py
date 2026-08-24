"""File and simulated video stream driver (§4.4)."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
import numpy as np
from PIL import Image
from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.video_source import VideoSource


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
        except Exception:
            # Fallback simulated frame generator
            self._is_connected = True

    def read(self) -> Frame | None:
        if not self._is_connected:
            return None

        img_shape = (640, 640, 3)
        if self.cap is not None:
            ret, img = self.cap.read()
            if not ret or img is None:
                if self.config.get("loop", False):
                    self.cap.set(0, 0)  # Rewind to start
                    ret, img = self.cap.read()
                    if not ret or img is None:
                        return None
                else:
                    self._is_connected = False
                    return None
            img_shape = img.shape

        self.frame_counter += 1
        mono_ns = time.monotonic_ns()
        wall = datetime.utcnow()

        return Frame(
            camera_id=self.config.get("camera_id", 1),
            stream_epoch=self.epoch,
            frame_index=self.frame_counter,
            monotonic_ns=mono_ns,
            wall_clock=wall,
            shm_name=f"shm_file_cam_{self.config.get('camera_id', 1)}_slot_{self.frame_counter % 8}",
            shape=img_shape,
            dtype="uint8",
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
