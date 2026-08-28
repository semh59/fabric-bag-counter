"""Real live video for non-counting cameras (vehicle_watchdog / auxiliary roles).

A line's `counting`-role camera is driven end-to-end by
`stream_renderer.LiveStreamRenderer` (real detection, tracking, gate
crossings, ledger writes) -- that pipeline is untouched here.

Every other camera on a line (damage-inspection view, pallet station,
weighbridge, yard overview, ...) previously had a real DB row and a real
`POST /cameras/{id}/test` connectivity check, but no way to actually watch
it live: the UI could either show one hardcoded "camera wall" of fake tiles,
or nothing. This module gives each of those cameras its own real
`cv2.VideoCapture`-backed MJPEG feed with a real, measured FPS -- no
detection overlay (that is specific to the counting pipeline), just an
honest live picture and an honest "no signal" placeholder when it isn't
connected.
"""

from __future__ import annotations

import logging
import time
from typing import Generator

import cv2
import numpy as np

from packages.cs_core.camera_source import resolve_camera_source
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.models_orm import CameraORM

logger = logging.getLogger(__name__)

_MAX_DIM = 960  # cap streamed frame size; a monitoring feed needs no more


class CameraFeed:
    """Real video capture + live status for a single (non-counting) camera."""

    def __init__(self, camera_id: int) -> None:
        self.camera_id = camera_id
        self.video_cap: cv2.VideoCapture | None = None
        self.connected = False
        self.last_error: str | None = None
        self._last_frame_time: float | None = None
        self._fps_ema: float | None = None
        self._label = f"KAM-{camera_id}"

    def set_label(self, label: str) -> None:
        self._label = label

    def connect(self, source: str | int) -> tuple[bool, str]:
        if self.video_cap is not None:
            try:
                self.video_cap.release()
            except Exception:
                pass
            self.video_cap = None
        self.connected = False

        if source == "" or source is None:
            self.last_error = "Source not configured."
            return False, self.last_error

        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            self.video_cap = cap
            self.connected = True
            self.last_error = None
            return True, f"Connected: {source}"

        self.last_error = f"Connection failed: {source}"
        return False, self.last_error

    def _placeholder_frame(self, text: str) -> np.ndarray:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (18, 20, 24)
        cv2.putText(frame, text, (24, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 96, 104), 1, cv2.LINE_AA)
        cv2.putText(frame, self._label, (24, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 146, 154), 1, cv2.LINE_AA)
        return frame

    def read_jpeg(self) -> bytes:
        """Read one real frame (or a real "no signal" placeholder) and encode it."""
        frame: np.ndarray | None = None
        if self.video_cap is not None and self.video_cap.isOpened():
            ret, raw = self.video_cap.read()
            if not ret:
                # A finite video file (as opposed to a live RTSP/USB stream)
                # legitimately runs out of frames -- loop it, matching
                # LiveStreamRenderer's identical handling for the counting
                # camera. For a real stream this is a harmless no-op retry:
                # a genuinely dead RTSP source still fails the re-read below.
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, raw = self.video_cap.read()
            if ret:
                h, w = raw.shape[:2]
                if max(h, w) > _MAX_DIM:
                    scale = _MAX_DIM / max(h, w)
                    raw = cv2.resize(raw, (int(w * scale), int(h * scale)))
                frame = raw
                self._tick_fps()
            else:
                self.connected = False
                self.last_error = "Could not read frame from stream."

        if frame is None:
            frame = self._placeholder_frame(
                "NO SIGNAL" if self.last_error is None else self.last_error
            )
        else:
            self._draw_hud(frame)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    def _tick_fps(self) -> None:
        now = time.time()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps_ema = inst_fps if self._fps_ema is None else (0.9 * self._fps_ema + 0.1 * inst_fps)
            self._last_frame_time = now

    def _draw_hud(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 24), (10, 14, 22), -1)
        fps_txt = f"{self._fps_ema:.1f} FPS" if self._fps_ema is not None else "—"
        cv2.putText(frame, f"● LIVE  {self._label}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 240, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, fps_txt, (w - 90, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 220), 1, cv2.LINE_AA)


_camera_feeds: dict[int, CameraFeed] = {}


def get_or_create_camera_feed(camera_id: int) -> CameraFeed:
    """Return this camera's live feed, connecting it from its DB row on first use."""
    feed = _camera_feeds.get(camera_id)
    if feed is not None:
        return feed

    feed = CameraFeed(camera_id)
    with get_sync_session() as db:
        cam = db.query(CameraORM).filter(CameraORM.id == camera_id).first()
        if cam is None:
            feed.last_error = "Camera not found."
        else:
            feed.set_label(f"CAM-{cam.id}")
            if cam.enabled:
                source = resolve_camera_source(cam.source_driver, cam.source_config)
                ok, msg = feed.connect(source)
                if not ok:
                    logger.info("Camera %s not connected on first use: %s", camera_id, msg)
            else:
                feed.last_error = "Camera disabled."
    _camera_feeds[camera_id] = feed
    return feed


def reconnect_camera_feed(camera_id: int, source: str | int) -> tuple[bool, str]:
    """Force this camera's live feed to (re)connect to a new source."""
    feed = get_or_create_camera_feed(camera_id)
    return feed.connect(source)


def get_camera_stream_generator(camera_id: int) -> Generator[bytes, None, None]:
    """Continuous MJPEG generator for one non-counting camera."""
    feed = get_or_create_camera_feed(camera_id)
    while True:
        frame_bytes = feed.read_jpeg()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.04)  # ~25 FPS -- these are monitoring feeds, not the detection pipeline
