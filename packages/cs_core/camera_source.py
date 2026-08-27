"""Resolve a CameraORM's driver/source_config into an OpenCV-openable source.

Shared by the connection-test endpoint and the per-camera live feed so the
same camera row always resolves to the same source the same way -- there was
previously a second, drifting copy of this logic inline in the test-camera
route.
"""

from __future__ import annotations

from typing import Any


def resolve_camera_source(source_driver: str, source_config: dict[str, Any] | None) -> str | int:
    """Return an OpenCV-openable source (URL string or device index int).

    Returns "" when nothing usable is configured yet -- callers should treat
    that as "not configured" rather than attempting to open it.
    """
    cfg = source_config or {}
    if source_driver == "usb":
        return int(cfg.get("device_index", 0))
    if source_driver in ("rtsp", "http", "file"):
        return cfg.get("rtsp_url") or cfg.get("url") or cfg.get("path") or ""
    return cfg.get("rtsp_url") or cfg.get("url") or cfg.get("device_index", "")
