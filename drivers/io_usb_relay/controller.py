"""USB Relay board controller (§4.4)."""

from __future__ import annotations

import logging
from typing import Any
from packages.cs_core.interfaces.io_controller import IoController

logger = logging.getLogger(__name__)


class UsbRelayIoController:
    """Controls physical USB 8-channel relay board for warning lights and conveyor interlock."""

    def __init__(self, port: str = "COM3", baudrate: int = 9600) -> None:
        self.port = port
        self.baudrate = baudrate
        self.channel_states: dict[str, bool] = {
            "green_light": False,
            "warning_yellow": False,
            "error_red": False,
            "conveyor_run": True,
        }

    def set_signal(self, name: str, value: bool) -> None:
        self.channel_states[name] = value
        logger.info(f"[UsbRelay on {self.port}] Channel '{name}' set to {value}")

    def read_signal(self, name: str) -> bool:
        return self.channel_states.get(name, False)
