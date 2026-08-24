"""No-op discrete I/O controller implementation (§4.4)."""

from __future__ import annotations

import logging
from packages.cs_core.interfaces.io_controller import IoController

logger = logging.getLogger(__name__)


class NoopIoController:
    """Simulated in-memory I/O controller for testing and headless environments."""

    def __init__(self) -> None:
        self.signals: dict[str, bool] = {}

    def set_signal(self, name: str, value: bool) -> None:
        self.signals[name] = value
        logger.info(f"[NoopIO] Signal '{name}' -> {value}")

    def read_signal(self, name: str) -> bool:
        return self.signals.get(name, False)
