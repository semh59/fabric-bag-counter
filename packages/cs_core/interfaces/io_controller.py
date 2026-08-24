"""IoController protocol for physical signals and relay hardware."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IoController(Protocol):
    """Protocol for interacting with external discrete I/O (relay lights, PLC trigger)."""

    def set_signal(self, name: str, value: bool) -> None:
        """Set output discrete signal (e.g. 'green_light', 'warning_horn', 'stop_belt')."""
        ...

    def read_signal(self, name: str) -> bool:
        """Read digital input status (e.g. 'photoeye_1', 'e_stop')."""
        ...
