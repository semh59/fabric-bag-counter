"""USB Relay board controller (§4.4).

Targets Numato Lab-compatible USB relay modules (numato.com) -- a real,
commercially available 8/16-channel relay board that shows up as a
standard virtual COM port and accepts a small ASCII command set:

    relay on <N>\\r     turn relay N on
    relay off <N>\\r    turn relay N off
    relay read <N>\\r   device replies with the relay's current state

A different relay board model needs a different driver (register a new
"cs.io_controller" entry point rather than special-casing it here) --
this class does not attempt to be a universal relay driver.
"""

from __future__ import annotations

import logging
from typing import Any

import serial

from packages.cs_core.interfaces.io_controller import IoController

logger = logging.getLogger(__name__)

# Which physical relay channel each named signal maps to. Override via the
# channel_map constructor argument if a given board is wired differently.
DEFAULT_CHANNEL_MAP: dict[str, int] = {
    "green_light": 0,
    "warning_yellow": 1,
    "error_red": 2,
    "conveyor_run": 3,
}


class UsbRelayIoController:
    """Real serial-port driver for a Numato-compatible USB relay board.

    Controls physical relay channels (warning lights, conveyor run/stop
    interlock) over a real serial connection -- opening the port is a real
    I/O operation that fails loudly (raises) if the hardware isn't present
    or the port is wrong, rather than silently pretending to succeed.
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 19200,
        timeout: float = 1.0,
        channel_map: dict[str, int] | None = None,
        connection: Any = None,
    ) -> None:
        """Open the relay board's serial port (or accept an already-open one).

        `connection` lets tests (or callers that already manage the serial
        connection lifecycle) inject a substitute -- anything exposing
        pyserial's write/read/reset_input_buffer/flush/close -- instead of
        this constructor opening a real OS-level COM port.
        """
        self.port = port
        self.baudrate = baudrate
        self.channel_map = dict(channel_map) if channel_map else dict(DEFAULT_CHANNEL_MAP)
        self._conn = connection if connection is not None else serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._last_known: dict[str, bool] = {}
        logger.info(f"[UsbRelay] Connected to relay board on {self.port} @ {self.baudrate} baud")

    def _relay_index(self, name: str) -> int:
        if name not in self.channel_map:
            raise KeyError(f"Unknown relay channel '{name}' -- not in channel_map {sorted(self.channel_map)}")
        return self.channel_map[name]

    def set_signal(self, name: str, value: bool) -> None:
        idx = self._relay_index(name)
        cmd = f"relay {'on' if value else 'off'} {idx}\r"
        self._conn.write(cmd.encode("ascii"))
        self._conn.flush()
        self._last_known[name] = value
        logger.info(f"[UsbRelay {self.port}] '{name}' (relay {idx}) -> {'ON' if value else 'OFF'}")

    def read_signal(self, name: str) -> bool:
        """Read a relay's real current state from the board.

        Numato boards echo the command and reply with the channel state;
        exact framing/prompt bytes vary slightly by firmware revision, so
        this checks for the literal "on"/"off" token in the response rather
        than parsing a fixed byte offset -- verify against the real board
        during commissioning if a firmware's reply format differs.
        """
        idx = self._relay_index(name)
        self._conn.reset_input_buffer()
        self._conn.write(f"relay read {idx}\r".encode("ascii"))
        self._conn.flush()
        raw = self._conn.read(64)
        text = raw.decode("ascii", errors="ignore").lower()
        if "off" in text:
            state = False
        elif "on" in text:
            state = True
        else:
            # No recognizable reply (board silent/disconnected) -- report
            # the last commanded state rather than fabricating a fresh one.
            state = self._last_known.get(name, False)
        self._last_known[name] = state
        return state

    def close(self) -> None:
        self._conn.close()
