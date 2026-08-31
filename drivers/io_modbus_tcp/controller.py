"""Industrial Modbus TCP PLC I/O Controller Driver (§4.4).

Production-grade Modbus TCP driver implementing MBAP framing, thread-safe connection pooling,
exponential backoff auto-reconnect, IEEE 754 32-bit float registers, multi-coil atomic writes (FC 15),
and full Modbus exception protocol handling.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Any

from packages.cs_core.interfaces.io_controller import IoController

logger = logging.getLogger(__name__)

# Standard Modbus Function Codes
FC_READ_COILS = 0x01
FC_READ_DISCRETE_INPUTS = 0x02
FC_READ_HOLDING_REGISTERS = 0x03
FC_READ_INPUT_REGISTERS = 0x04
FC_WRITE_SINGLE_COIL = 0x05
FC_WRITE_SINGLE_REGISTER = 0x06
FC_WRITE_MULTIPLE_COILS = 0x0F
FC_WRITE_MULTIPLE_REGISTERS = 0x10

# Modbus Standard Exception Codes
MODBUS_EXCEPTIONS: dict[int, str] = {
    0x01: "Illegal Function Code",
    0x02: "Illegal Data Address",
    0x03: "Illegal Data Value",
    0x04: "Slave Device Failure",
    0x05: "Acknowledge (Operation in Progress)",
    0x06: "Slave Device Busy",
    0x0A: "Gateway Path Unavailable",
    0x0B: "Gateway Target Device Failed to Respond",
}

# Default Mapping for Coils
DEFAULT_COIL_MAP: dict[str, int] = {
    "conveyor_run": 0,
    "reject_diverter": 1,
    "warning_horn": 2,
    "green_light": 3,
    "error_red": 4,
    "photoeye_interlock": 5,
}

# Default Mapping for Holding Registers (16-bit and 32-bit Float)
DEFAULT_REGISTER_MAP: dict[str, int] = {
    "counted_total": 100,
    "target_count": 101,
    "belt_speed_mm_s": 102,
    "active_session_id": 103,
    "area_estimate_m2": 104,  # Uses 2 registers (104, 105) as IEEE 754 Float
}


class ModbusTcpError(RuntimeError):
    """Raised when a Modbus TCP protocol or hardware communication error occurs."""


class ModbusTcpIoController:
    """Enterprise-grade industrial Modbus TCP client for PLC automation."""

    def __init__(
        self,
        host: str = "192.168.1.100",
        port: int = 502,
        unit_id: int = 1,
        timeout_seconds: float = 2.0,
        max_retries: int = 3,
        coil_map: dict[str, int] | None = None,
        register_map: dict[str, int] | None = None,
        socket_client: Any = None,
    ) -> None:
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.coil_map = dict(coil_map) if coil_map is not None else dict(DEFAULT_COIL_MAP)
        self.register_map = dict(register_map) if register_map is not None else dict(DEFAULT_REGISTER_MAP)

        self._lock = threading.RLock()
        self._transaction_id = 1
        self._sock = socket_client
        self._last_connected_at: float = 0.0

    def _next_tx_id(self) -> int:
        with self._lock:
            tx = self._transaction_id
            self._transaction_id = (self._transaction_id + 1) % 65535 or 1
            return tx

    def _get_socket(self) -> Any:
        """Thread-safe socket retrieval with exponential backoff auto-reconnect."""
        with self._lock:
            if self._sock is not None:
                return self._sock

            last_err = None
            for attempt in range(self.max_retries):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(self.timeout)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.connect((self.host, self.port))
                    self._sock = s
                    self._last_connected_at = time.time()
                    logger.info(f"[ModbusTCP] Connected to PLC at {self.host}:{self.port} (Unit: {self.unit_id})")
                    return self._sock
                except Exception as exc:
                    last_err = exc
                    backoff = min(2.0, 0.25 * (2 ** attempt))
                    time.sleep(backoff)

            raise ModbusTcpError(f"Failed to connect to Modbus PLC at {self.host}:{self.port}: {last_err}") from last_err

    def _send_and_receive(self, pdu: bytes, expected_fc: int) -> bytes:
        """Send MBAP + PDU, validate transaction ID and check exception responses."""
        with self._lock:
            sock = self._get_socket()
            tx_id = self._next_tx_id()
            length = len(pdu) + 1  # PDU + Unit ID byte

            # MBAP Header: TxID (2B), ProtocolID=0 (2B), Length (2B), UnitID (1B)
            mbap = struct.pack(">HHHB", tx_id, 0, length, self.unit_id)
            request = mbap + pdu

            try:
                sock.sendall(request)
                # Read 7-byte MBAP header
                resp_header = self._recv_exact(sock, 7)
                resp_tx, resp_proto, resp_len, resp_uid = struct.unpack(">HHHB", resp_header)

                if resp_tx != tx_id:
                    raise ModbusTcpError(f"Transaction ID mismatch: expected {tx_id}, got {resp_tx}")

                # Read remaining PDU bytes
                resp_pdu = self._recv_exact(sock, resp_len - 1)
                fc = resp_pdu[0]

                # Check if exception response
                if fc == (expected_fc | 0x80):
                    exc_code = resp_pdu[1] if len(resp_pdu) > 1 else 0
                    desc = MODBUS_EXCEPTIONS.get(exc_code, f"Unknown Error (0x{exc_code:02X})")
                    raise ModbusTcpError(f"Modbus PLC Exception [FC={fc:#04x}]: {desc}")

                if fc != expected_fc:
                    raise ModbusTcpError(f"Function Code mismatch: expected {expected_fc}, got {fc}")

                return resp_pdu

            except Exception as exc:
                self.close()
                if isinstance(exc, ModbusTcpError):
                    raise
                raise ModbusTcpError(f"Modbus communication error on {self.host}:{self.port}: {exc}") from exc

    def _recv_exact(self, sock: Any, num_bytes: int) -> bytes:
        """Read exactly num_bytes from socket stream."""
        buf = bytearray()
        while len(buf) < num_bytes:
            chunk = sock.recv(num_bytes - len(buf))
            if not chunk:
                raise ModbusTcpError(f"Connection closed prematurely by remote host (got {len(buf)}/{num_bytes} bytes)")
            buf.extend(chunk)
        return bytes(buf)

    # -----------------------------------------------------------------------
    # Coils (Discrete Digital I/O)
    # -----------------------------------------------------------------------

    def set_signal(self, name: str, value: bool) -> None:
        """Write single digital coil output (FC 05)."""
        if name not in self.coil_map:
            raise KeyError(f"Unknown coil signal '{name}' -- registered coils: {sorted(self.coil_map.keys())}")

        address = self.coil_map[name]
        coil_val = 0xFF00 if value else 0x0000
        pdu = struct.pack(">BHH", FC_WRITE_SINGLE_COIL, address, coil_val)
        self._send_and_receive(pdu, expected_fc=FC_WRITE_SINGLE_COIL)
        logger.info(f"[ModbusTCP {self.host}] Signal '{name}' (Coil {address}) -> {'ON' if value else 'OFF'}")

    def read_signal(self, name: str) -> bool:
        """Read single digital coil status (FC 01)."""
        if name not in self.coil_map:
            raise KeyError(f"Unknown coil signal '{name}' -- registered coils: {sorted(self.coil_map.keys())}")

        address = self.coil_map[name]
        pdu = struct.pack(">BHH", FC_READ_COILS, address, 1)
        resp_pdu = self._send_and_receive(pdu, expected_fc=FC_READ_COILS)
        byte_count = resp_pdu[1]
        if byte_count < 1:
            raise ModbusTcpError("Invalid response byte count in coil read")
        return bool(resp_pdu[2] & 0x01)

    def write_multiple_coils(self, start_name: str, values: list[bool]) -> None:
        """Write contiguous block of coils atomically (FC 15)."""
        if start_name not in self.coil_map:
            raise KeyError(f"Unknown starting coil '{start_name}'")
        start_addr = self.coil_map[start_name]
        qty = len(values)

        byte_count = (qty + 7) // 8
        coil_bytes = bytearray(byte_count)
        for i, val in enumerate(values):
            if val:
                coil_bytes[i // 8] |= (1 << (i % 8))

        pdu = struct.pack(">BHHB", FC_WRITE_MULTIPLE_COILS, start_addr, qty, byte_count) + coil_bytes
        self._send_and_receive(pdu, expected_fc=FC_WRITE_MULTIPLE_COILS)

    # -----------------------------------------------------------------------
    # Registers (16-bit INT & 32-bit IEEE 754 FLOAT)
    # -----------------------------------------------------------------------

    def write_register(self, name: str, value: int) -> None:
        """Write single 16-bit holding register (FC 06)."""
        if name not in self.register_map:
            raise KeyError(f"Unknown register '{name}' -- registered registers: {sorted(self.register_map.keys())}")

        address = self.register_map[name]
        pdu = struct.pack(">BHH", FC_WRITE_SINGLE_REGISTER, address, int(value) & 0xFFFF)
        self._send_and_receive(pdu, expected_fc=FC_WRITE_SINGLE_REGISTER)

    def read_register(self, name: str) -> int:
        """Read single 16-bit holding register (FC 03)."""
        if name not in self.register_map:
            raise KeyError(f"Unknown register '{name}' -- registered registers: {sorted(self.register_map.keys())}")

        address = self.register_map[name]
        pdu = struct.pack(">BHH", FC_READ_HOLDING_REGISTERS, address, 1)
        resp_pdu = self._send_and_receive(pdu, expected_fc=FC_READ_HOLDING_REGISTERS)
        val = struct.unpack(">H", resp_pdu[2:4])[0]
        return val

    def write_float32(self, name: str, value: float) -> None:
        """Write 32-bit IEEE 754 Float across two 16-bit registers (FC 16)."""
        if name not in self.register_map:
            raise KeyError(f"Unknown float register '{name}'")
        address = self.register_map[name]
        raw_bytes = struct.pack(">f", float(value))
        pdu = struct.pack(">BHHB", FC_WRITE_MULTIPLE_REGISTERS, address, 2, 4) + raw_bytes
        self._send_and_receive(pdu, expected_fc=FC_WRITE_MULTIPLE_REGISTERS)

    def read_float32(self, name: str) -> float:
        """Read 32-bit IEEE 754 Float across two 16-bit registers (FC 03)."""
        if name not in self.register_map:
            raise KeyError(f"Unknown float register '{name}'")
        address = self.register_map[name]
        pdu = struct.pack(">BHH", FC_READ_HOLDING_REGISTERS, address, 2)
        resp_pdu = self._send_and_receive(pdu, expected_fc=FC_READ_HOLDING_REGISTERS)
        val = struct.unpack(">f", resp_pdu[2:6])[0]
        return float(val)

    def close(self) -> None:
        """Close active socket cleanly."""
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
