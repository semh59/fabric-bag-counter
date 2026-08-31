"""Unit tests for standalone Modbus TCP PLC Server Simulator."""

import socket
import struct
import threading
import time
import pytest
from tools.modbus_server import ModbusTcpServer


def test_standalone_modbus_server_handles_live_tcp_client():
    server = ModbusTcpServer(host="127.0.0.1", port=5502)
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    time.sleep(0.3)  # Wait for socket bind

    try:
        # Connect real TCP client
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(("127.0.0.1", 5502))

        # 1. Write Single Coil (Coil 0 -> ON)
        req_write_coil = struct.pack(">HHHBBHH", 1, 0, 6, 1, 5, 0, 0xFF00)
        s.sendall(req_write_coil)
        resp = s.recv(12)
        assert len(resp) == 12
        assert server.coils[0] is True

        # 2. Read Coils (Coil 0 -> should be 1)
        req_read_coil = struct.pack(">HHHBBHH", 2, 0, 6, 1, 1, 0, 1)
        s.sendall(req_read_coil)
        resp_read = s.recv(10)
        assert len(resp_read) == 10
        coil_byte = resp_read[9]
        assert coil_byte & 0x01 == 1

        # 3. Write Single Register (Register 100 -> 999)
        req_write_reg = struct.pack(">HHHBBHH", 3, 0, 6, 1, 6, 100, 999)
        s.sendall(req_write_reg)
        resp_reg = s.recv(12)
        assert len(resp_reg) == 12
        assert server.registers[100] == 999

        # 4. Read Holding Register (Register 100 -> should be 999)
        req_read_reg = struct.pack(">HHHBBHH", 4, 0, 6, 1, 3, 100, 1)
        s.sendall(req_read_reg)
        resp_rreg = s.recv(11)
        assert len(resp_rreg) == 11
        val = struct.unpack(">H", resp_rreg[9:11])[0]
        assert val == 999

        s.close()
    finally:
        server.stop()
