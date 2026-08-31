"""Unit tests for Industrial Modbus TCP PLC Driver (§4.4)."""

import struct
import pytest
from drivers.io_modbus_tcp.controller import ModbusTcpIoController, ModbusTcpError


class MockModbusSocket:
    """Mock TCP socket simulating an industrial Modbus TCP PLC server."""

    def __init__(self) -> None:
        self.sent_bytes = bytearray()
        self.coils: dict[int, bool] = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}
        self.registers: dict[int, int] = {100: 0, 101: 500, 102: 1200, 103: 1, 104: 0, 105: 0}
        self._next_response = bytearray()

    def sendall(self, data: bytes) -> None:
        self.sent_bytes.extend(data)
        if len(data) >= 7:
            tx_id, proto, length, unit_id = struct.unpack(">HHHB", data[:7])
            fc = data[7]

            if fc == 5:  # Write Single Coil
                addr, val = struct.unpack(">HH", data[8:12])
                self.coils[addr] = (val == 0xFF00)
                self._next_response = bytearray(data)

            elif fc == 1:  # Read Coils
                addr, qty = struct.unpack(">HH", data[8:12])
                coil_val = 1 if self.coils.get(addr, False) else 0
                resp = struct.pack(">HHHBBB", tx_id, 0, 4, unit_id, 1, 1) + bytes([coil_val])
                self._next_response = bytearray(resp)

            elif fc == 6:  # Write Single Register
                addr, val = struct.unpack(">HH", data[8:12])
                self.registers[addr] = val
                self._next_response = bytearray(data)

            elif fc == 3:  # Read Holding Registers
                addr, qty = struct.unpack(">HH", data[8:12])
                data_bytes = bytearray()
                for i in range(qty):
                    data_bytes.extend(struct.pack(">H", self.registers.get(addr + i, 0)))
                resp = struct.pack(">HHHBBB", tx_id, 0, 3 + len(data_bytes), unit_id, 3, len(data_bytes)) + data_bytes
                self._next_response = bytearray(resp)

            elif fc == 16:  # Write Multiple Registers
                addr, qty, byte_count = struct.unpack(">HHB", data[8:13])
                raw_payload = data[13:13 + byte_count]
                for i in range(qty):
                    self.registers[addr + i] = struct.unpack(">H", raw_payload[i*2:(i+1)*2])[0]
                resp = struct.pack(">HHHBBHH", tx_id, 0, 6, unit_id, 16, addr, qty)
                self._next_response = bytearray(resp)

            elif fc == 15:  # Write Multiple Coils
                addr, qty, byte_count = struct.unpack(">HHB", data[8:13])
                coil_bytes = data[13:13 + byte_count]
                for i in range(qty):
                    bit = bool(coil_bytes[i // 8] & (1 << (i % 8)))
                    self.coils[addr + i] = bit
                resp = struct.pack(">HHHBBHH", tx_id, 0, 6, unit_id, 15, addr, qty)
                self._next_response = bytearray(resp)

    def recv(self, bufsize: int) -> bytes:
        data = bytes(self._next_response[:bufsize])
        self._next_response = self._next_response[bufsize:]
        return data

    def close(self) -> None:
        pass


def test_modbus_tcp_write_and_read_coil():
    mock_sock = MockModbusSocket()
    plc = ModbusTcpIoController(host="192.168.1.100", port=502, socket_client=mock_sock)

    assert plc.read_signal("conveyor_run") is False
    plc.set_signal("conveyor_run", True)
    assert mock_sock.coils[0] is True
    assert plc.read_signal("conveyor_run") is True


def test_modbus_tcp_write_and_read_register():
    mock_sock = MockModbusSocket()
    plc = ModbusTcpIoController(host="192.168.1.100", port=502, socket_client=mock_sock)

    assert plc.read_register("target_count") == 500
    plc.write_register("counted_total", 450)
    assert mock_sock.registers[100] == 450
    assert plc.read_register("counted_total") == 450


def test_modbus_tcp_write_and_read_float32():
    mock_sock = MockModbusSocket()
    plc = ModbusTcpIoController(host="192.168.1.100", port=502, socket_client=mock_sock)

    # Write 32-bit float
    plc.write_float32("area_estimate_m2", 42.75)
    val = plc.read_float32("area_estimate_m2")
    assert val == pytest.approx(42.75, rel=1e-4)


def test_modbus_tcp_write_multiple_coils():
    mock_sock = MockModbusSocket()
    plc = ModbusTcpIoController(host="192.168.1.100", port=502, socket_client=mock_sock)

    plc.write_multiple_coils("conveyor_run", [True, False, True])
    assert mock_sock.coils[0] is True
    assert mock_sock.coils[1] is False
    assert mock_sock.coils[2] is True
