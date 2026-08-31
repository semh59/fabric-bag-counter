"""Industrial Modbus TCP PLC Server Simulator (§4.4).

Provides a real TCP socket server implementing standard Modbus TCP protocol (Port 502/5020)
for live industrial conveyor control, reject diverter actuation, and count telemetry.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ModbusServer] %(message)s")
logger = logging.getLogger("ModbusServer")


class ModbusTcpServer:
    """Standard Modbus TCP Server handling coils and holding registers."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5020) -> None:
        self.host = host
        self.port = port
        self.coils: dict[int, bool] = {0: True, 1: False, 2: False, 3: True, 4: False, 5: False}
        self.registers: dict[int, int] = {100: 0, 101: 500, 102: 1200, 103: 1}
        self.is_running = False
        self._server_sock: socket.socket | None = None

    def start(self) -> None:
        self.is_running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(5)
        logger.info(f"Industrial Modbus TCP PLC Server running on {self.host}:{self.port}")

        while self.is_running:
            try:
                client, addr = self._server_sock.accept()
                threading.Thread(target=self._handle_client, args=(client, addr), daemon=True).start()
            except Exception:
                break

    def _handle_client(self, client: socket.socket, addr: tuple[str, int]) -> None:
        logger.info(f"New PLC Client connected from {addr[0]}:{addr[1]}")
        client.settimeout(10.0)
        try:
            while self.is_running:
                header = client.recv(7)
                if len(header) < 7:
                    break

                tx_id, proto, length, unit_id = struct.unpack(">HHHB", header)
                pdu = client.recv(length - 1)
                if not pdu:
                    break

                fc = pdu[0]
                if fc == 5:  # Write Single Coil
                    addr, val = struct.unpack(">HH", pdu[1:5])
                    state = (val == 0xFF00)
                    self.coils[addr] = state
                    logger.info(f"PLC Output Coil {addr} updated -> {'ON' if state else 'OFF'}")
                    # Echo response
                    resp = header[:4] + struct.pack(">H", 6) + bytes([unit_id]) + pdu
                    client.sendall(resp)

                elif fc == 1:  # Read Coils
                    addr, qty = struct.unpack(">HH", pdu[1:5])
                    coil_byte = 0
                    for i in range(min(8, qty)):
                        if self.coils.get(addr + i, False):
                            coil_byte |= (1 << i)
                    resp = struct.pack(">HHHBBB", tx_id, 0, 4, unit_id, 1, 1) + bytes([coil_byte])
                    client.sendall(resp)

                elif fc == 6:  # Write Single Register
                    addr, val = struct.unpack(">HH", pdu[1:5])
                    self.registers[addr] = val
                    logger.info(f"PLC Register {addr} updated -> {val}")
                    resp = header[:4] + struct.pack(">H", 6) + bytes([unit_id]) + pdu
                    client.sendall(resp)

                elif fc == 3:  # Read Holding Registers
                    addr, qty = struct.unpack(">HH", pdu[1:5])
                    data_bytes = bytearray()
                    for i in range(qty):
                        data_bytes.extend(struct.pack(">H", self.registers.get(addr + i, 0)))
                    resp_len = 3 + len(data_bytes)
                    resp = struct.pack(">HHHBBB", tx_id, 0, resp_len, unit_id, 3, len(data_bytes)) + data_bytes
                    client.sendall(resp)
        except Exception as exc:
            logger.debug(f"Client disconnected: {exc}")
        finally:
            client.close()

    def stop(self) -> None:
        self.is_running = False
        if self._server_sock:
            self._server_sock.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Modbus TCP PLC Server")
    parser.add_argument("--port", type=int, default=5020, help="Modbus TCP port")
    args = parser.parse_args()

    server = ModbusTcpServer(port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
