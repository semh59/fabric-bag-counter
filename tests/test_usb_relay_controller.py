"""Tests for the Numato-compatible USB relay driver (§4.4).

Exercises the real command-encoding/response-parsing logic against a
minimal in-memory stand-in for a pyserial connection (write/read/flush/
reset_input_buffer) -- not a mock of set_signal/read_signal themselves, so
the actual ASCII protocol framing is what's under test.
"""

import pytest

from drivers.io_usb_relay.controller import DEFAULT_CHANNEL_MAP, UsbRelayIoController


class FakeSerialPort:
    """Minimal stand-in for a pyserial.Serial connection to a real relay board."""

    def __init__(self, read_reply: bytes = b"") -> None:
        self.written: list[bytes] = []
        self._read_reply = read_reply
        self.flush_count = 0
        self.reset_count = 0
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def flush(self) -> None:
        self.flush_count += 1

    def reset_input_buffer(self) -> None:
        self.reset_count += 1

    def read(self, n: int) -> bytes:
        return self._read_reply

    def close(self) -> None:
        self.closed = True


def test_set_signal_sends_real_relay_on_command():
    port = FakeSerialPort()
    ctrl = UsbRelayIoController(connection=port)
    ctrl.set_signal("conveyor_run", True)
    assert port.written[-1] == b"relay on 3\r"  # conveyor_run -> channel 3
    assert port.flush_count == 1


def test_set_signal_sends_real_relay_off_command():
    port = FakeSerialPort()
    ctrl = UsbRelayIoController(connection=port)
    ctrl.set_signal("error_red", False)
    assert port.written[-1] == b"relay off 2\r"  # error_red -> channel 2


def test_read_signal_parses_real_board_reply():
    port = FakeSerialPort(read_reply=b"relay read 3\r\non\r\n>")
    ctrl = UsbRelayIoController(connection=port)
    assert ctrl.read_signal("conveyor_run") is True
    assert port.written[-1] == b"relay read 3\r"
    assert port.reset_count == 1


def test_read_signal_parses_off_reply():
    port = FakeSerialPort(read_reply=b"relay read 1\r\noff\r\n>")
    ctrl = UsbRelayIoController(connection=port)
    assert ctrl.read_signal("warning_yellow") is False


def test_read_signal_falls_back_to_last_known_state_on_unparseable_reply():
    port = FakeSerialPort(read_reply=b"")  # board silent/disconnected
    ctrl = UsbRelayIoController(connection=port)
    ctrl.set_signal("green_light", True)
    # No reply at all -- must not fabricate a state, must report what was
    # last actually commanded.
    assert ctrl.read_signal("green_light") is True


def test_unknown_channel_raises_instead_of_silently_no_op():
    port = FakeSerialPort()
    ctrl = UsbRelayIoController(connection=port)
    with pytest.raises(KeyError):
        ctrl.set_signal("not_a_real_channel", True)


def test_custom_channel_map_overrides_default():
    port = FakeSerialPort()
    ctrl = UsbRelayIoController(connection=port, channel_map={"conveyor_run": 7})
    ctrl.set_signal("conveyor_run", True)
    assert port.written[-1] == b"relay on 7\r"


def test_default_channel_map_matches_documented_wiring():
    assert DEFAULT_CHANNEL_MAP == {
        "green_light": 0,
        "warning_yellow": 1,
        "error_red": 2,
        "conveyor_run": 3,
    }


def test_close_closes_the_real_connection():
    port = FakeSerialPort()
    ctrl = UsbRelayIoController(connection=port)
    ctrl.close()
    assert port.closed is True


def test_implements_io_controller_protocol():
    from packages.cs_core.interfaces.io_controller import IoController

    port = FakeSerialPort()
    ctrl = UsbRelayIoController(connection=port)
    assert isinstance(ctrl, IoController)
