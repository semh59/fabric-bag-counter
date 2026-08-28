"""Tests for the RTSP video driver (§4.4) -- previously had zero coverage.

Focus: the real open()/read()/close() lifecycle and, specifically, that
open() actually bounds RTSP connect/read time via cv2's params-array
VideoCapture constructor instead of the plain cv2.VideoCapture(url) call,
which was verified (against a real unreachable address, not a guess) to
block for far longer than the ingest worker's ~1s reconnect backoff.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from drivers.video_rtsp.driver import RtspVideoSource


def test_open_passes_timeout_params_before_connecting():
    """open() must use the params-array VideoCapture constructor (url,
    apiPreference, params) -- setting timeouts via .set() after construction
    is documented to be too late for CAP_PROP_OPEN_TIMEOUT_MSEC, since the
    plain constructor already blocks trying to connect before .set() could
    run."""
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True

    with patch("cv2.VideoCapture", return_value=fake_cap) as mock_ctor, \
         patch("cv2.CAP_FFMPEG", 1900), \
         patch("cv2.CAP_PROP_OPEN_TIMEOUT_MSEC", 53), \
         patch("cv2.CAP_PROP_READ_TIMEOUT_MSEC", 54):
        source = RtspVideoSource()
        source.open({"url": "rtsp://camera.local/stream"}, epoch=1)

    assert mock_ctor.call_count == 1
    args, kwargs = mock_ctor.call_args
    assert args[0] == "rtsp://camera.local/stream"
    assert args[1] == 1900  # cv2.CAP_FFMPEG
    params = args[2]
    assert params == [53, 5000, 54, 5000]  # default 5000ms for both
    assert source.is_connected is True


def test_open_honors_custom_timeout_config():
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True

    with patch("cv2.VideoCapture", return_value=fake_cap) as mock_ctor:
        source = RtspVideoSource()
        source.open({"url": "rtsp://camera.local/stream", "open_timeout_ms": 1500, "read_timeout_ms": 2500}, epoch=1)

    args, _ = mock_ctor.call_args
    params = args[2]
    assert 1500 in params
    assert 2500 in params


def test_open_sets_not_connected_when_isopened_false():
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=fake_cap):
        source = RtspVideoSource()
        source.open({"url": "rtsp://dead.local/stream"}, epoch=1)

    assert source.is_connected is False


def test_open_handles_constructor_exception_gracefully():
    with patch("cv2.VideoCapture", side_effect=RuntimeError("ffmpeg backend unavailable")):
        source = RtspVideoSource()
        source.open({"url": "rtsp://camera.local/stream"}, epoch=1)

    assert source.is_connected is False
    assert source.cap is None


def test_read_returns_none_and_disconnects_on_failed_frame():
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.read.return_value = (False, None)

    with patch("cv2.VideoCapture", return_value=fake_cap):
        source = RtspVideoSource()
        source.open({"url": "rtsp://camera.local/stream"}, epoch=1)

    transport = MagicMock()
    frame = source.read(transport)

    assert frame is None
    assert source.is_connected is False


def test_read_returns_real_frame_and_writes_to_transport():
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_cap.read.return_value = (True, img)

    with patch("cv2.VideoCapture", return_value=fake_cap):
        source = RtspVideoSource()
        source.open({"url": "rtsp://camera.local/stream", "camera_id": 7}, epoch=3)

    transport = MagicMock()
    frame = source.read(transport)

    assert frame is not None
    assert frame.camera_id == 7
    assert frame.stream_epoch == 3
    assert frame.frame_index == 1
    assert frame.shape == img.shape
    transport.write_image_data.assert_called_once()
    written_shm_name = transport.write_image_data.call_args[0][0]
    assert written_shm_name == frame.shm_name


def test_close_releases_capture_and_marks_disconnected():
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True

    with patch("cv2.VideoCapture", return_value=fake_cap):
        source = RtspVideoSource()
        source.open({"url": "rtsp://camera.local/stream"}, epoch=1)

    source.close()
    fake_cap.release.assert_called_once()
    assert source.is_connected is False
