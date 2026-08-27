"""Integration tests proving SharedMemoryTransport's pixel buffer is backed by
real OS-level shared memory (multiprocessing.shared_memory), not an in-process
dict masquerading as "shared memory" IPC.

The key test (`test_independent_handle_reads_identical_bytes`) writes a frame
through the transport, then opens a *second, wholly independent*
`multiprocessing.shared_memory.SharedMemory` handle by name only -- exactly
what a genuinely separate OS process would do, since it never touches the
transport instance or its internal Python objects, only the OS-level shared
memory name. This "simulates two OS processes within the same test process"
as specified: a real second process attaching to the block by name would see
byte-identical results for the same reason this second handle does -- the
data lives in the OS shared memory segment, not in Python-process memory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from multiprocessing import shared_memory

import numpy as np
import pytest

from packages.cs_core.frame import Frame
from packages.cs_core.transport import SharedMemoryTransport


def _make_frame(shm_name: str, shape: tuple[int, ...], camera_id: int = 1, frame_index: int = 0) -> Frame:
    return Frame(
        camera_id=camera_id,
        stream_epoch=1,
        frame_index=frame_index,
        monotonic_ns=0,
        wall_clock=datetime.now(timezone.utc),
        shm_name=shm_name,
        shape=shape,
        dtype="uint8",
    )


def test_write_then_read_back_is_byte_identical():
    """write_image_data followed by get_image_data returns byte-identical pixels
    (round-trip through the transport's own in-process handle)."""
    transport = SharedMemoryTransport(ring_slots=4)
    try:
        img = (np.random.RandomState(42).rand(64, 96, 3) * 255).astype(np.uint8)
        shm_name = "fabric_test_shm_roundtrip"

        transport.write_image_data(shm_name, img)
        readback = transport.get_image_data(shm_name)

        assert readback is not None
        assert readback.shape == img.shape
        assert readback.dtype == img.dtype
        assert np.array_equal(readback, img)
        assert readback.tobytes() == img.tobytes()
    finally:
        transport.close()


def test_independent_handle_reads_identical_bytes():
    """A second, independent SharedMemory handle opened purely by name (never
    touching the transport object) reconstructs byte-identical pixel data --
    this is exactly what a separate OS process attaching by `frame.shm_name`
    would do, proving the buffer is real OS-level shared memory and not a
    Python-process-local dict.
    """
    transport = SharedMemoryTransport(ring_slots=4)
    shm_name = "fabric_test_shm_cross_handle"
    external_handle = None
    try:
        img = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        transport.write_image_data(shm_name, img)

        # Independent attach by name only -- no reference to `transport` at all.
        external_handle = shared_memory.SharedMemory(name=shm_name, create=False)
        reconstructed = np.ndarray(img.shape, dtype=img.dtype, buffer=external_handle.buf)

        assert np.array_equal(reconstructed, img)
        assert bytes(external_handle.buf[: img.nbytes]) == img.tobytes()
    finally:
        if external_handle is not None:
            external_handle.close()
        transport.close()


def test_release_unlinks_the_shared_memory_block():
    """release() actually frees the OS shared memory block -- a subsequent
    attach-by-name must fail with FileNotFoundError, proving the block is
    genuinely gone (not just dropped from an in-process dict, which would
    be invisible to any other process holding/attempting the same name).
    """
    transport = SharedMemoryTransport(ring_slots=4)
    shm_name = "fabric_test_shm_release"
    try:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        transport.write_image_data(shm_name, img)

        frame = _make_frame(shm_name, img.shape)
        transport.release(frame)

        assert transport.get_image_data(shm_name) is None
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name, create=False)
    finally:
        transport.close()


def test_ring_full_drop_releases_shm_of_evicted_frame():
    """When the ring buffer is full, publish() evicts the oldest frame and must
    also free its shared memory block so blocks don't leak."""
    transport = SharedMemoryTransport(ring_slots=2)
    cam_id = 42
    names: list[str] = []
    try:
        for i in range(3):
            img = np.full((8, 8, 3), i, dtype=np.uint8)
            shm_name = f"fabric_test_shm_ring_{i}"
            transport.write_image_data(shm_name, img)
            frame = _make_frame(shm_name, img.shape, camera_id=cam_id, frame_index=i)
            transport.publish(frame)
            names.append(shm_name)

        # ring_slots=2: publishing the 3rd frame must evict frame 0.
        stats = transport.get_stats()
        assert stats["dropped_frame_counts"][cam_id] == 1
        assert transport.get_image_data(names[0]) is None
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=names[0], create=False)

        # The still-live frames' blocks must remain readable.
        assert transport.get_image_data(names[1]) is not None
        assert transport.get_image_data(names[2]) is not None
    finally:
        for f in transport.consume(timeout_ms=20):
            transport.release(f)
        transport.close()


def test_write_image_data_replaces_block_without_leaking():
    """Writing a new image under a name that already has a live block (ring
    slot reuse, e.g. camera wraps back to slot 0) replaces it cleanly: the
    new read reflects only the new bytes and the old block is unlinked."""
    transport = SharedMemoryTransport(ring_slots=4)
    shm_name = "fabric_test_shm_slot_reuse"
    try:
        img_a = np.full((16, 16, 3), 1, dtype=np.uint8)
        img_b = np.full((16, 16, 3), 2, dtype=np.uint8)

        transport.write_image_data(shm_name, img_a)
        first = transport.get_image_data(shm_name)
        assert np.array_equal(first, img_a)

        transport.write_image_data(shm_name, img_b)
        second = transport.get_image_data(shm_name)
        assert np.array_equal(second, img_b)
        assert not np.array_equal(second, img_a)
    finally:
        transport.close()
