"""SharedMemoryTransport: Zero-copy ring buffer IPC frame transport (§4.5).

Frame *metadata* (camera id, stream epoch, frame index, timestamps, and the
shared-memory block name/shape/dtype the pixels live in) flows through an
in-process per-camera ring buffer -- a bounded deque per camera, exactly as
before. That part is intentionally cheap and in-process: IngestWorker and
InferenceWorker in this codebase share one SharedMemoryTransport instance.

Frame *pixel data* is written into real OS-level shared memory segments via
the stdlib ``multiprocessing.shared_memory.SharedMemory``. A genuinely
separate OS process that only knows a frame's ``shm_name`` (carried on the
``Frame`` dataclass, see packages/cs_core/frame.py) can independently do::

    shm = shared_memory.SharedMemory(name=frame.shm_name)
    arr = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)

and get the identical pixel bytes with zero copies -- this is exactly the
reconstruction step ``get_image_data`` below performs when called
in-process, and what a real cross-process inference worker would do
out-of-process (see that method's docstring for why it then copies the
result out before returning). ``write_image_data`` is the mirror: it
creates a named block sized for the frame and copies the array's bytes into
it once.

Ring design: camera drivers name their shared-memory blocks using a fixed
per-camera slot cycle (``shm_<driver>_cam_<id>_slot_<frame_counter % N>``,
see drivers/video_rtsp/driver.py and drivers/video_file/driver.py), so in
steady state there are only ever ``ring_slots`` distinct block *names* live
per camera -- a "pool of reusable named slots" matching the ring buffer the
class docstring has always claimed. Each write to a given name creates a
fresh block and destroys whatever previously held that name (see
`write_image_data` for why this is a fresh block rather than an in-place
overwrite). Blocks are also destroyed (closed + unlinked) when: (a) a frame
carrying that name is dropped from a full ring (``publish``), (b) a frame
is explicitly released after processing (``release``), or (c) the
transport itself is torn down (``close``) -- between these paths, a name
is never left holding a block nobody can still legitimately reach, so
segments don't leak.

Windows note (this dev/deploy environment is Windows): unlike POSIX,
``multiprocessing.shared_memory`` on Windows has no filesystem-backed
persistence for a segment -- the OS reference-counts the underlying mapping
and frees it automatically once every open handle to that name is closed,
in *any* process. Consequences that shaped this implementation:
  * ``SharedMemory.unlink()`` is a documented no-op on Windows (there is no
    POSIX ``/dev/shm`` entry to remove) -- we still call it unconditionally
    on cleanup since it is required on POSIX and harmless on Windows.
  * A block only stays alive as long as *some* handle referencing it remains
    open somewhere. This transport keeps its own writer-side handle for
    every live slot in ``self._shm_slots`` for exactly this reason -- if we
    only ever created a handle transiently inside ``write_image_data`` and
    let it go out of scope, Windows would free the segment out from under
    any reader before ``get_image_data``/a real remote process could attach.
  * Every handle we open (including read-side attaches in a genuinely
    separate process) must eventually be closed or the segment leaks for
    the life of that process -- there is no unlink-while-still-mapped
    semantics to fall back on as there is on POSIX.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from packages.cs_core.frame import Frame
from packages.cs_core.interfaces.frame_transport import FrameTransport, PublishResult

logger = logging.getLogger(__name__)


@dataclass
class _ShmSlot:
    """Bookkeeping for one live shared-memory backed image buffer."""

    handle: shared_memory.SharedMemory
    shape: tuple[int, ...]
    dtype: str


class SharedMemoryTransport:
    """Ring buffer frame transport across ingest workers and inference engine (§4.5)."""

    def __init__(self, ring_slots: int = 8) -> None:
        self.ring_slots = ring_slots
        self._lock = threading.Lock()
        # Per-camera frame ring queues (metadata only)
        self._camera_queues: dict[int, collections.deque[Frame]] = {}
        # Per-slot real shared memory handles: shm_name -> _ShmSlot
        self._shm_slots: dict[str, _ShmSlot] = {}
        # Dropped frame counters
        self.dropped_frame_counts: dict[int, int] = collections.defaultdict(int)
        self.consecutive_drops: dict[int, int] = collections.defaultdict(int)

    # ------------------------------------------------------------------
    # Pixel data: real OS-level shared memory
    # ------------------------------------------------------------------

    def write_image_data(self, shm_name: str, image: np.ndarray) -> None:
        """Copy image bytes into a named OS-level shared memory block.

        Always creates a brand-new block for `shm_name`, closing/unlinking
        any previous block registered under that name first. This
        deliberately mirrors the "new object per write" semantics of a
        dict-of-copies design: named ring slots get physically reused
        (drivers cycle `..._slot_{frame_counter % ring_slots}`), but each
        write is a fresh, independently-owned segment rather than an
        in-place mutation of a block a slow consumer might still be
        reading -- reusing the same open handle in place would let a
        producer silently overwrite pixel data out from under a consumer
        that has not released the previous frame with the same slot name
        yet. A consumer that is too slow to keep up with the ring (i.e. the
        matching Frame metadata was already evicted from the queue, see
        `publish`) will instead simply find no block under that name -- the
        `get_image_data` contract already treats that as "unavailable"
        rather than serving stale/wrong pixels.
        """
        image = np.ascontiguousarray(image)
        nbytes = max(image.nbytes, 1)

        with self._lock:
            if shm_name in self._shm_slots:
                self._close_slot_locked(shm_name)

            handle = self._create_shm_locked(shm_name, nbytes)
            slot = _ShmSlot(handle=handle, shape=tuple(image.shape), dtype=str(image.dtype))
            self._shm_slots[shm_name] = slot

            dest = np.ndarray(image.shape, dtype=image.dtype, buffer=slot.handle.buf)
            dest[...] = image

    def _create_shm_locked(self, shm_name: str, nbytes: int) -> shared_memory.SharedMemory:
        """Create a fresh named shared memory block, clearing a stale one if needed."""
        try:
            return shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
        except FileExistsError:
            # A stale block from a previous (crashed/leaked) run still holds this
            # name. Attach, release it, and recreate cleanly.
            try:
                stale = shared_memory.SharedMemory(name=shm_name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            return shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)

    def get_image_data(self, shm_name: str) -> np.ndarray | None:
        """Reconstruct the numpy image array for `shm_name`.

        Internally this is exactly the zero-copy reconstruction a genuinely
        separate process would do after attaching --
        ``np.ndarray(shape, dtype, buffer=shm.buf)`` directly over the OS
        shared memory mapping, no pixel bytes are copied to get there. The
        result is then copied out (`.copy()`) before being returned: this
        transport instance keeps owning and reusing the underlying
        `SharedMemory` handle (see `write_image_data`/`_close_slot_locked`),
        and CPython raises `BufferError` if that handle's mmap is closed
        while any numpy array still holds an open buffer export into it --
        which would otherwise happen the moment a caller holds onto the
        returned frame past a later `release()`/ring-eviction/slot-reuse
        for the same name. Copying here decouples the returned array's
        lifetime from the shared memory block's lifetime, exactly as a real
        remote process would have to (attach, reconstruct, copy what it
        needs, then detach) since it cannot keep the producer's slot pinned
        open indefinitely either.

        Returns None if no block is currently registered under that name
        (already released or never written) -- callers must not fabricate a
        substitute frame in that case.
        """
        with self._lock:
            slot = self._shm_slots.get(shm_name)
            if slot is None:
                return None
            view = np.ndarray(slot.shape, dtype=slot.dtype, buffer=slot.handle.buf)
            return view.copy()

    def _close_slot_locked(self, shm_name: str) -> None:
        """Close and unlink the shared memory block for `shm_name`, if any."""
        slot = self._shm_slots.pop(shm_name, None)
        if slot is None:
            return
        try:
            slot.handle.close()
        except Exception:
            logger.debug(f"[SharedMemoryTransport] Error closing shm block {shm_name!r}", exc_info=True)
        try:
            slot.handle.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            # unlink() is a documented no-op on Windows; any other platform
            # error here is non-fatal for the caller but worth logging.
            logger.debug(f"[SharedMemoryTransport] Error unlinking shm block {shm_name!r}", exc_info=True)

    # ------------------------------------------------------------------
    # Frame metadata ring buffer
    # ------------------------------------------------------------------

    def publish(self, frame: Frame) -> PublishResult:
        """Publish frame metadata into camera ring buffer.

        If ring buffer is full, drops the oldest frame (§4.5 non-blocking ingest)
        and releases its shared memory block.
        """
        with self._lock:
            queue = self._camera_queues.setdefault(frame.camera_id, collections.deque(maxlen=self.ring_slots))

            if len(queue) >= self.ring_slots:
                # Ring full: discard oldest frame and free its shm block
                dropped = queue.popleft()
                self._close_slot_locked(dropped.shm_name)
                self.dropped_frame_counts[frame.camera_id] += 1
                self.consecutive_drops[frame.camera_id] += 1
            else:
                self.consecutive_drops[frame.camera_id] = 0

            queue.append(frame)

            return PublishResult(
                success=True,
                dropped_frames=self.dropped_frame_counts[frame.camera_id],
                consecutive_drops=self.consecutive_drops[frame.camera_id],
            )

    def consume(self, timeout_ms: int = 30) -> list[Frame]:
        """Consume available frames across all camera ring queues within timeout."""
        start_t = time.monotonic()
        consumed: list[Frame] = []

        while (time.monotonic() - start_t) * 1000.0 < timeout_ms:
            with self._lock:
                for cid, queue in self._camera_queues.items():
                    if queue:
                        consumed.append(queue.popleft())
            if consumed:
                break
            time.sleep(0.002)

        return consumed

    def release(self, frame: Frame) -> None:
        """Release shared memory block after inference completion."""
        with self._lock:
            self._close_slot_locked(frame.shm_name)

    def close(self) -> None:
        """Release every remaining shared memory block. Call on transport teardown.

        Since blocks are only truly freed once every open handle to them is
        closed (see the Windows note in the module docstring), a long-lived
        transport instance should call this on shutdown to avoid leaking
        segments that were published but never consumed/released.
        """
        with self._lock:
            for shm_name in list(self._shm_slots.keys()):
                self._close_slot_locked(shm_name)

    def get_stats(self) -> dict[str, Any]:
        """Return frame drop statistics across cameras."""
        with self._lock:
            return {
                "dropped_frame_counts": dict(self.dropped_frame_counts),
                "consecutive_drops": dict(self.consecutive_drops),
            }
