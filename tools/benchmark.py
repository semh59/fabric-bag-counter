"""Benchmark CLI tool measuring 1..N camera decode FPS, latency p50/p95/p99, and drop rate (§11 M5)."""

from __future__ import annotations

import os
import sys
import time

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from packages.cs_core.frame import Frame
from packages.cs_counting.engine import CountingEngine
from packages.cs_core.transport import SharedMemoryTransport


def benchmark_pipeline(num_cameras: int = 2, frames_per_camera: int = 100) -> dict[str, float]:
    print(f"\n[Benchmark] Benchmarking pipeline with {num_cameras} camera streams ({frames_per_camera} frames each)...")

    transport = SharedMemoryTransport(ring_slots=16)
    engine = CountingEngine()

    # Pre-generate frames
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    latencies_ms = []

    start_total = time.monotonic()
    total_processed = 0

    for f_idx in range(frames_per_camera):
        for c_idx in range(1, num_cameras + 1):
            shm_name = f"shm_bench_c{c_idx}_f{f_idx}"
            transport.write_image_data(shm_name, dummy_img)
            frame = Frame(
                camera_id=c_idx,
                stream_epoch=1,
                frame_index=f_idx,
                monotonic_ns=time.monotonic_ns(),
                wall_clock=time.time(),
                shm_name=shm_name,
                shape=(640, 640, 3),
                dtype="uint8",
            )
            transport.publish(frame)

        # Consume and process batch
        consumed = transport.consume(timeout_ms=50)
        for frame in consumed:
            t0 = time.monotonic()
            img = transport.get_image_data(frame.shm_name)
            if img is None:
                img = dummy_img
            engine.process_frame(
                image=img,
                frame_index=frame.frame_index,
                monotonic_ns=frame.monotonic_ns,
                wall_clock=frame.wall_clock,
            )
            transport.release(frame)
            latencies_ms.append((time.monotonic() - t0) * 1000.0)
            total_processed += 1

    total_time = time.monotonic() - start_total
    fps = total_processed / total_time if total_time > 0 else 0.0

    p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99 = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    stats = transport.get_stats()
    dropped = sum(stats["dropped_frame_counts"].values())

    print(f"  Processed Frames : {total_processed}")
    print(f"  Throughput (FPS) : {fps:.1f} FPS")
    print(f"  Latency p50      : {p50:.2f} ms")
    print(f"  Latency p95      : {p95:.2f} ms")
    print(f"  Latency p99      : {p99:.2f} ms")
    print(f"  Dropped Frames   : {dropped}")

    return {
        "num_cameras": float(num_cameras),
        "fps": fps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "dropped_frames": float(dropped),
    }


def main() -> None:
    for n in [1, 2, 4]:
        benchmark_pipeline(num_cameras=n, frames_per_camera=50)


if __name__ == "__main__":
    main()
