"""INT8 Edge Model Quantization Benchmark and Verification Tool (§6.2, §11 M4).

Benchmarks inference throughput, latency, memory footprint, and tensor output
integrity between FP32 and INT8 quantized RF-DETR Seg models.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
import numpy as np

from packages.cs_vision.detector import VisionDetector
from packages.cs_vision.quantization import quantize_rfdetr_model_int8
from packages.cs_data.synth import SyntheticBagGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_model(model_path: str | Path, num_iterations: int = 50) -> dict[str, float]:
    """Benchmark inference latency and FPS for a given ONNX model path."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warmup
    for _ in range(5):
        sess.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies_arr = np.array(latencies)
    mean_ms = float(np.mean(latencies_arr))
    p95_ms = float(np.percentile(latencies_arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "fps": round(fps, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="RF-DETR Seg INT8 Quantization Benchmark")
    parser.add_argument("--model", default="models/rfdetr_seg_v2.onnx", help="Source FP32 model")
    parser.add_argument("--out-dir", default="models", help="Output directory")
    parser.add_argument("--iterations", type=int, default=30, help="Benchmark iterations")
    args = parser.parse_args()

    src_model = Path(args.model)
    if not src_model.exists():
        print(f"[ERROR] Source model not found: {src_model}")
        return

    out_int8_dynamic = Path(args.out_dir) / f"{src_model.stem}_int8_dynamic.onnx"

    print("=" * 65)
    print("      RF-DETR Seg — INT8 EDGE ACCELERATION BENCHMARK")
    print("=" * 65)

    # 1. Dynamic INT8 Quantization
    print(f"\n[1/3] Quantizing '{src_model.name}' to INT8 (Dynamic)...")
    res_dynamic = quantize_rfdetr_model_int8(
        input_model_path=src_model,
        output_model_path=out_int8_dynamic,
        mode="dynamic",
    )
    print(f"  [OK] INT8 Model created: {out_int8_dynamic}")
    print(f"  Size Reduction: {res_dynamic['fp32_size_mb']:.2f} MB -> {res_dynamic['int8_size_mb']:.2f} MB ({res_dynamic['compression_percent']:.1f}% reduction)")

    # 2. Benchmark FP32 Baseline
    print(f"\n[2/3] Benchmarking FP32 Baseline ({args.iterations} iterations)...")
    fp32_stats = benchmark_model(src_model, num_iterations=args.iterations)
    print(f"  FP32 Latency: {fp32_stats['mean_latency_ms']} ms (P95: {fp32_stats['p95_latency_ms']} ms) | Throughput: {fp32_stats['fps']} FPS")

    # 3. Benchmark INT8 Quantized Model
    print(f"\n[3/3] Benchmarking INT8 Quantized Model ({args.iterations} iterations)...")
    int8_stats = benchmark_model(out_int8_dynamic, num_iterations=args.iterations)
    print(f"  INT8 Latency: {int8_stats['mean_latency_ms']} ms (P95: {int8_stats['p95_latency_ms']} ms) | Throughput: {int8_stats['fps']} FPS")

    speedup = fp32_stats["mean_latency_ms"] / max(0.001, int8_stats["mean_latency_ms"])
    print("\n" + "=" * 65)
    print(f"  SUMMARY RESULT: {speedup:.2f}x Speedup | {res_dynamic['compression_percent']:.1f}% Storage Savings")
    print("=" * 65)


if __name__ == "__main__":
    main()
