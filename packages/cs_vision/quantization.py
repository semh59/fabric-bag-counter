"""INT8 Model Quantization Engine for Edge Accelerators & Industrial Devices (§6.2, §11 M4).

Implements INT8 quantization (Static & Dynamic) for RF-DETR Seg models using ONNX Runtime
and NVIDIA TensorRT calibration principles, reducing model memory footprint by up to 75%
and cutting edge CPU/GPU inference latency to sub-10ms.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_vision.preprocess import preprocess_image

logger = logging.getLogger(__name__)


class BagOnnxCalibrationDataReader:
    """Feeds preprocessed conveyor camera frames to the ONNX Runtime INT8 calibrator."""

    def __init__(
        self,
        data_dir: Path | None = None,
        max_samples: int = 30,
        input_size: tuple[int, int] = (640, 640),
        input_name: str = "input",
    ) -> None:
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.input_size = input_size
        self.input_name = input_name
        self._samples: list[dict[str, np.ndarray]] = []
        self._index = 0
        self._prepare_calibration_data()

    def _prepare_calibration_data(self) -> None:
        """Collect real factory conveyor images or generate representative synthetic frames."""
        collected: list[np.ndarray] = []

        # 1. Try real factory frames from data/real_bags or specified directory
        target_dirs = [self.data_dir] if self.data_dir else [Path("data/real_bags/images"), Path("data/extracted_frames")]
        for p in target_dirs:
            if p and p.exists():
                for f in sorted(list(p.glob("*.jpg")) + list(p.glob("*.png"))):
                    img = cv2.imread(str(f))
                    if img is not None:
                        collected.append(img)
                    if len(collected) >= self.max_samples:
                        break
            if len(collected) >= self.max_samples:
                break

        # 2. If fewer than 10 real images found, supplement with photorealistic synthetic bag scenes
        if len(collected) < 10:
            gen = SyntheticBagGenerator()
            needed = self.max_samples - len(collected)
            for i in range(needed):
                # Varied bag counts (1, 2, 3) to calibrate both single and touching bag activation distributions
                num_bags = 1 + (i % 3)
                scene = gen.generate_scene(num_bags=num_bags)
                collected.append(scene["image"])

        # Preprocess each image into float32 NCHW tensor
        for img in collected[:self.max_samples]:
            blob, _, _ = preprocess_image(img, self.input_size, apply_retinex=True)
            self._samples.append({self.input_name: blob})

        logger.info(f"[INT8 Calibrator] Prepared {len(self._samples)} calibration samples (Input: {self.input_name})")

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return next input tensor dictionary for ONNX Runtime calibration."""
        if self._index < len(self._samples):
            item = self._samples[self._index]
            self._index += 1
            return item
        return None

    def rewind(self) -> None:
        self._index = 0


def quantize_rfdetr_model_int8(
    input_model_path: str | Path,
    output_model_path: str | Path | None = None,
    mode: str = "dynamic",
    calibration_data_dir: str | Path | None = None,
    max_calibration_samples: int = 30,
) -> dict[str, Any]:
    """Quantize an RF-DETR Seg ONNX model to INT8 precision.

    Parameters:
        input_model_path: Path to source FP32 ONNX model.
        output_model_path: Destination path for quantized INT8 model.
        mode: 'dynamic' (quantizes weights to INT8, activations quantized on-the-fly; fast & robust),
              or 'static' (full INT8 weights + activations using offline calibration reader).
        calibration_data_dir: Directory containing real images for static calibration.
        max_calibration_samples: Number of calibration frames to sample.

    Returns:
        dict containing 'status', 'output_path', 'fp32_size_mb', 'int8_size_mb', 'compression_ratio'.
    """
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    in_path = Path(input_model_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Source model not found at {in_path}")

    out_path = Path(output_model_path) if output_model_path else in_path.with_name(f"{in_path.stem}_int8.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fp32_size = in_path.stat().st_size / (1024 * 1024)
    t0 = time.perf_counter()

    # Discover input tensor name
    sess = ort.InferenceSession(str(in_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    if mode.lower() == "static":
        logger.info(f"[INT8 Quantization] Running STATIC calibration on '{in_path.name}'...")
        calibrator = BagOnnxCalibrationDataReader(
            data_dir=Path(calibration_data_dir) if calibration_data_dir else None,
            max_samples=max_calibration_samples,
            input_name=input_name,
        )
        quantize_static(
            model_input=str(in_path),
            model_output=str(out_path),
            calibration_data_reader=calibrator,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    else:
        logger.info(f"[INT8 Quantization] Running DYNAMIC weight quantization on '{in_path.name}'...")
        quantize_dynamic(
            model_input=str(in_path),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
        )

    elapsed = time.perf_counter() - t0
    int8_size = out_path.stat().st_size / (1024 * 1024)
    ratio = (1.0 - (int8_size / max(0.001, fp32_size))) * 100.0

    # Verification: Test run inference with the quantized model
    test_session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    test_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    outputs = test_session.run(None, {test_session.get_inputs()[0].name: test_input})
    assert len(outputs) >= 3, "Quantized model failed to produce required output tensors!"

    logger.info(
        f"[INT8 Quantization] Completed in {elapsed:.2f}s! "
        f"Size: {fp32_size:.2f}MB -> {int8_size:.2f}MB ({ratio:.1f}% reduction). Output: {out_path}"
    )

    return {
        "status": "success",
        "output_path": str(out_path),
        "mode": mode,
        "fp32_size_mb": round(fp32_size, 2),
        "int8_size_mb": round(int8_size, 2),
        "compression_percent": round(ratio, 1),
        "quantization_time_seconds": round(elapsed, 2),
    }
