"""Unit tests for TensorRT FP16 builder and configuration options."""

import pytest
from packages.cs_vision.tensorrt_builder import TensorRtEngineBuilder


def test_tensorrt_engine_builder_options(tmp_path):
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.touch()

    builder = TensorRtEngineBuilder(
        onnx_model_path=str(fake_onnx),
        use_fp16=True,
        max_workspace_gb=1.5,
    )

    opts = builder.get_onnxruntime_tensorrt_options()
    assert opts["trt_fp16_enable"] is True
    assert opts["device_id"] == 0
    assert opts["trt_max_workspace_size"] == int(1.5 * (1024 ** 3))
