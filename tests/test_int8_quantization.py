"""Unit and integration tests for INT8 Quantization and TensorRT Engine Builder (§6.2, §11 M4)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import ModelVersionORM, UserAccountORM
from packages.cs_storage.repositories.user_repo import UserRepository
from packages.cs_vision.detector import VisionDetector
from packages.cs_vision.quantization import BagOnnxCalibrationDataReader, quantize_rfdetr_model_int8
from packages.cs_vision.tensorrt_builder import TensorRtEngineBuilder
from services.api.main import app
from services.jobrunner.worker import JobrunnerWorker

client = TestClient(app)


def test_calibration_data_reader():
    reader = BagOnnxCalibrationDataReader(max_samples=5, input_name="images")
    sample = reader.get_next()
    assert sample is not None
    assert "images" in sample
    assert sample["images"].shape == (1, 3, 640, 640)
    assert sample["images"].dtype == np.float32

    count = 1
    while reader.get_next() is not None:
        count += 1
    assert count == 5

    # Rewind
    reader.rewind()
    assert reader.get_next() is not None


def test_quantize_rfdetr_model_dynamic(tmp_path: Path):
    src_model = Path("models/rfdetr_seg_v2.onnx")
    assert src_model.exists()

    out_int8 = tmp_path / "rfdetr_int8_test.onnx"
    res = quantize_rfdetr_model_int8(
        input_model_path=src_model,
        output_model_path=out_int8,
        mode="dynamic",
    )

    assert res["status"] == "success"
    assert out_int8.exists()
    assert res["int8_size_mb"] < res["fp32_size_mb"]
    assert res["compression_percent"] > 40.0

    # Verify inference execution with VisionDetector
    detector = VisionDetector(model_path=str(out_int8), conf_threshold=0.35, allow_fallback=False)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    out = detector.predict(dummy_frame)
    assert hasattr(out, "bag_bodies")
    assert out.is_fallback_mode is False


def test_tensorrt_builder_int8_options(tmp_path: Path):
    src_model = Path("models/rfdetr_seg_v2.onnx")
    builder = TensorRtEngineBuilder(
        onnx_model_path=str(src_model),
        engine_output_path=str(tmp_path / "test.engine"),
        use_fp16=True,
        use_int8=True,
    )

    opts = builder.get_onnxruntime_tensorrt_options()
    assert opts["trt_int8_enable"] is True
    assert opts["trt_fp16_enable"] is True
    assert opts["trt_int8_use_native_calibration_table"] is True
    assert "trt_int8_calibration_cache_name" in opts


def test_jobrunner_quantize_int8_job(tmp_path: Path):
    from unittest.mock import MagicMock
    worker = JobrunnerWorker(lease_seconds=30)
    out_model = tmp_path / "job_int8_test.onnx"
    job = MagicMock()
    job.id = 888
    job.kind = "quantize_int8"
    job.payload = {
        "model_path": "models/rfdetr_seg_v2.onnx",
        "output_path": str(out_model),
        "mode": "dynamic",
    }
    result = worker.execute_job(job)
    assert result["status"] == "success"
    assert out_model.exists()



def test_api_quantize_int8_endpoint():
    init_db_sync()
    with get_sync_session() as db:
        UserRepository(db).seed_default_users()
        mv = db.query(ModelVersionORM).first()
        if not mv:
            mv = ModelVersionORM(
                onnx_hash="sha256:test12345",
                onnx_path="models/rfdetr_seg_v2.onnx",
                stage="draft",
            )
            db.add(mv)
            db.commit()
            db.refresh(mv)
        mv_id = mv.id

    # Authenticate as engineer
    login_res = client.post("/api/auth/login", json={"username": "engineer", "password": "eng123"})
    assert login_res.status_code == 200
    token = login_res.json()["token"]

    headers = {"Authorization": f"Bearer {token}"}
    res = client.post(f"/api/models/{mv_id}/quantize_int8", json={"mode": "dynamic"}, headers=headers)
    assert res.status_code == 202
    data = res.json()
    assert data["kind"] == "quantize_int8"
    assert "job_id" in data
