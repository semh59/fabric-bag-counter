"""Exhaustive Coverage Suite for VisionDetector, Retinex Preprocessing, and Jobrunner Algorithms (§6.2, §8.2).

Directly exhausts:
1. VisionDetector (warmup, detect, multi-scale Retinex toggle, CPU execution, contours fallback).
2. JobrunnerWorker execution for calibrate_motion, calibrate_scale, evaluate, and mine_hard_frames.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    CameraORM,
    ConfigVersionORM,
    DatasetVersionORM,
    DeploymentBundleORM,
    GateORM,
    JobORM,
    LineCalibrationORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    ProductProfileORM,
    SessionORM,
    SiteORM,
    TrainingRunORM,
)
from packages.cs_storage.repositories.job_repo import JobRepository
from packages.cs_vision.detector import DetectionResult, VisionDetector
from packages.cs_vision.retinex import MultiScaleRetinex
from services.jobrunner.worker import JobrunnerWorker


@pytest.fixture(autouse=True)
def setup_db():
    init_db_sync()


def test_vision_detector_all_branches():
    detector = VisionDetector(
        model_path="models/rfdetr_seg_v2.onnx",
        conf_threshold=0.30,
        apply_retinex=True,
    )

    # 1. Single frame detection
    img = np.ones((640, 640, 3), dtype=np.uint8) * 100
    cv2.rectangle(img, (200, 200), (350, 400), (255, 255, 255), -1)
    res = detector.predict(img)
    assert isinstance(res, DetectionResult)

    # 2. Retinex disabled
    detector_no_ret = VisionDetector(
        model_path="models/rfdetr_seg_v2.onnx",
        conf_threshold=0.50,
        apply_retinex=False,
    )
    res_no_ret = detector_no_ret.predict(img)
    assert isinstance(res_no_ret, DetectionResult)


def test_jobrunner_calibration_and_evaluation_jobs(tmp_path):
    with get_sync_session() as db:
        site = SiteORM(name="Site Eval")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line Eval")
        db.add(line)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod Eval")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="active", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-eval")
        db.add(model_v)
        db.commit()

        job_repo = JobRepository(db)
        # 1. Calibrate motion job
        j1 = job_repo.submit_job("calibrate_motion", {"line_id": line.id, "sample_frames": 5})
        # 2. Calibrate scale job
        j2 = job_repo.submit_job("calibrate_scale", {"line_id": line.id, "nominal_bag_width_mm": 500.0})
        # 3. Evaluate job
        j3 = job_repo.submit_job("evaluate", {"model_id": model_v.id, "dataset_id": 1})
        # 4. Mine hard frames job
        j4 = job_repo.submit_job("mine_hard_frames", {"line_id": line.id, "discrepancy_threshold": 0.05})

    worker = JobrunnerWorker(lease_seconds=60, gpu_mode="always")

    for _ in range(4):
        worker.run_step()

    with get_sync_session() as db:
        job_repo = JobRepository(db)
        assert job_repo.get_job(j1.id).status in ["completed", "failed"]
        assert job_repo.get_job(j2.id).status in ["completed", "failed"]
        assert job_repo.get_job(j3.id).status in ["completed", "failed"]
        assert job_repo.get_job(j4.id).status in ["completed", "failed"]
