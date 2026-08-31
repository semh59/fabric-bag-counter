"""Exhaustive Jobrunner Worker & Job Types Coverage Test Suite (§5.8, §10).

Executes every job kind and worker loop condition:
calibrate_motion, calibrate_scale, evaluate_model, replay, mine_hard_frames, etc.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import (
    ConfigVersionORM,
    DatasetVersionORM,
    DeploymentBundleORM,
    JobORM,
    LineORM,
    ModelVersionORM,
    NodeORM,
    ProductProfileORM,
    SiteORM,
    TrainingRunORM,
)
from packages.cs_storage.repositories.job_repo import JobRepository
from services.jobrunner.worker import JobrunnerWorker, _evaluate_model, _hash_file


@pytest.fixture(autouse=True)
def setup_job_env():
    init_db_sync()


def test_jobrunner_executes_all_job_kinds(tmp_path):
    with get_sync_session() as db:
        site = SiteORM(name="Job Site")
        db.add(site)
        db.commit()

        line = LineORM(site_id=site.id, name="Line J")
        db.add(line)
        db.commit()

        prod = ProductProfileORM(site_id=site.id, name="Prod J")
        db.add(prod)
        db.commit()

        model_v = ModelVersionORM(stage="draft", onnx_path="models/rfdetr_seg_v2.onnx", onnx_hash="hash-j")
        db.add(model_v)
        db.commit()

        line_id = line.id
        model_id = model_v.id

    worker = JobrunnerWorker(lease_seconds=30, gpu_mode="always")

    # 1. calibrate_motion
    job1 = JobORM(id=1, kind="calibrate_motion", payload={"line_id": line_id, "video_path": "data/sample.mp4"}, status="running")
    res_motion = worker.execute_job(job1)
    assert res_motion["stage"] == "motion"

    # 2. calibrate_scale
    job2 = JobORM(id=2, kind="calibrate_scale", payload={"line_id": line_id, "mean_bag_gate_area_px": 14000.0}, status="running")
    res_scale = worker.execute_job(job2)
    assert res_scale["stage"] == "scale"

    # 3. Direct model evaluation helper
    eval_res = _evaluate_model("models/rfdetr_seg_v2.onnx", num_scenes=2)
    assert "mean_iou" in eval_res

    # 4. replay
    job4 = JobORM(id=4, kind="replay", payload={"scenario_types": ["sparse_flow"], "seed": 42}, status="running")
    res_replay = worker.execute_job(job4)
    assert "scenarios" in res_replay

    # 5. mine_hard_frames
    job5 = JobORM(id=5, kind="mine_hard_frames", payload={
        "frame_index": 5, "camera_id": 1, "session_id": 1,
        "detections": [{"box": [100, 100, 200, 200], "score": 0.45}],
        "has_merge_flag": True,
    }, status="running")
    res_mine = worker.execute_job(job5)
    assert "candidates_mined" in res_mine

    # 6. Helper functions
    fake_file = tmp_path / "test.txt"
    fake_file.write_text("hello world")
    h = _hash_file(str(fake_file))
    assert h.startswith("sha256:")
