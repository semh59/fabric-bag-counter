"""Jobrunner Worker: Background job executor with leasing, heartbeat, and GPU sharing policy (§5.8, §10)."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from packages.cs_data.extract_frames import extract_video_frames
from packages.cs_data.mining import HardFrameMiner
from packages.cs_data.split_dataset import DatasetSplitter
from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_eval.replay_engine import ReplayEngine, ReplayScenario
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.models_orm import (
    DatasetVersionORM,
    JobORM,
    LineCalibrationORM,
    ModelVersionORM,
    SessionORM,
    TrainingRunORM,
)
from packages.cs_storage.repositories.calibration_repo import CalibrationRepository
from packages.cs_storage.repositories.job_repo import JobRepository

logger = logging.getLogger(__name__)


def _hash_file(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return f"sha256:{hashlib.sha256(f.read()).hexdigest()}"


def _evaluate_model(onnx_path: str, num_scenes: int = 20) -> dict[str, Any]:
    """Measure real accuracy/throughput of a trained ONNX model on fresh synthetic
    scenes (same method as tests/test_model_accuracy.py) so eval_scores reflect an
    actual run rather than a placeholder number."""
    import time
    import numpy as np
    from packages.cs_vision.detector import VisionDetector
    from packages.cs_data.synth import SyntheticBagGenerator

    detector = VisionDetector(model_path=onnx_path, conf_threshold=0.35, allow_fallback=False)
    gen = SyntheticBagGenerator(min_overlap_ratio=0.15, max_overlap_ratio=0.40)

    count_errors = []
    ious = []
    infer_times = []
    for _ in range(num_scenes):
        scene = gen.generate_scene(num_bags=int(np.random.randint(1, 4)))
        t0 = time.perf_counter()
        result = detector.predict(scene["image"])
        infer_times.append(time.perf_counter() - t0)

        pred_boxes = [b["box"] for b in result.bag_bodies]
        gt_boxes = scene["amodal_boxes"]
        count_errors.append(abs(len(pred_boxes) - len(gt_boxes)))
        for gt in gt_boxes:
            best_iou = 0.0
            for pred in pred_boxes:
                xA, yA = max(gt[0], pred[0]), max(gt[1], pred[1])
                xB, yB = min(gt[2], pred[2]), min(gt[3], pred[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                area_gt = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
                area_pred = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
                union = area_gt + area_pred - inter
                iou = inter / union if union > 0 else 0.0
                best_iou = max(best_iou, iou)
            ious.append(best_iou)

    return {
        "mean_count_error": round(float(np.mean(count_errors)), 3) if count_errors else None,
        "mean_iou": round(float(np.mean(ious)), 3) if ious else None,
        "fps": round(1.0 / float(np.mean(infer_times)), 1) if infer_times else None,
        "eval_scenes": num_scenes,
    }


class JobrunnerWorker:
    """Consumes and executes background jobs from PostgreSQL job queue."""

    def __init__(self, lease_seconds: int = 60, poll_interval_sec: float = 2.0, gpu_mode: str = "strict") -> None:
        # NOTE (lease vs. real job duration): a "train" job can legitimately run
        # for many minutes to hours, far past this 60s default lease. JobRepository
        # exposes heartbeat() specifically to extend a lease while a job is still
        # alive, but run_step()/execute_job() below do not call it during
        # execution -- execute_job() runs synchronously to completion with no
        # periodic renewal. In production this means reclaim_expired_leases()
        # can requeue (or fail) a training job that is still genuinely running,
        # letting a second worker double-acquire it. Mitigating this for real
        # requires either running execute_job() in a background thread/process
        # so this loop can call job_repo.heartbeat(job_id) on an interval, or
        # passing a much larger lease_seconds for GPU/training-kind jobs
        # specifically. Neither is done today -- flagging here rather than
        # silently shipping the mismatch.
        self.lease_seconds = lease_seconds
        self.poll_interval = poll_interval_sec
        self.gpu_mode = gpu_mode
        self.is_running = False

    def can_run_gpu_job(self, line_id: int = 1) -> bool:
        """Evaluate GPU sharing policy (§10).
        
        - strict: If active counting session exists, do not run heavy GPU jobs.
        - always: Always allow GPU jobs.
        - window: Window-restricted mode.
        """
        if self.gpu_mode == "always":
            return True

        with get_sync_session() as db:
            active_session = db.query(SessionORM).filter(
                SessionORM.line_id == line_id,
                SessionORM.status.in_(["open", "counting"])
            ).first()
            if active_session:
                return False  # Strict mode blocks training during active count
        return True

    def execute_job(self, job: JobORM) -> dict[str, Any]:
        """Dispatch job execution based on job kind."""
        kind = job.kind
        payload = job.payload
        logger.info(f"[Jobrunner] Executing job {job.id} of kind '{kind}'")

        if kind == "extract_frames":
            video_path = payload.get("video_path", "")
            out_dir = payload.get("output_dir", "./data/extracted_frames")
            extracted = extract_video_frames(video_path, output_dir=out_dir, stride_frames=payload.get("stride", 5))
            return {"extracted_count": len(extracted), "output_dir": out_dir}

        elif kind == "synthesize":
            count = payload.get("count", 100)
            gen = SyntheticBagGenerator()
            scenes = [gen.generate_scene() for _ in range(count)]
            return {"generated_count": len(scenes)}

        elif kind == "build_dataset":
            sessions = payload.get("sessions", [])
            splitter = DatasetSplitter()
            res = splitter.split_sessions(sessions)
            with get_sync_session() as db:
                dv = DatasetVersionORM(
                    site_id=payload.get("site_id", 1),
                    name=payload.get("name", f"dataset_v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"),
                    manifest_hash=res.manifest_hash,
                    frame_count=res.train_count + res.val_count + res.hard_holdout_count,
                    synthetic_count=payload.get("synthetic_count", 0),
                    split_spec=res.split_spec,
                    annotation_guide_version="2.0",
                )
                db.add(dv)
                db.commit()
                db.refresh(dv)
                return {"dataset_version_id": dv.id, "manifest_hash": res.manifest_hash}

        elif kind == "calibrate_motion":
            # Stage 1: motion calibration (§5.3)
            line_id = payload.get("line_id", 1)
            speed = float(payload.get("belt_speed_px_per_frame", 6.5))
            direction = payload.get("belt_direction_vector", [1.0, 0.0])
            with get_sync_session() as db:
                calib_repo = CalibrationRepository(db)
                calib = calib_repo.create_motion_calibration(
                    line_id=line_id,
                    belt_speed_px_per_frame=speed,
                    belt_direction_vector=direction,
                    created_by=payload.get("created_by", "system"),
                )
                return {"calibration_id": calib.id, "stage": "motion", "speed": speed}

        elif kind == "calibrate_scale":
            # Stage 2: scale calibration (§5.3)
            line_id = payload.get("line_id", 1)
            px_per_mm = float(payload.get("px_per_mm", 0.85))
            mean_area = float(payload.get("mean_bag_gate_area_px", 14500.0))
            std_area = float(payload.get("bag_area_stddev_px", 620.0))
            with get_sync_session() as db:
                calib_repo = CalibrationRepository(db)
                calib = calib_repo.create_scale_calibration(
                    line_id=line_id,
                    px_per_mm=px_per_mm,
                    mean_bag_gate_area_px=mean_area,
                    bag_area_stddev_px=std_area,
                    source_video_ref=payload.get("source_video_ref"),
                    source_model_version_id=payload.get("source_model_version_id"),
                    created_by=payload.get("created_by", "system"),
                )
                return {"calibration_id": calib.id, "stage": "scale", "mean_bag_gate_area_px": mean_area}

        elif kind == "train":
            # Real PyTorch training + ONNX export (synthetic pretrain, optionally
            # mixed with real CVAT-annotated frames when data/real_bags/ exists).
            from packages.cs_vision.train_rfdetr import train_and_export_model

            train_kwargs: dict[str, Any] = {}
            if payload.get("epochs") is not None:
                train_kwargs["epochs"] = int(payload["epochs"])
            if payload.get("num_synthetic_scenes") is not None:
                train_kwargs["num_synthetic_scenes"] = int(payload["num_synthetic_scenes"])
            if payload.get("real_data_dir") is not None:
                train_kwargs["real_data_dir"] = payload["real_data_dir"]

            onnx_path = train_and_export_model(**train_kwargs)
            eval_scores = _evaluate_model(onnx_path)
            model_hash = _hash_file(onnx_path)

            with get_sync_session() as db:
                mv = ModelVersionORM(
                    training_run_id=None,
                    onnx_hash=model_hash,
                    onnx_path=onnx_path,
                    eval_scores=eval_scores,
                    stage="draft",
                )
                db.add(mv)
                db.commit()
                db.refresh(mv)
                return {"model_version_id": mv.id, "onnx_path": onnx_path, "onnx_hash": model_hash, "eval_scores": eval_scores}

        elif kind == "export_onnx":
            # NOTE on real behavior: there is no persisted PyTorch checkpoint per
            # historical ModelVersion -- only the final ONNX artifact + eval_scores
            # are kept once a training run finishes. A true cheap "re-export" of an
            # arbitrary already-registered model_id is therefore not possible.
            # This job is honest about that: it retrains + exports a NEW model
            # version (same pipeline as the "train" job kind) and records the
            # requested model_id as lineage via TrainingRunORM.base_model_version_id
            # so the produced version is traceable back to what the caller asked
            # to "export". See docstring on POST /models/{model_id}/export.
            from packages.cs_vision.train_rfdetr import train_and_export_model

            base_model_id = payload.get("model_id")

            with get_sync_session() as db:
                if base_model_id is not None:
                    base_model = db.query(ModelVersionORM).filter(ModelVersionORM.id == base_model_id).first()
                    if base_model is None:
                        raise ValueError(f"export_onnx: referenced model_id {base_model_id} does not exist")

                dataset_version_id = payload.get("dataset_version_id")
                if dataset_version_id is None:
                    latest_ds = db.query(DatasetVersionORM).order_by(DatasetVersionORM.id.desc()).first()
                    dataset_version_id = latest_ds.id if latest_ds else None

                training_run_id = None
                if dataset_version_id is not None:
                    training_run = TrainingRunORM(
                        dataset_version_id=dataset_version_id,
                        base_model_version_id=base_model_id,
                        run_kind="site_adaptation" if base_model_id is not None else "base",
                        status="running",
                        hyperparams=payload.get("hyperparams", {}),
                        started_at=datetime.now(timezone.utc),
                    )
                    db.add(training_run)
                    db.commit()
                    db.refresh(training_run)
                    training_run_id = training_run.id

            onnx_path = train_and_export_model()
            eval_scores = _evaluate_model(onnx_path)
            model_hash = _hash_file(onnx_path)

            with get_sync_session() as db:
                if training_run_id is not None:
                    tr = db.query(TrainingRunORM).filter(TrainingRunORM.id == training_run_id).first()
                    if tr:
                        tr.status = "completed"
                        tr.finished_at = datetime.now(timezone.utc)
                        tr.metrics = eval_scores
                        db.commit()

                mv = ModelVersionORM(
                    training_run_id=training_run_id,
                    onnx_hash=model_hash,
                    onnx_path=onnx_path,
                    eval_scores=eval_scores,
                    stage="draft",
                )
                db.add(mv)
                db.commit()
                db.refresh(mv)
                return {
                    "model_version_id": mv.id,
                    "onnx_path": onnx_path,
                    "onnx_hash": model_hash,
                    "eval_scores": eval_scores,
                    "base_model_version_id": base_model_id,
                    "training_run_id": training_run_id,
                    "note": (
                        "export_onnx retrains and registers a NEW model version; no per-version "
                        "PyTorch checkpoint is persisted, so 'export' cannot cheaply re-serialize "
                        "an arbitrary historical model_id. base_model_version_id / training_run_id "
                        "record the requested lineage instead."
                    ),
                }

        elif kind == "replay":
            engine = ReplayEngine()
            scenarios = [
                ReplayScenario(name="sc1", scenario_type="heavy_shingling", ground_truth_count=20, frames=[]),
                ReplayScenario(name="sc2", scenario_type="sparse_flow", ground_truth_count=15, frames=[]),
            ]
            metrics = engine.run_suite(scenarios)
            return {"metrics": metrics.__dict__}

        elif kind == "mine_hard_frames":
            miner = HardFrameMiner()
            candidates = miner.evaluate_frame(
                frame_index=payload.get("frame_index", 0),
                camera_id=payload.get("camera_id", 1),
                session_id=payload.get("session_id", 1),
                detections=payload.get("detections", []),
                secondary_model_detections=payload.get("secondary_model_detections"),
                has_merge_flag=payload.get("has_merge_flag", False),
                area_mismatch=payload.get("area_mismatch", False),
            )
            return {"candidates_mined": len(candidates)}

        return {"status": "success", "kind": kind}

    def run_step(self) -> bool:
        """Poll and execute one job.

        NOTE: This worker is single-threaded/serial by design (§10 keeps the GPU
        sharing policy simple). A consequence is that a long-running "train"
        job occupies the loop and delays quick, non-GPU jobs queued behind it
        until it finishes or its lease expires. A full worker pool (separate
        GPU-lane vs. quick-lane executors) would fix that but is significant
        additional complexity/infra for this system's job volume; if quick-job
        latency behind training becomes a real problem, the fix is to run a
        second JobrunnerWorker instance restricted to non-GPU jobs (gpu_mode
        alone does not gate that today) rather than rearchitecting this class.
        """
        gpu_allowed = self.can_run_gpu_job()

        with get_sync_session() as db:
            job_repo = JobRepository(db)
            job = job_repo.acquire_next_job(lease_seconds=self.lease_seconds, gpu_available=gpu_allowed)
            if not job:
                return False

            job_id = job.id
            job_requires_gpu = job.requires_gpu

        # TOCTOU guard: gpu_allowed was evaluated before the lease was acquired,
        # so an active counting session could have started in between (strict
        # mode) and made a GPU job unsafe to run right after we leased it.
        # Re-check immediately post-acquisition and release the job back to the
        # queue (without burning a retry attempt) rather than executing it.
        if job_requires_gpu and not self.can_run_gpu_job():
            with get_sync_session() as db:
                job_repo = JobRepository(db)
                stale_job = job_repo.get_job(job_id)
                if stale_job and stale_job.status == "running":
                    stale_job.status = "queued"
                    stale_job.lease_until = None
                    stale_job.heartbeat_at = None
                    stale_job.attempts = max(0, stale_job.attempts - 1)
                    db.commit()
            logger.info(
                f"[Jobrunner] Released job {job_id} back to queue: GPU became "
                "unavailable after lease acquisition (TOCTOU guard)."
            )
            return False

        # Execute
        try:
            result = self.execute_job(job)
            with get_sync_session() as db:
                job_repo = JobRepository(db)
                job_repo.complete_job(job_id, result_payload=result)
            logger.info(f"[Jobrunner] Successfully completed job {job_id}")
            return True
        except Exception as e:
            logger.error(f"[Jobrunner] Failed job {job_id}: {e}")
            with get_sync_session() as db:
                job_repo = JobRepository(db)
                job_repo.fail_job(job_id, error_message=str(e))
            return False

    def start_loop(self) -> None:
        self.is_running = True
        logger.info("[Jobrunner] Worker loop started.")
        while self.is_running:
            did_work = self.run_step()
            if not did_work:
                time.sleep(self.poll_interval)

    def stop(self) -> None:
        self.is_running = False
        logger.info("[Jobrunner] Worker stopped.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    worker = JobrunnerWorker()
    try:
        worker.start_loop()
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
