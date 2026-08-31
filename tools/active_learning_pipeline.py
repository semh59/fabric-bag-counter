"""Active Learning & CVAT Ingestion CLI Pipeline (§6.3, §6.4).

Connects video frame extraction, hard-frame mining, CVAT task provisioning,
dataset synchronization into data/real_bags, and RF-DETR fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from packages.cs_data.cvat_client import CvatClient, CvatApiError
from packages.cs_data.extract_frames import extract_video_frames
from packages.cs_data.mining import HardFrameMiner
from packages.cs_vision.train_rfdetr import train_and_export_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ActiveLearningPipeline")


def extract_and_mine(video_path: str, output_dir: str = "./data/extracted_frames", stride: int = 5) -> list[str]:
    """Extract periodic frames from conveyor video and save to disk."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting frames from {video_path} (stride={stride})...")
    frames = extract_video_frames(video_path, output_dir=output_dir, stride_frames=stride)
    logger.info(f"Extracted {len(frames)} candidate frames to {output_dir}")
    return frames


def push_to_cvat(
    frame_paths: list[str],
    task_name: str,
    cvat_url: str = "http://localhost:8088/api",
    auth_token: str | None = None,
) -> int | None:
    """Create CVAT task with standard 2-class specification and upload frame batch."""
    client = CvatClient(base_url=cvat_url, auth_token=auth_token)
    logger.info(f"Connecting to CVAT at {cvat_url}...")

    try:
        task = client.create_task(name=task_name)
        task_id = int(task["id"])
        logger.info(f"Created CVAT Task #{task_id}: '{task_name}'")

        if frame_paths:
            logger.info(f"Uploading {len(frame_paths)} frames to CVAT Task #{task_id}...")
            client.upload_task_data(task_id=task_id, image_paths=frame_paths)
            logger.info(f"Successfully uploaded frames to Task #{task_id}")
        return task_id
    except CvatApiError as exc:
        logger.error(f"CVAT API Error: {exc}")
        return None
    except Exception as exc:
        logger.error(f"Could not connect to CVAT: {exc}")
        return None


def sync_and_train(real_data_dir: str = "./data/real_bags", epochs: int = 25, num_synthetic: int = 300) -> str:
    """Verify data/real_bags structure and run hybrid synthetic+real fine-tuning."""
    real_p = Path(real_data_dir)
    ann_path = real_p / "annotations.json"

    if not ann_path.exists():
        logger.warning(f"No annotations.json found in {real_data_dir}. Fine-tuning will proceed with synthetic pretraining.")
    else:
        logger.info(f"Found real annotations in {ann_path}. Blending with synthetic conveyor scenes...")

    logger.info(f"Starting RF-DETR Training: epochs={epochs}, synthetic_scenes={num_synthetic}")
    onnx_output = train_and_export_model(
        epochs=epochs,
        num_synthetic_scenes=num_synthetic,
        real_data_dir=real_data_dir if ann_path.exists() else None,
    )
    logger.info(f"Training and export complete: {onnx_output}")
    return onnx_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fabric Bag Counter Active Learning Pipeline")
    parser.add_argument("--video", type=str, help="Path to input video file (MP4/AVI)")
    parser.add_argument("--stride", type=int, default=10, help="Frame extraction stride")
    parser.add_argument("--cvat-url", type=str, default="http://localhost:8088/api", help="CVAT REST API URL")
    parser.add_argument("--cvat-token", type=str, default=None, help="CVAT Token auth (optional)")
    parser.add_argument("--task-name", type=str, default=None, help="CVAT Task name")
    parser.add_argument("--train-only", action="store_true", help="Skip extraction and train directly with data/real_bags")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--synthetic-count", type=int, default=200, help="Synthetic scene count")
    args = parser.parse_args()

    if args.train_only:
        sync_and_train(epochs=args.epochs, num_synthetic=args.synthetic_count)
        return

    if not args.video:
        logger.error("Please provide --video or --train-only")
        parser.print_help()
        sys.exit(1)

    # 1. Extract frames
    frames = extract_and_mine(args.video, stride=args.stride)

    # 2. Push to CVAT
    t_name = args.task_name or f"ActiveLearning_{Path(args.video).stem}_{int(time.time())}"
    task_id = push_to_cvat(frames, task_name=t_name, cvat_url=args.cvat_url, auth_token=args.cvat_token)

    if task_id:
        print("\n" + "=" * 65)
        print(f"  [SUCCESS] Task created in CVAT: Task #{task_id}")
        print(f"  Open Web Dashboard or CVAT UI at: {args.cvat_url.replace('/api', '')}")
        print("  Annotate bags -> Export COCO 1.0 -> Save into data/real_bags/")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
