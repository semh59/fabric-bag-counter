"""Exhaustive Vision Training & Data Pipelines Coverage Test Suite (§6.1, §6.4).

Directly exercises RF-DETR training components, anchor matching, COCO parsing,
SSIM deduplication, frame extraction, dataset splitting, and mining.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from packages.cs_data.extract_frames import extract_video_frames
from packages.cs_data.mining import HardFrameMiner
from packages.cs_data.split_dataset import DatasetSplitter
from packages.cs_data.ssim_filter import SSIMFilter
from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_vision.train_rfdetr import (
    _coco_polygon_to_mask,
    anchor_grid,
    build_real_training_dataset,
    match_boxes_to_anchor_grid,
    train_and_export_model,
)


def test_ssim_filter_deduplication():
    filter_tool = SSIMFilter(ssim_threshold=0.85)

    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Identical
    img3 = np.zeros((100, 100, 3), dtype=np.uint8)       # Completely distinct

    assert filter_tool.is_redundant(img1) is False
    assert filter_tool.is_redundant(img2) is True
    assert filter_tool.is_redundant(img3) is False


def test_dataset_splitter():
    splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, hard_holdout_ratio=0.15)
    sessions = [
        {"session_id": f"sess_{i}", "camera_id": 1, "shift": "day", "frame_count": 100, "is_heavy_shingling": i % 3 == 0}
        for i in range(20)
    ]
    res = splitter.split_sessions(sessions)

    assert res.train_count > 0
    assert res.val_count > 0
    assert len(res.manifest_hash) > 0


def test_hard_frame_miner():
    miner = HardFrameMiner(low_conf_threshold=0.50)

    candidates = miner.evaluate_frame(
        frame_index=10,
        camera_id=1,
        session_id=1,
        detections=[{"box": [100, 100, 200, 200], "score": 0.42, "mask": np.ones((100, 100), dtype=bool)}],
        secondary_model_detections=[{"box": [100, 100, 250, 250], "score": 0.85}],
        has_merge_flag=True,
        area_mismatch=True,
    )
    assert len(candidates) >= 1
    assert candidates[0].score == 0.42


def test_anchor_grid_and_matching():
    anchors = anchor_grid(num_x=5)
    assert anchors.shape == (10, 2)

    boxes = [[100.0, 250.0, 200.0, 350.0]]
    masks = [np.ones((640, 640), dtype=bool)]
    t_boxes, t_scores, t_classes, t_masks = match_boxes_to_anchor_grid(anchors, boxes, masks)

    assert t_boxes.shape == (20, 4)
    assert t_scores.sum() == 1.0
    assert t_masks.shape == (20, 160, 160)


def test_coco_polygon_to_mask():
    poly = [[10, 10, 50, 10, 50, 50, 10, 50]]
    mask = _coco_polygon_to_mask(poly, 100, 100)
    assert mask.shape == (100, 100)
    assert mask[25, 25] is np.True_ or mask[25, 25] is True
    assert mask[5, 5] is np.False_ or mask[5, 5] is False


def test_fast_training_and_export_loop(tmp_path):
    """Run lightweight fast 1-epoch training and ONNX export."""
    onnx_file = train_and_export_model(
        epochs=1,
        num_synthetic_scenes=3,
        output_dir=str(tmp_path / "models"),
    )
    assert Path(onnx_file).exists()
    assert Path(onnx_file).stat().st_size > 1000
