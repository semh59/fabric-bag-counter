"""Tests that cls_head/print_mark classification gets a real trained signal.

Prior to this fix, both build_synthetic_training_dataset and
build_real_training_dataset always called match_boxes_to_anchor_grid()
without a `classes` argument, so every matched query's classification
target (t_classes) was hard-coded to 0.0 -- BCE against a constant target
is trivially "learnable" (always predict ~0) without encoding any real
information. Since VisionDetector consumes the model's classes output to
build DetectionResult.print_marks (packages/cs_vision/postprocess.py:
class_id 1 = print_mark), this meant the print-mark/logo signal exposed to
callers was never actually trained. These tests lock in the real fix:
SyntheticBagGenerator's has_print flag and CVAT's print_mark annotations
now flow into real classification targets.
"""

import json
import random

import numpy as np
import pytest
import torch

from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_vision.train_rfdetr import (
    anchor_grid,
    build_real_training_dataset,
    build_synthetic_training_dataset,
    match_boxes_to_anchor_grid,
)


def test_generate_scene_has_print_marks_aligned_with_amodal_boxes():
    random.seed(11)
    gen = SyntheticBagGenerator(canvas_size=(1400, 640), min_overlap_ratio=0.15, max_overlap_ratio=0.40)
    scene = gen.generate_scene(num_bags=6, bag_colors=[(220, 215, 200)])

    assert len(scene["has_print_marks"]) == len(scene["amodal_boxes"])
    assert all(isinstance(v, bool) for v in scene["has_print_marks"])


def test_generate_scene_has_print_marks_matches_print_mark_box_placement():
    """Where has_print_marks[i] is True, a print_marks box's center must
    actually fall inside that bag's amodal_box (and vice versa) -- the two
    fields must describe the same underlying bags, not just happen to have
    compatible types."""
    random.seed(23)
    gen = SyntheticBagGenerator(canvas_size=(1400, 640), min_overlap_ratio=0.15, max_overlap_ratio=0.40)
    scene = gen.generate_scene(num_bags=6, bag_colors=[(220, 215, 200)])

    print_mark_centers = [
        ((pm[0] + pm[2]) / 2.0, (pm[1] + pm[3]) / 2.0) for pm in scene["print_marks"]
    ]
    assert len(print_mark_centers) > 0, "test scene must contain at least one print mark to be meaningful"

    for box, has_print in zip(scene["amodal_boxes"], scene["has_print_marks"]):
        x1, y1, x2, y2 = box
        covered = any(x1 <= cx <= x2 and y1 <= cy <= y2 for cx, cy in print_mark_centers)
        assert covered == has_print


def test_match_boxes_to_anchor_grid_assigns_real_classes():
    anchors = anchor_grid()
    boxes = [[100.0, 100.0, 200.0, 250.0], [300.0, 100.0, 400.0, 250.0]]
    masks = [np.zeros((16, 16), dtype=bool), np.zeros((16, 16), dtype=bool)]

    _, t_scores, t_classes, _ = match_boxes_to_anchor_grid(anchors, boxes, masks, classes=[1.0, 0.0])

    matched_idx = torch.nonzero(t_scores > 0.5).squeeze(-1).tolist()
    assert len(matched_idx) == 2
    matched_classes = sorted(t_classes[matched_idx].tolist())
    assert matched_classes == [0.0, 1.0]


def test_build_synthetic_training_dataset_produces_nontrivial_class_signal():
    """Regression guard for the always-0 bug: across a modest sample, at
    least one matched (positive) query must have t_classes == 1.0 -- with
    the old code this was provably impossible regardless of seed."""
    random.seed(42)
    np.random.seed(42)
    dataset = build_synthetic_training_dataset(num_samples=40)

    saw_positive_class = False
    saw_zero_class = False
    for _img, _t_boxes, t_scores, t_classes, _t_masks in dataset:
        pos = t_scores > 0.5
        if pos.sum() == 0:
            continue
        pos_classes = t_classes[pos]
        if (pos_classes == 1.0).any():
            saw_positive_class = True
        if (pos_classes == 0.0).any():
            saw_zero_class = True

    assert saw_positive_class, "no matched bag ever got class=1.0 (print mark) -- signal is still constant"
    assert saw_zero_class, "no matched bag ever got class=0.0 (no print mark) -- signal is still constant"


def _write_coco_fixture(tmp_path, img_w=640, img_h=640):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    from PIL import Image
    Image.new("RGB", (img_w, img_h), (30, 30, 30)).save(images_dir / "frame_0.jpg")

    coco = {
        "images": [{"id": 1, "file_name": "frame_0.jpg", "width": img_w, "height": img_h}],
        "categories": [{"id": 1, "name": "bag_body"}, {"id": 2, "name": "print_mark"}],
        "annotations": [
            # Bag A: has a print_mark centered inside it.
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 150, 200], "segmentation": []},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [90, 100, 40, 30]},
            # Bag B: no print_mark anywhere near it.
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [350, 50, 150, 200], "segmentation": []},
        ],
    }
    ann_path = tmp_path / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return tmp_path


def test_build_real_training_dataset_derives_class_from_print_mark_annotations(tmp_path):
    data_dir = _write_coco_fixture(tmp_path)
    dataset = build_real_training_dataset(data_dir)

    assert len(dataset) == 1
    _img, _t_boxes, t_scores, t_classes, _t_masks = dataset[0]
    pos = t_scores > 0.5
    assert pos.sum() == 2

    pos_classes = sorted(t_classes[pos].tolist())
    assert pos_classes == [0.0, 1.0]
