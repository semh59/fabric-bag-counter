"""Model Accuracy, Sanity Checks, and Security Verification Tests (§6.2, §6.3, §8.1, §8.2).

Evaluates RF-DETR Seg ONNX model accuracy on realistic overlapping bag scenes,
verifies genuine discrimination between empty and full conveyor states,
and verifies cryptographic JWT and bcrypt security guarantees.
"""

import os
import random
import pytest
import numpy as np
from pathlib import Path

from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_vision.detector import VisionDetector
from services.api.auth import CurrentUser, create_access_token, get_current_user, SECRET_KEY
from packages.cs_storage.repositories.user_repo import hash_password, verify_password


def compute_box_iou(boxA: list[float] | np.ndarray, boxB: list[float] | np.ndarray) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def test_rfdetr_seg_accuracy_on_overlapping_bags():
    """Test model detection and segmentation accuracy on realistic overlapping bag scenes.
    
    Computes genuine bag count error and mean bounding box IoU (§6.2, §6.3).
    """
    model_path = str(Path(__file__).resolve().parent.parent / "models" / "rfdetr_seg_v2.onnx")
    assert os.path.exists(model_path), f"Trained ONNX model missing at {model_path}"

    detector = VisionDetector(model_path=model_path, conf_threshold=0.35, allow_fallback=False)
    assert detector.session is not None, "ONNX Runtime inference session must be active"

    # Fixed seed: scene composition (packages.cs_data.synth) draws from the
    # global `random` module, so an unseeded run makes the measured mean
    # count error/IoU non-reproducible between test runs.
    random.seed(1234)
    np.random.seed(1234)

    gen = SyntheticBagGenerator(min_overlap_ratio=0.15, max_overlap_ratio=0.40)

    # 60 scenes (not 20): the mean count error is a discrete-valued average,
    # so a small sample keeps enough sampling noise to flip pass/fail run to
    # run even for a fixed underlying model.
    total_eval_scenes = 60
    count_errors = []
    matched_ious = []

    for _ in range(total_eval_scenes):
        num_target_bags = np.random.randint(1, 4)
        scene = gen.generate_scene(num_bags=num_target_bags)
        image = scene["image"]
        gt_boxes = scene["amodal_boxes"]

        result = detector.predict(image)
        pred_boxes = [bag["box"] for bag in result.bag_bodies]

        # Bag count error
        count_err = abs(len(pred_boxes) - len(gt_boxes))
        count_errors.append(count_err)

        # Match each ground truth box with best predicted box
        for gt in gt_boxes:
            best_iou = 0.0
            for pred in pred_boxes:
                iou = compute_box_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
            matched_ious.append(best_iou)

    mean_count_error = float(np.mean(count_errors))
    mean_iou = float(np.mean(matched_ious))
    exact_count_acc = float(np.mean(np.array(count_errors) == 0))

    print(f"\n[Test Result] RF-DETR Seg Real Accuracy over {total_eval_scenes} scenes:")
    print(f"              Mean Bag Count Error: {mean_count_error:.2f} bags")
    print(f"              Exact Count Accuracy: {exact_count_acc * 100:.1f}%")
    print(f"              Mean Bounding Box IoU: {mean_iou:.3f}")

    assert mean_count_error <= 0.80, f"Mean count error {mean_count_error:.2f} exceeds threshold 0.80"
    assert mean_iou >= 0.40, f"Mean IoU {mean_iou:.3f} below 0.40 threshold"


def test_model_outputs_differ_on_empty_vs_full_conveyor():
    """Model Sanity Check: Model must produce distinct outputs on empty vs full conveyor.
    
    Empty conveyor must yield 0 detections and low scores.
    Full conveyor must yield localized detections with high confidence.
    """
    model_path = str(Path(__file__).resolve().parent.parent / "models" / "rfdetr_seg_v2.onnx")
    detector = VisionDetector(model_path=model_path, conf_threshold=0.35, allow_fallback=False)

    random.seed(1234)
    np.random.seed(1234)
    gen = SyntheticBagGenerator()

    # 1. Empty conveyor
    empty_image = np.array(gen.create_empty_conveyor())
    res_empty = detector.predict(empty_image)
    assert len(res_empty.bag_bodies) == 0, f"Expected 0 bags on empty conveyor, got {len(res_empty.bag_bodies)}"

    # 2. Conveyor with bags
    scene_full = gen.generate_scene(num_bags=2)
    res_full = detector.predict(scene_full["image"])
    assert len(res_full.bag_bodies) >= 1, f"Expected at least 1 bag on full conveyor, got {len(res_full.bag_bodies)}"

    # 3. Direct confidence comparison
    from packages.cs_vision.preprocess import preprocess_image
    blob_e, _, _ = preprocess_image(empty_image, (640, 640))
    blob_f, _, _ = preprocess_image(scene_full["image"], (640, 640))

    scores_empty = detector.session.run(None, {"images": blob_e})[1][0]
    scores_full = detector.session.run(None, {"images": blob_f})[1][0]

    max_empty_score = float(np.max(scores_empty))
    max_full_score = float(np.max(scores_full))

    print(f"\n[Sanity Check] Empty Max Score: {max_empty_score:.4f} | Full Max Score: {max_full_score:.4f}")
    assert max_empty_score < 0.20, f"Empty conveyor max score {max_empty_score:.4f} is too high"
    assert max_full_score > 0.50, f"Full conveyor max score {max_full_score:.4f} is too low"
    assert max_full_score > (max_empty_score + 0.30), "Full conveyor confidence must clearly exceed empty conveyor"


def test_security_jwt_tampering_rejected():
    """Security verification: arbitrary/fake token strings MUST NOT grant authentication (§8.1)."""
    # 1. Fake split token string (e.g. "1:admin:admin:x") must fail signature verification
    fake_token = "1:admin:admin:x"
    with pytest.raises(Exception):
        get_current_user(authorization=f"Bearer {fake_token}", session_token=None)

    # 2. Tampered JWT signature must fail
    valid_token = create_access_token(data={"sub": "1", "username": "admin", "role": "admin"})
    tampered_token = valid_token[:-4] + "abcd"
    with pytest.raises(Exception):
        get_current_user(authorization=f"Bearer {tampered_token}", session_token=None)

    # 3. Valid signed token must succeed
    user = get_current_user(authorization=f"Bearer {valid_token}", session_token=None)
    assert user.username == "admin"
    assert user.role.value == "admin"


def test_security_bcrypt_unique_salts():
    """Security verification: bcrypt hashes must have unique per-user salts (§8.1)."""
    password = "SuperSecretPassword2026!"
    hash1 = hash_password(password)
    hash2 = hash_password(password)

    # Hashes must differ because of unique random salts
    assert hash1 != hash2, "Bcrypt must generate unique salts for each hash operation"
    assert verify_password(password, hash1) is True
    assert verify_password(password, hash2) is True
    assert verify_password("WrongPassword", hash1) is False


def test_evaluate_model_synthetic_fallback():
    """Verify _evaluate_model runs synthetic evaluation when real annotations are missing."""
    import os
    import sys
    from pathlib import Path
    from services.jobrunner.worker import _evaluate_model

    model_path = str(Path(__file__).resolve().parent.parent / "models" / "rfdetr_seg_v2.onnx")
    assert os.path.exists(model_path)

    res = _evaluate_model(model_path, num_scenes=3, real_data_dir="/non_existent_path_xyz")
    assert res["dataset_type"] == "synthetic"
    assert res["eval_scenes"] == 3
    assert "mean_count_error" in res
    assert "mean_iou" in res
    assert "fps" in res


def test_evaluate_model_real_holdout(tmp_path):
    """Verify _evaluate_model correctly evaluates on genuine holdout dataset when present."""
    import json
    import os
    from pathlib import Path
    from PIL import Image
    from services.jobrunner.worker import _evaluate_model

    model_path = str(Path(__file__).resolve().parent.parent / "models" / "rfdetr_seg_v2.onnx")
    assert os.path.exists(model_path)

    # Setup temporary real dataset structure
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    img = Image.new("RGB", (640, 640), (120, 120, 120))
    img.save(images_dir / "test_frame.jpg")

    coco_data = {
        "images": [{"id": 1, "file_name": "test_frame.jpg", "width": 640, "height": 640}],
        "categories": [{"id": 1, "name": "bag_body"}, {"id": 2, "name": "print_mark"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "segmentation": []},
        ],
    }
    with open(tmp_path / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    res = _evaluate_model(model_path, real_data_dir=str(tmp_path))
    assert res["dataset_type"] == "real_holdout"
    assert res["eval_scenes"] == 1
    assert "mean_count_error" in res
    assert "mean_iou" in res
    assert "fps" in res

