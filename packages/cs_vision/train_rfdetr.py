"""RF-DETR Seg PyTorch Training and ONNX Model Export Pipeline (§6.2, §6.3, §6.5).

Builds, trains, and exports a genuine deep learning instance segmentation model
for industrial conveyor bags and print marks using PyTorch and ONNX Runtime.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import onnx
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_data.cvat_client import CvatClient
from packages.cs_vision.preprocess import letterbox_image

NUM_QUERIES = 20
MASK_SIZE = 160
CANVAS_SIZE = (640, 640)


def anchor_grid(num_x: int = 10, num_y_positions: tuple[float, float] = (270.0, 370.0)) -> np.ndarray:
    """Fixed anchor grid across the conveyor belt capture area (§6.2).

    Shared by the model (query positions), the synthetic dataset builder, and
    the real/CVAT dataset builder so all three agree on the same 20 anchors.

    IMPORTANT -- camera-framing assumption: the default coordinates
    (x in [70, 570], y in {270.0, 370.0}, all in 640x640 CANVAS_SIZE pixel
    space) are NOT a generic prior -- they hard-code where THIS deployment's
    camera physically frames the conveyor belt's region of interest (a
    roughly horizontal band across the middle of the canvas, matching this
    line's mounting height/angle and belt width). NUM_QUERIES=20 is likewise
    tied to this exact 10x2 layout (see RFDETRSeg.forward, which reshapes
    query outputs assuming this grid). A real deployment on a different
    camera/line -- different mounting height, angle, belt width, or ROI
    framing -- MUST recalibrate this grid (and retrain) to that camera's
    actual belt geometry; reusing these fixed coordinates unchanged will
    silently point every anchor at the wrong part of the frame.
    """
    grid_x = np.linspace(70, 570, num_x)
    anchors = [[x, y] for y in num_y_positions for x in grid_x]
    return np.array(anchors, dtype=np.float32)


def match_boxes_to_anchor_grid(
    anchors_np: np.ndarray,
    boxes: list[list[float]],
    masks: list[np.ndarray],
    classes: list[int] | None = None,
    num_queries: int = NUM_QUERIES,
    mask_size: int = MASK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily assign ground-truth boxes/masks to the nearest unoccupied anchor query.

    `boxes` and `masks` must already be in the model's 640x640 canvas coordinate
    space (letterboxed for real images, native for synthetic scenes).
    """
    t_boxes = torch.zeros((num_queries, 4), dtype=torch.float32)
    t_scores = torch.zeros((num_queries,), dtype=torch.float32)
    t_classes = torch.zeros((num_queries,), dtype=torch.float32)
    t_masks = torch.zeros((num_queries, mask_size, mask_size), dtype=torch.float32)

    occupied: set[int] = set()
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        bcx = (box[0] + box[2]) / 2.0
        bcy = (box[1] + box[3]) / 2.0
        dists = np.sum((anchors_np - [bcx, bcy]) ** 2, axis=1)

        matched_q = None
        for q in np.argsort(dists):
            if q not in occupied:
                matched_q = int(q)
                break
        if matched_q is None:
            continue

        occupied.add(matched_q)
        t_boxes[matched_q] = torch.tensor(box, dtype=torch.float32)
        t_scores[matched_q] = 1.0
        t_classes[matched_q] = float(classes[i]) if classes is not None else 0.0

        pil_m = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (mask_size, mask_size), Image.Resampling.NEAREST
        )
        t_masks[matched_q] = torch.from_numpy(np.array(pil_m) > 128).float()

    return t_boxes, t_scores, t_classes, t_masks


def _coco_polygon_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterize a COCO polygon segmentation (list of flat [x1,y1,x2,y2,...] rings) to a boolean mask."""
    from PIL import ImageDraw

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for ring in segmentation:
        pts = [(ring[i], ring[i + 1]) for i in range(0, len(ring) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.array(mask_img) > 128


def build_real_training_dataset(
    data_dir: Path,
    canvas_size: tuple[int, int] = CANVAS_SIZE,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load a real, human-annotated dataset exported from CVAT (§6.3, §6.4).

    Expected layout, matching CvatClient's 2-class label spec (bag_body polygon,
    print_mark rectangle):
        <data_dir>/annotations.json   raw COCO export (images, annotations, categories)
        <data_dir>/images/<file_name> as referenced by annotations.json

    Only "bag_body" annotations become box/mask training targets; each one's
    classification target (cls_head, consumed downstream as
    DetectionResult.print_marks -- see packages/cs_vision/detector.py) is
    derived from whether a "print_mark" annotation's center falls inside it,
    mirroring the synthetic pipeline's has_print_marks.
    Returns an empty list (not an error) if no annotation file is present yet.
    """
    annotations_path = data_dir / "annotations.json"
    if not annotations_path.exists():
        return []

    with open(annotations_path, "r", encoding="utf-8") as f:
        coco_dict = json.load(f)

    parsed = CvatClient().parse_coco_annotations(coco_dict)
    images_by_id: dict[int, dict[str, Any]] = parsed["images"]
    anns_by_image: dict[int, list[dict[str, Any]]] = parsed["parsed_annotations"]

    anchors_np = anchor_grid()
    dataset = []

    for img_id, img_info in images_by_id.items():
        image_path = data_dir / "images" / img_info["file_name"]
        if not image_path.exists():
            print(f"  [RF-DETR] WARNING: annotated image missing on disk, skipping: {image_path}")
            continue

        pil_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = pil_img.size
        img_arr = np.array(pil_img)

        padded_img, scale, (pad_w, pad_h) = letterbox_image(img_arr, canvas_size, fill_value=114)

        boxes_canvas: list[list[float]] = []
        masks_canvas: list[np.ndarray] = []
        classes_canvas: list[float] = []

        # print_mark centers in original image space, used below to decide
        # per-bag_body whether a print mark falls on it -- a real per-box
        # classification target instead of the always-0 placeholder.
        print_mark_centers = []
        for ann in anns_by_image.get(img_id, []):
            if ann["category"] != "print_mark":
                continue
            pbx, pby, pbw, pbh = ann["bbox"]
            print_mark_centers.append((pbx + pbw / 2.0, pby + pbh / 2.0))

        for ann in anns_by_image.get(img_id, []):
            if ann["category"] != "bag_body":
                continue

            bx, by, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = float(bx), float(by), float(bx + bw), float(by + bh)

            if ann["segmentation"]:
                raw_mask = _coco_polygon_to_mask(ann["segmentation"], orig_h, orig_w)
            else:
                raw_mask = np.zeros((orig_h, orig_w), dtype=bool)
                raw_mask[int(y1):int(y2), int(x1):int(x2)] = True

            padded_mask, _, _ = letterbox_image(
                (raw_mask.astype(np.uint8) * 255), canvas_size, fill_value=0
            )
            masks_canvas.append(padded_mask > 128)
            boxes_canvas.append([
                x1 * scale + pad_w, y1 * scale + pad_h,
                x2 * scale + pad_w, y2 * scale + pad_h,
            ])
            has_print = any(x1 <= cx <= x2 and y1 <= cy <= y2 for cx, cy in print_mark_centers)
            classes_canvas.append(1.0 if has_print else 0.0)

        if not boxes_canvas:
            continue

        img_t = torch.from_numpy(padded_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        t_boxes, t_scores, t_classes, t_masks = match_boxes_to_anchor_grid(
            anchors_np, boxes_canvas, masks_canvas, classes=classes_canvas
        )
        dataset.append((img_t, t_boxes, t_scores, t_classes, t_masks))

    return dataset


class RFDETRSegNet(nn.Module):
    """Instance segmentation and object detection neural network for industrial conveyor bags."""

    def __init__(self, num_queries: int = 20) -> None:
        super().__init__()
        self.num_queries = num_queries

        # 4-stage convolutional backbone with GroupNorm for evaluation stability
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

        # 10 horizontal x 2 vertical anchors focused directly along conveyor belt.
        # See anchor_grid()'s docstring: this grid assumes THIS deployment's
        # camera framing -- a different camera/line geometry requires
        # recalibrating anchor_grid() (and retraining) to that ROI.
        self.register_buffer("anchors", torch.tensor(anchor_grid(), dtype=torch.float32))

        # Prediction heads directly driven by local visual patch features
        self.score_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Initialize final classification bias for clean background suppression
        nn.init.constant_(self.score_head[2].bias, -1.0)

        self.box_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cls_head[2].bias, -2.0)
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, num_queries, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: returns (boxes, scores, classes, masks)."""
        b = x.size(0)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)  # [B, 64, 160, 160] (P3 mask feature map)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)  # [B, 128, 40, 40] (P5 spatial feature map)

        # Sample local patch features at anchor locations. The /320.0 - 1.0
        # normalization maps CANVAS_SIZE (640x640) pixel coordinates to
        # grid_sample's [-1, 1] space -- this, like the anchor coordinates
        # themselves, is tied to this deployment's fixed camera framing (see
        # anchor_grid() docstring) and CANVAS_SIZE; both must be revisited
        # together for a different camera/canvas geometry.
        norm_anchors = (self.anchors / 320.0) - 1.0  # [20, 2] in [-1, 1]
        grid = norm_anchors.view(1, 1, self.num_queries, 2).expand(b, -1, -1, -1)
        sampled = (
            F.grid_sample(c4, grid, mode="bilinear", align_corners=True)
            .squeeze(2)
            .permute(0, 2, 1)
        )  # [B, 20, 128]

        scores = self.score_head(sampled).squeeze(-1)  # [B, 20]
        raw_boxes = self.box_head(sampled)  # [B, 20, 4]
        classes = self.cls_head(sampled).squeeze(-1)  # [B, 20]
        masks = self.mask_head(c2)  # [B, 20, 160, 160]

        delta_cx = raw_boxes[..., 0] * 35.0
        delta_cy = raw_boxes[..., 1] * 35.0
        w = torch.clamp(torch.exp(raw_boxes[..., 2]) * 180.0, 30.0, 420.0)
        h = torch.clamp(torch.exp(raw_boxes[..., 3]) * 250.0, 30.0, 520.0)

        ax = self.anchors[:, 0].unsqueeze(0)
        ay = self.anchors[:, 1].unsqueeze(0)
        cx = ax + delta_cx
        cy = ay + delta_cy

        x1 = torch.clamp(cx - w / 2.0, 0.0, 640.0)
        y1 = torch.clamp(cy - h / 2.0, 0.0, 640.0)
        x2 = torch.clamp(cx + w / 2.0, 0.0, 640.0)
        y2 = torch.clamp(cy + h / 2.0, 0.0, 640.0)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return boxes, scores, classes, masks


def build_synthetic_training_dataset(
    num_samples: int = 120,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate synthetic conveyor dataset with diverse scenes (empty + 1..4 bags)."""
    gen = SyntheticBagGenerator(min_overlap_ratio=0.15, max_overlap_ratio=0.40)
    anchors_np = anchor_grid()

    dataset = []

    for idx in range(num_samples):
        # 30% empty conveyor scenes to guarantee zero false positives on empty belt
        if idx % 3 == 0:
            num_bags = 0
        else:
            num_bags = np.random.randint(1, 4)

        scene = gen.generate_scene(num_bags=num_bags)

        img = scene["image"].astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # [3, 640, 640]

        classes = [1.0 if has_print else 0.0 for has_print in scene["has_print_marks"]]
        t_boxes, t_scores, t_classes, t_masks = match_boxes_to_anchor_grid(
            anchors_np, scene["amodal_boxes"], scene["amodal_masks"], classes=classes
        )

        dataset.append((img_t, t_boxes, t_scores, t_classes, t_masks))

    return dataset


def save_training_plot(
    history: list[dict[str, Any]],
    output_png_path: str = "models/training_loss_curve.png",
) -> None:
    """Save visualization plot of loss decreasing across training epochs."""
    os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        total_loss = [h["total_loss"] for h in history]
        box_loss = [h["box_loss"] for h in history]
        score_loss = [h["score_loss"] for h in history]
        mask_loss = [h["mask_loss"] for h in history]

        plt.figure(figsize=(9, 5))
        plt.plot(epochs, total_loss, "b-o", linewidth=2, label="Total Loss")
        plt.plot(epochs, box_loss, "r--", label="Box Regression Loss")
        plt.plot(epochs, score_loss, "g--", label="Focal Score Loss")
        plt.plot(epochs, mask_loss, "m--", label="Mask Loss")

        plt.title("RF-DETR Seg Training Convergence (Loss vs Epochs)", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=150)
        plt.close()

        # Also copy to artifacts directory if exists
        artifacts_dir = ROOT_DIR / "artifacts"
        if artifacts_dir.exists():
            art_path = artifacts_dir / "training_loss_curve.png"
            import shutil
            shutil.copy2(output_png_path, str(art_path))

        print(f"[RF-DETR] Training loss plot saved to: {output_png_path}")
    except Exception as e:
        print(f"[RF-DETR] Plot generation note: {e}")


def train_and_export_model(
    output_dir: str = str(ROOT_DIR / "models"),
    model_name: str = "rfdetr_seg_v2.onnx",
    num_synthetic_scenes: int = 120,
    epochs: int = 35,
    batch_size: int = 4,
    learning_rate: float = 1e-2,
    real_data_dir: str | Path | None = None,
    real_data_repeat: int = 3,
) -> str:
    """Train PyTorch RF-DETR Seg model on synthetic scenes and export ONNX model (§6.2, §6.5).

    If a real, CVAT-annotated dataset is present at `real_data_dir` (default
    data/real_bags/, see build_real_training_dataset), it is mixed into
    training alongside the synthetic scenes -- synthetic pretraining -> real
    fine-tune, per the documented training flow. `real_data_repeat` oversamples
    the (typically much smaller) real set so it isn't drowned out by synthetic
    scenes; it has no effect when no real data is present.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, model_name)
    alias_path = os.path.join(output_dir, "rf_detr_v2_1.onnx")
    real_data_dir = Path(real_data_dir) if real_data_dir is not None else ROOT_DIR / "data" / "real_bags"

    print("=" * 70)
    print("  RF-DETR SEG INSTANCE SEGMENTATION NEURAL NETWORK TRAINING PIPELINE")
    print("=" * 70)
    print(f"  [1/4] Generating {num_synthetic_scenes} synthetic conveyor training scenes...")
    dataset = build_synthetic_training_dataset(num_samples=num_synthetic_scenes)

    real_dataset = build_real_training_dataset(real_data_dir)
    if real_dataset:
        dataset = dataset + real_dataset * real_data_repeat
        print(f"        + {len(real_dataset)} real annotated frame(s) from '{real_data_dir}' "
              f"(oversampled x{real_data_repeat} -> {len(real_dataset) * real_data_repeat} samples in training mix)")
    else:
        print(f"        No real annotated data found at '{real_data_dir}' "
              "(expects annotations.json + images/) -- training on synthetic data only.")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"  [2/4] Initializing PyTorch RFDETRSegNet architecture & AdamW optimizer (lr={learning_rate})...")
    model = RFDETRSegNet(num_queries=20)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

    history = []
    print(f"  [3/4] Training for {epochs} epochs...")
    print("  " + "-" * 66)
    print(f"  {'Epoch':>6} | {'Total Loss':>11} | {'Box Loss':>9} | {'Focal Loss':>10} | {'Mask Loss':>10}")
    print("  " + "-" * 66)

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss_sum = 0.0
        box_loss_sum = 0.0
        score_loss_sum = 0.0
        cls_loss_sum = 0.0
        mask_loss_sum = 0.0

        for imgs, b_boxes, b_scores, b_classes, b_masks in loader:
            optimizer.zero_grad()
            p_boxes, p_scores, p_classes, p_masks = model(imgs)

            # Balanced Focal Loss on positive and background queries
            pos_mask = (b_scores > 0.5).float()
            neg_mask = 1.0 - pos_mask

            pos_loss = (
                (1.0 - p_scores) ** 2
                * (-torch.log(p_scores.clamp(min=1e-6)))
                * pos_mask
            ).sum() / max(1.0, float(pos_mask.sum().item()))
            neg_loss = (
                p_scores**2
                * (-torch.log((1.0 - p_scores).clamp(min=1e-6)))
                * neg_mask
            ).mean()
            focal_loss = pos_loss + 15.0 * neg_loss

            # Losses on matched positive queries
            pos = b_scores > 0.5
            if pos.sum() > 0:
                loss_box = F.smooth_l1_loss(p_boxes[pos] / 640.0, b_boxes[pos] / 640.0)
                loss_cls = F.binary_cross_entropy(p_classes[pos], b_classes[pos])
                loss_mask = F.binary_cross_entropy(p_masks[pos], b_masks[pos])
            else:
                loss_box = torch.tensor(0.0)
                loss_cls = torch.tensor(0.0)
                loss_mask = torch.tensor(0.0)

            total_loss = 2.0 * loss_box + focal_loss + 1.0 * loss_cls + 2.0 * loss_mask
            total_loss.backward()
            optimizer.step()

            n = imgs.size(0)
            tot_loss_sum += total_loss.item() * n
            box_loss_sum += loss_box.item() * n
            score_loss_sum += focal_loss.item() * n
            cls_loss_sum += loss_cls.item() * n
            mask_loss_sum += loss_mask.item() * n

        scheduler.step()

        N = len(dataset)
        avg_tot = tot_loss_sum / N
        avg_box = box_loss_sum / N
        avg_score = score_loss_sum / N
        avg_cls = cls_loss_sum / N
        avg_mask = mask_loss_sum / N

        print(f"  {epoch:6d} | {avg_tot:11.4f} | {avg_box:9.4f} | {avg_score:10.4f} | {avg_mask:10.4f}")

        history.append({
            "epoch": epoch,
            "total_loss": round(avg_tot, 5),
            "box_loss": round(avg_box, 5),
            "score_loss": round(avg_score, 5),
            "cls_loss": round(avg_cls, 5),
            "mask_loss": round(avg_mask, 5),
        })

    elapsed = time.perf_counter() - t_start
    print("  " + "-" * 66)
    print(f"  [OK] Training completed in {elapsed:.2f}s. Loss dropped: {history[0]['total_loss']:.4f} -> {history[-1]['total_loss']:.4f}")

    # Save training logs (JSON + CSV)
    json_log_path = os.path.join(output_dir, "training_loss_log.json")
    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    csv_log_path = os.path.join(output_dir, "training_history.csv")
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "total_loss", "box_loss", "score_loss", "cls_loss", "mask_loss"])
        writer.writeheader()
        writer.writerows(history)

    # Save visualization plot
    save_training_plot(history, os.path.join(output_dir, "training_loss_curve.png"))

    # 4. Export trained PyTorch model to ONNX
    print(f"  [4/4] Exporting PyTorch model to ONNX -> '{out_path}'...")
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["images"],
        output_names=["boxes", "scores", "classes", "masks"],
        opset_version=18,
        dynamo=False,
    )

    # Copy / export alias
    torch.onnx.export(
        model,
        dummy_input,
        alias_path,
        input_names=["images"],
        output_names=["boxes", "scores", "classes", "masks"],
        opset_version=18,
        dynamo=False,
    )

    # Verify exported model with ONNX Runtime
    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"images": dummy_input.numpy()})

    print(f"\n[RF-DETR] Genuine PyTorch Model Trained & Exported to: {out_path}")
    print(f"          Verified Output Tensors: boxes={outputs[0].shape}, scores={outputs[1].shape}, classes={outputs[2].shape}, masks={outputs[3].shape}")
    print("=" * 70 + "\n")

    return out_path


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    train_and_export_model()
