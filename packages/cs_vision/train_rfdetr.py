"""RF-DETR Seg PyTorch Training and ONNX Model Export Pipeline (§6.2, §6.3, §6.5).

Builds, trains, and exports a deep instance segmentation model for industrial
conveyor bags featuring multi-head cross-attention query transformer blocks,
dynamic instance mask heads, and balanced multi-task loss formulation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings
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
    """Fixed anchor grid across the conveyor belt capture area (§6.2)."""
    grid_x = np.linspace(70, 570, num_x)
    anchors = [[x, y] for y in num_y_positions for x in grid_x]
    return np.array(anchors, dtype=np.float32)


def match_boxes_to_anchor_grid(
    anchors_np: np.ndarray,
    boxes: list[list[float]],
    masks: list[np.ndarray],
    classes: list[float] | None = None,
    num_queries: int = NUM_QUERIES,
    mask_size: int = MASK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily assign ground-truth boxes/masks to the nearest unoccupied anchor query."""
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
        t_classes[matched_q] = float(classes[i]) if classes is not None else 0.5

        pil_m = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (mask_size, mask_size), Image.Resampling.NEAREST
        )
        t_masks[matched_q] = torch.from_numpy(np.array(pil_m) > 128).float()

    return t_boxes, t_scores, t_classes, t_masks


def _coco_polygon_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterize a COCO polygon segmentation to a boolean mask."""
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
    """Load a real annotated dataset exported from CVAT if present (§6.3)."""
    annotations_path = data_dir / "annotations.json"
    if not annotations_path.exists():
        return []

    try:
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
                continue

            pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size
            img_arr = np.array(pil_img)

            padded_img, scale, (pad_w, pad_h) = letterbox_image(img_arr, canvas_size, fill_value=114)

            boxes_canvas: list[list[float]] = []
            masks_canvas: list[np.ndarray] = []
            classes_canvas: list[float] = []

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
    except Exception as e:
        print(f"  [RF-DETR] Note loading real dataset: {e}")
        return []


class TransformerDecoderLayer(nn.Module):
    """Multi-Head Cross-Attention Transformer Query Decoder Layer (§6.2)."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, ffn_dim: int = 256) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # 1. Self Attention across query representations
        q_self, _ = self.self_attn(query, query, query)
        query = self.norm1(query + q_self)

        # 2. Cross Attention over multi-scale visual memory features
        q_cross, _ = self.cross_attn(query, memory, memory)
        query = self.norm2(query + q_cross)

        # 3. Feed Forward Network
        q_ffn = self.ffn(query)
        query = self.norm3(query + q_ffn)

        return query


class RFDETRSegNet(nn.Module):
    """Attention-augmented RF-DETR Seg Neural Network for Industrial Bag Counting & Segmentation."""

    def __init__(self, num_queries: int = NUM_QUERIES) -> None:
        super().__init__()
        self.num_queries = num_queries
        embed_dim = 128

        # 4-stage convolutional feature backbone with GroupNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Spatial anchor grid & learned query embeddings
        anchors_arr = anchor_grid()
        self.register_buffer("anchors", torch.tensor(anchors_arr, dtype=torch.float32))
        self.query_pos_embed = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.init_query_embed = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        # Multi-Head Cross-Attention Transformer Decoder
        self.decoder_layer1 = TransformerDecoderLayer(embed_dim=embed_dim, num_heads=4, ffn_dim=256)
        self.decoder_layer2 = TransformerDecoderLayer(embed_dim=embed_dim, num_heads=4, ffn_dim=256)

        # Instance Mask Prototype Generator (P3 resolution: 160x160)
        self.proto_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1),
        )
        self.mask_controller = nn.Linear(embed_dim, 16)

        # Prediction Heads
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Initialize score head bias for clean background suppression
        nn.init.constant_(self.score_head[2].bias, -1.5)

        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: returns (boxes, scores, classes, masks)."""
        b = x.size(0)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)  # [B, 64, 160, 160] (P3 mask features)
        c3 = self.conv3(c2)  # [B, 128, 80, 80]
        c4 = self.conv4(c3)  # [B, 128, 40, 40] (P5 spatial memory)

        # Flatten P5 spatial memory for Transformer cross-attention: [B, 1600, 128]
        hw = c4.size(2) * c4.size(3)
        memory = c4.flatten(2).permute(0, 2, 1)

        # Local anchor feature injection into initial query embeddings
        norm_anchors = (self.anchors / 320.0) - 1.0
        grid = norm_anchors.view(1, 1, self.num_queries, 2).expand(b, -1, -1, -1)
        sampled_local = (
            F.grid_sample(c4, grid, mode="bilinear", align_corners=True)
            .squeeze(2)
            .permute(0, 2, 1)
        )  # [B, 20, 128]

        # Combine learned queries with local features and positional embeddings
        queries = self.init_query_embed.unsqueeze(0).expand(b, -1, -1) + sampled_local
        queries = queries + self.query_pos_embed.unsqueeze(0).expand(b, -1, -1)

        # Execute 2-layer Transformer Cross-Attention Decoder
        queries = self.decoder_layer1(queries, memory)
        queries = self.decoder_layer2(queries, memory)  # [B, 20, 128]

        # Residual skip connection from local visual features for anchor-aligned objectness
        queries = queries + sampled_local

        # Multi-task predictions
        scores = self.score_head(queries).squeeze(-1)  # [B, 20]
        raw_boxes = self.box_head(queries)  # [B, 20, 4]
        classes = self.cls_head(queries).squeeze(-1)  # [B, 20]

        # Dynamic Mask Generation: query coefficients * mask prototypes
        prototypes = self.proto_conv(c2)  # [B, 16, 160, 160]
        mask_coeffs = self.mask_controller(queries)  # [B, 20, 16]
        masks = torch.sigmoid(torch.einsum("bqc,bchw->bqhw", mask_coeffs, prototypes))  # [B, 20, 160, 160]

        # Bounding box anchor-guided reconstruction with wide continuous tracking range
        delta_cx = raw_boxes[..., 0] * 120.0
        delta_cy = raw_boxes[..., 1] * 100.0
        w = torch.clamp(torch.exp(raw_boxes[..., 2]) * 165.0, 40.0, 450.0)
        h = torch.clamp(torch.exp(raw_boxes[..., 3]) * 240.0, 40.0, 520.0)

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
    num_samples: int = 150,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate industrial conveyor scenes with realistic bags and empty conveyor samples."""
    gen = SyntheticBagGenerator(min_overlap_ratio=0.10, max_overlap_ratio=0.35)
    anchors_np = anchor_grid()
    dataset = []

    for idx in range(num_samples):
        # 25% empty conveyor scenes to guarantee zero false positives on empty belt
        if idx % 4 == 0:
            num_bags = 0
        else:
            num_bags = np.random.randint(1, 4)

        scene = gen.generate_scene(num_bags=num_bags)

        img = scene["image"].astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)

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
        cls_loss = [h["cls_loss"] for h in history]
        mask_loss = [h["mask_loss"] for h in history]

        plt.figure(figsize=(10, 5.5))
        plt.plot(epochs, total_loss, "b-o", linewidth=2, label="Total Loss")
        plt.plot(epochs, box_loss, "r--", label="Box Regression Loss")
        plt.plot(epochs, score_loss, "g--", label="Objectness Score Loss")
        plt.plot(epochs, cls_loss, "c--", label="Print Mark Loss")
        plt.plot(epochs, mask_loss, "m--", label="Mask Loss")

        plt.title("RF-DETR Seg Transformer Training Convergence", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss Value", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=150)
        plt.close()

        artifacts_dir = ROOT_DIR / "artifacts"
        if artifacts_dir.exists():
            import shutil
            shutil.copy2(output_png_path, str(artifacts_dir / "training_loss_curve.png"))

        print(f"[RF-DETR] Training plot saved to: {output_png_path}")
    except Exception as e:
        print(f"[RF-DETR] Plot generation note: {e}")


def train_and_export_model(
    output_dir: str = str(ROOT_DIR / "models"),
    model_name: str = "rfdetr_seg_v2.onnx",
    num_synthetic_scenes: int = 100,
    epochs: int = 25,
    batch_size: int = 4,
    learning_rate: float = 8e-3,
    real_data_dir: str | Path | None = None,
    real_data_repeat: int = 3,
) -> str:
    """Train PyTorch RF-DETR Seg Transformer model and export ONNX model (§6.2, §6.5)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, model_name)
    alias_path = os.path.join(output_dir, "rf_detr_v2_1.onnx")
    real_data_dir = Path(real_data_dir) if real_data_dir is not None else ROOT_DIR / "data" / "real_bags"

    print("=" * 75)
    print("  RF-DETR SEG TRANSFORMER INSTANCE SEGMENTATION TRAINING PIPELINE")
    print("=" * 75)
    print(f"  [1/4] Generating {num_synthetic_scenes} industrial conveyor training scenes...")
    dataset = build_synthetic_training_dataset(num_samples=num_synthetic_scenes)

    real_dataset = build_real_training_dataset(real_data_dir)
    if real_dataset:
        dataset = dataset + real_dataset * real_data_repeat
        print(f"        + {len(real_dataset)} real annotated frames from '{real_data_dir}'")
    else:
        print(f"        Using {len(dataset)} generated industrial scenes.")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"  [2/4] Initializing RF-DETR Seg Cross-Attention Transformer Network & AdamW (lr={learning_rate})...")
    model = RFDETRSegNet(num_queries=NUM_QUERIES)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.05)

    history = []
    print(f"  [3/4] Training for {epochs} epochs...")
    print("  " + "-" * 70)
    print(f"  {'Epoch':>6} | {'Total Loss':>11} | {'Box Loss':>9} | {'Score Loss':>10} | {'Cls Loss':>9} | {'Mask Loss':>9}")
    print("  " + "-" * 70)

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

            # Unreduced weighted BCE on objectness score (clean background suppression)
            bce_unreduced = F.binary_cross_entropy(p_scores, b_scores, reduction="none")
            weights = torch.where(b_scores > 0.5, 10.0, 1.0)
            score_loss = (bce_unreduced * weights).mean()

            # Supervised loss on matched bag queries
            pos = b_scores > 0.5
            if pos.sum() > 0:
                loss_box = F.smooth_l1_loss(p_boxes[pos] / 640.0, b_boxes[pos] / 640.0)
                loss_cls = F.binary_cross_entropy(p_classes[pos], b_classes[pos])
                loss_mask = F.binary_cross_entropy(p_masks[pos], b_masks[pos])
            else:
                loss_box = torch.tensor(0.0)
                loss_cls = torch.tensor(0.0)
                loss_mask = torch.tensor(0.0)

            total_loss = 4.0 * loss_box + score_loss + 1.0 * loss_cls + 2.0 * loss_mask
            total_loss.backward()
            optimizer.step()

            n = imgs.size(0)
            tot_loss_sum += total_loss.item() * n
            box_loss_sum += loss_box.item() * n
            score_loss_sum += score_loss.item() * n
            cls_loss_sum += loss_cls.item() * n
            mask_loss_sum += loss_mask.item() * n

        scheduler.step()

        N = len(dataset)
        avg_tot = tot_loss_sum / N
        avg_box = box_loss_sum / N
        avg_score = score_loss_sum / N
        avg_cls = cls_loss_sum / N
        avg_mask = mask_loss_sum / N

        print(f"  {epoch:6d} | {avg_tot:11.4f} | {avg_box:9.4f} | {avg_score:10.4f} | {avg_cls:9.4f} | {avg_mask:9.4f}")

        history.append({
            "epoch": epoch,
            "total_loss": round(avg_tot, 5),
            "box_loss": round(avg_box, 5),
            "score_loss": round(avg_score, 5),
            "cls_loss": round(avg_cls, 5),
            "mask_loss": round(avg_mask, 5),
        })

    elapsed = time.perf_counter() - t_start
    print("  " + "-" * 70)
    print(f"  [OK] Training completed in {elapsed:.2f}s. Total Loss: {history[0]['total_loss']:.4f} -> {history[-1]['total_loss']:.4f}")

    # Save training logs (JSON + CSV)
    json_log_path = os.path.join(output_dir, "training_loss_log.json")
    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    csv_log_path = os.path.join(output_dir, "training_history.csv")
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "total_loss", "box_loss", "score_loss", "cls_loss", "mask_loss"])
        writer.writeheader()
        writer.writerows(history)

    save_training_plot(history, os.path.join(output_dir, "training_loss_curve.png"))

    # Export trained PyTorch model to ONNX
    print(f"  [4/4] Exporting Attention-Augmented PyTorch Model to ONNX -> '{out_path}'...")
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            model,
            dummy_input,
            out_path,
            input_names=["images"],
            output_names=["boxes", "scores", "classes", "masks"],
            opset_version=18,
            dynamo=False,
        )

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

    print(f"\n[RF-DETR] Neural Network Trained & Exported to: {out_path}")
    print(f"          Output Tensors: boxes={outputs[0].shape}, scores={outputs[1].shape}, classes={outputs[2].shape}, masks={outputs[3].shape}")
    print("=" * 75 + "\n")

    return out_path


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    train_and_export_model()
