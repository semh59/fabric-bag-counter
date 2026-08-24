"""Postprocessing for RF-DETR Seg outputs (masks and boxes)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def postprocess_rfdetr_seg(
    boxes: np.ndarray,          # (N, 4) in [x1, y1, x2, y2]
    scores: np.ndarray,         # (N,)
    classes: np.ndarray,        # (N,) class_id: 0 = bag_body, 1 = print_mark
    raw_masks: np.ndarray | None,  # (N, mh, mw) or (N, H, W) logits/probs
    orig_shape: tuple[int, int],   # (orig_h, orig_w)
    scale: float,
    pad: tuple[int, int],          # (pad_w, pad_h)
    conf_threshold: float = 0.40,
    mask_threshold: float = 0.50,
) -> tuple[list[dict], list[dict]]:
    """Decode raw model predictions into bag_body masks and print_mark bounding boxes.
    
    Returns:
        (bag_bodies, print_marks)
        bag_bodies: list of dicts {"box": [x1, y1, x2, y2], "score": float, "mask": np.ndarray (orig_h, orig_w boolean)}
        print_marks: list of dicts {"box": [x1, y1, x2, y2], "score": float}
    """
    orig_h, orig_w = orig_shape
    pad_w, pad_h = pad

    bag_bodies = []
    print_marks = []

    for i in range(len(scores)):
        score = float(scores[i])
        if score < conf_threshold:
            continue

        cls_id = int(classes[i])
        box = boxes[i]

        # Invert letterbox on box coordinates: [x1, y1, x2, y2]
        x1 = max(0.0, (box[0] - pad_w) / scale)
        y1 = max(0.0, (box[1] - pad_h) / scale)
        x2 = min(float(orig_w), (box[2] - pad_w) / scale)
        y2 = min(float(orig_h), (box[3] - pad_h) / scale)

        if x2 <= x1 or y2 <= y1:
            continue

        unpadded_box = [x1, y1, x2, y2]

        if cls_id == 0:  # bag_body
            if raw_masks is not None and i < len(raw_masks):
                # Decode mask
                rmask = raw_masks[i]
                if rmask.ndim == 2:
                    # Resize mask to original image size
                    pil_mask = Image.fromarray((rmask > mask_threshold).astype(np.uint8) * 255)
                    # Unpad and scale
                    # First unpad
                    mask_h, mask_w = rmask.shape
                    if mask_h != orig_h or mask_w != orig_w:
                        # Full image mask resizing
                        resized_mask = pil_mask.resize((int(mask_w / scale), int(mask_h / scale)), Image.Resampling.NEAREST)
                        full_mask = np.zeros((orig_h, orig_w), dtype=bool)
                        res_arr = np.array(resized_mask) > 128
                        rh = min(orig_h, res_arr.shape[0])
                        rw = min(orig_w, res_arr.shape[1])
                        full_mask[:rh, :rw] = res_arr[:rh, :rw]
                    else:
                        full_mask = rmask > mask_threshold
                else:
                    full_mask = np.zeros((orig_h, orig_w), dtype=bool)
                    bx1, by1, bx2, by2 = map(int, unpadded_box)
                    full_mask[by1:by2, bx1:bx2] = True
            else:
                # Bbox fallback mask
                full_mask = np.zeros((orig_h, orig_w), dtype=bool)
                bx1, by1, bx2, by2 = map(int, unpadded_box)
                full_mask[by1:by2, bx1:bx2] = True

            bag_bodies.append({
                "box": unpadded_box,
                "score": score,
                "mask": full_mask,
            })
        elif cls_id == 1:  # print_mark
            print_marks.append({
                "box": unpadded_box,
                "score": score,
            })

    return bag_bodies, print_marks
