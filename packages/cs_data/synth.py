"""Synthetic bag generator with amodal segmentation masks and shingling (§6.5)."""

from __future__ import annotations

import math
import random
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


class SyntheticBagGenerator:
    """Generates synthetic conveyor scenes with amodal annotations and overlap shingling."""

    def __init__(
        self,
        canvas_size: tuple[int, int] = (640, 640),
        min_overlap_ratio: float = 0.0,
        max_overlap_ratio: float = 0.45,
    ) -> None:
        self.canvas_size = canvas_size
        self.min_overlap = min_overlap_ratio
        self.max_overlap = max_overlap_ratio

    def create_empty_conveyor(self) -> Image.Image:
        """Create or sample an empty textured conveyor background."""
        w, h = self.canvas_size
        # Dark gray / rubber conveyor texture with subtle lines
        bg_color = (45, 45, 48)
        img = Image.new("RGB", (w, h), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw belt track lines
        for y in range(0, h, 20):
            draw.line([(0, y), (w, y)], fill=(40, 40, 42), width=1)
        # Side rails
        draw.rectangle([(0, 0), (w, 25)], fill=(70, 70, 75))
        draw.rectangle([(0, h - 25), (w, h)], fill=(70, 70, 75))

        return img

    def create_bag_template(
        self,
        base_size: tuple[int, int] = (160, 240),
        color: tuple[int, int, int] = (220, 215, 200),  # woven pp bag texture
        has_print_mark: bool = True,
    ) -> tuple[Image.Image, Image.Image, list[float] | None]:
        """Generate a single bag template image, its binary alpha mask, and print mark box."""
        bw, bh = base_size
        bag_img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bag_img)

        # Rounded bag body
        draw.rounded_rectangle([(4, 4), (bw - 4, bh - 4)], radius=15, fill=(*color, 255), outline=(170, 165, 150), width=2)

        # Bag stitching lines at edges
        draw.line([(8, 12), (bw - 8, 12)], fill=(120, 115, 100, 255), width=2)
        draw.line([(8, bh - 12), (bw - 8, bh - 12)], fill=(120, 115, 100, 255), width=2)

        print_box = None
        if has_print_mark:
            # Add center print mark / logo box
            pw, ph = int(bw * 0.5), int(bh * 0.3)
            px1 = (bw - pw) // 2
            py1 = (bh - ph) // 2
            px2 = px1 + pw
            py2 = py1 + ph
            draw.rectangle([(px1, py1), (px2, py2)], fill=(180, 50, 40, 220))
            draw.text((px1 + 10, py1 + 10), "FABRIC", fill=(255, 255, 255, 255))
            print_box = [float(px1), float(py1), float(px2), float(py2)]

        mask = bag_img.split()[3]
        return bag_img, mask, print_box

    def generate_scene(
        self,
        num_bags: int | None = None,
        bag_colors: list[tuple[int, int, int]] | None = None,
    ) -> dict[str, Any]:
        """Compose a synthetic training scene with amodal COCO annotations.
        
        Returns:
            {
                "image": np.ndarray (H, W, 3),
                "amodal_masks": list of np.ndarray (H, W bool),
                "amodal_boxes": list of [x1, y1, x2, y2],
                "print_marks": list of [x1, y1, x2, y2],
                "visible_ratios": list of float,
            }
        """
        w, h = self.canvas_size
        canvas = self.create_empty_conveyor()

        if num_bags is None:
            num_bags = random.randint(1, 4)

        colors = bag_colors or [(220, 215, 200), (230, 225, 210), (210, 205, 190)]

        # Determine placements along belt axis with deliberate shingling overlaps
        bag_records = []
        amodal_masks = []
        amodal_boxes = []
        print_mark_boxes = []
        visible_ratios = []

        start_x = 50
        y_center = h // 2

        for i in range(num_bags):
            color = random.choice(colors)
            has_print = random.random() > 0.3
            bag_img, bag_mask, pbox = self.create_bag_template(
                base_size=(random.randint(140, 180), random.randint(220, 260)),
                color=color,
                has_print_mark=has_print,
            )

            # Random slight rotation (-15 to +15 deg)
            rot_deg = random.uniform(-15, 15)
            bag_img_rot = bag_img.rotate(rot_deg, expand=True, resample=Image.Resampling.BILINEAR)
            bag_mask_rot = bag_mask.rotate(rot_deg, expand=True, resample=Image.Resampling.NEAREST)

            bw, bh = bag_img_rot.size

            # Apply shingling overlap along x axis
            overlap_ratio = random.uniform(self.min_overlap, self.max_overlap) if i > 0 else 0.0
            x_pos = int(start_x - (bw * overlap_ratio)) if i > 0 else start_x
            y_pos = int(y_center - bh // 2 + random.randint(-20, 20))

            start_x = x_pos + bw

            bag_records.append({
                "img": bag_img_rot,
                "mask": bag_mask_rot,
                "pos": (x_pos, y_pos),
                "size": (bw, bh),
                "has_print": has_print,
                "pbox": pbox,
            })

        # Z-order paste: bags on top overlap those underneath
        # We compute AMODAL masks (whole body regardless of occlusion)
        canvas_rgba = canvas.convert("RGBA")

        for i, rec in enumerate(bag_records):
            bx, by = rec["pos"]
            bw, bh = rec["size"]

            # Paste on composite canvas
            canvas_rgba.paste(rec["img"], (bx, by), rec["mask"])

            # Compute full amodal mask on canvas coordinates
            full_amodal_mask = np.zeros((h, w), dtype=bool)
            m_arr = np.array(rec["mask"]) > 128

            # Clamping
            dst_x1 = max(0, bx)
            dst_y1 = max(0, by)
            dst_x2 = min(w, bx + bw)
            dst_y2 = min(h, by + bh)

            src_x1 = dst_x1 - bx
            src_y1 = dst_y1 - by
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)

            if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                full_amodal_mask[dst_y1:dst_y2, dst_x1:dst_x2] = m_arr[src_y1:src_y2, src_x1:src_x2]

            amodal_masks.append(full_amodal_mask)
            amodal_boxes.append([float(dst_x1), float(dst_y1), float(dst_x2), float(dst_y2)])

            if rec["has_print"] and rec["pbox"] is not None:
                # Estimate transformed print mark box
                px1, py1, px2, py2 = rec["pbox"]
                print_mark_boxes.append([
                    float(bx + px1), float(by + py1),
                    float(bx + px2), float(by + py2),
                ])

            visible_ratio = 1.0 - (0.5 * overlap_ratio if i < len(bag_records) - 1 else 0.0)
            visible_ratios.append(visible_ratio)

        # Apply realistic distortions (lighting, slight blur)
        final_img = canvas_rgba.convert("RGB")
        enhancer = ImageEnhance.Brightness(final_img)
        final_img = enhancer.enhance(random.uniform(0.85, 1.15))

        if random.random() > 0.5:
            final_img = final_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

        return {
            "image": np.array(final_img),
            "amodal_masks": amodal_masks,
            "amodal_boxes": amodal_boxes,
            "print_marks": print_mark_boxes,
            "visible_ratios": visible_ratios,
        }
