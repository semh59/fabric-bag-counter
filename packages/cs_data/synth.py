"""Synthetic bag generator with amodal segmentation masks and shingling (§6.5)."""

from __future__ import annotations

import logging
import math
import random
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# Truetype fonts to try (in order) for legible print-mark text. Falls back to
# PIL's tiny bitmap default font (rendered at a higher supersample scale) if
# none of these are resolvable on the current platform.
_PRINT_MARK_FONT_CANDIDATES = ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf")


def _load_print_mark_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont | None:
    """Best-effort load of a real truetype font for print-mark text; None if unavailable."""
    for name in _PRINT_MARK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return None


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
    ) -> tuple[Image.Image, Image.Image, list[float] | None, Image.Image | None]:
        """Generate a single bag template image, its binary alpha mask, its print mark
        box (in bag-local, unrotated coordinates), and a single-channel "print mask"
        image (mode "L", same size as the bag) whose non-zero pixels mark the print
        box region. The print mask lets callers retransform the print box through
        whatever rotation is later applied to the bag image (see generate_scene),
        by rotating this mask with the exact same PIL call and re-deriving the
        bounding box from the rotated pixels, instead of naively offsetting the
        original untransformed coordinates.
        """
        bw, bh = base_size
        bag_img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bag_img)

        # Rounded bag body
        draw.rounded_rectangle([(4, 4), (bw - 4, bh - 4)], radius=15, fill=(*color, 255), outline=(170, 165, 150), width=2)

        # Bag stitching lines at edges
        draw.line([(8, 12), (bw - 8, 12)], fill=(120, 115, 100, 255), width=2)
        draw.line([(8, bh - 12), (bw - 8, bh - 12)], fill=(120, 115, 100, 255), width=2)

        print_box = None
        print_mask_img = None
        if has_print_mark:
            # Add center print mark / logo box
            pw, ph = int(bw * 0.5), int(bh * 0.3)
            px1 = (bw - pw) // 2
            py1 = (bh - ph) // 2
            px2 = px1 + pw
            py2 = py1 + ph
            draw.rectangle([(px1, py1), (px2, py2)], fill=(180, 50, 40, 220))

            # Render the print-mark text at a supersampled scale (or with a real
            # truetype font when available) so it isn't the tiny/illegible PIL
            # default bitmap font baked straight into the bag image.
            label = "FABRIC"
            supersample = 4
            font = _load_print_mark_font(size=int(ph * 0.5))
            if font is not None:
                text_layer_size = (pw, ph)
                text_layer = Image.new("RGBA", text_layer_size, (0, 0, 0, 0))
                tdraw = ImageDraw.Draw(text_layer)
                bbox = tdraw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = (pw - tw) // 2 - bbox[0]
                ty = (ph - th) // 2 - bbox[1]
                tdraw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)
            else:
                # No truetype font resolvable: fall back to the default bitmap
                # font but render it big via supersampling, then downsample with
                # a high-quality filter for much better perceived legibility.
                big_size = (pw * supersample, ph * supersample)
                text_layer_big = Image.new("RGBA", big_size, (0, 0, 0, 0))
                tdraw = ImageDraw.Draw(text_layer_big)
                bbox = tdraw.textbbox((0, 0), label)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                scale = min(big_size[0] / max(tw, 1), big_size[1] / max(th, 1)) * 0.7
                text_layer_scaled = Image.new("RGBA", big_size, (0, 0, 0, 0))
                tdraw2 = ImageDraw.Draw(text_layer_scaled)
                tdraw2.text((0, 0), label, fill=(255, 255, 255, 255))
                new_w = max(1, int(text_layer_scaled.width * scale))
                new_h = max(1, int(text_layer_scaled.height * scale))
                text_layer_resized = text_layer_scaled.resize((new_w, new_h), Image.Resampling.LANCZOS)
                text_layer = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
                text_layer.paste(
                    text_layer_resized,
                    ((pw - new_w) // 2, (ph - new_h) // 2),
                    text_layer_resized,
                )

            bag_img.alpha_composite(text_layer, (px1, py1))
            print_box = [float(px1), float(py1), float(px2), float(py2)]

            print_mask_img = Image.new("L", (bw, bh), 0)
            pm_draw = ImageDraw.Draw(print_mask_img)
            pm_draw.rectangle([(px1, py1), (px2, py2)], fill=255)

        mask = bag_img.split()[3]
        return bag_img, mask, print_box, print_mask_img

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
            bag_img, bag_mask, pbox, pmask_img = self.create_bag_template(
                base_size=(random.randint(140, 180), random.randint(220, 260)),
                color=color,
                has_print_mark=has_print,
            )

            # Random slight rotation (-15 to +15 deg)
            rot_deg = random.uniform(-15, 15)
            bag_img_rot = bag_img.rotate(rot_deg, expand=True, resample=Image.Resampling.BILINEAR)
            bag_mask_rot = bag_mask.rotate(rot_deg, expand=True, resample=Image.Resampling.NEAREST)
            # Rotate the print-mark mask through the *exact same* PIL call (same
            # angle, same expand=True canvas recentering) used for the bag image
            # itself, so its post-rotation pixel footprint is genuinely correct
            # instead of naively offsetting the pre-rotation box coordinates.
            pmask_rot = (
                pmask_img.rotate(rot_deg, expand=True, resample=Image.Resampling.NEAREST)
                if pmask_img is not None
                else None
            )

            bw, bh = bag_img_rot.size

            # Apply shingling overlap along x axis. overlap_ratio here describes
            # how much *this* bag (i) overlaps backward onto the previous bag
            # (i-1); it must be captured per-bag (not left as a shared loop
            # variable) because it is read back below, once per bag, when
            # deriving each bag's own visible_ratio.
            overlap_ratio = random.uniform(self.min_overlap, self.max_overlap) if i > 0 else 0.0
            x_pos = int(start_x - (bw * overlap_ratio)) if i > 0 else start_x
            y_pos = int(y_center - bh // 2 + random.randint(-20, 20))

            start_x = x_pos + bw

            bag_records.append({
                "img": bag_img_rot,
                "mask": bag_mask_rot,
                "pmask_rot": pmask_rot,
                "pos": (x_pos, y_pos),
                "size": (bw, bh),
                "has_print": has_print,
                "pbox": pbox,
                "overlap_ratio": overlap_ratio,
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

            if rec["has_print"] and rec["pbox"] is not None and rec["pmask_rot"] is not None:
                # Derive the *actually* rotated print mark box by reading back
                # the bounding box of the non-zero pixels in the print mask
                # after it went through the identical rotate(expand=True) call
                # as the bag image, then offsetting by the final paste position.
                pm_arr = np.array(rec["pmask_rot"]) > 128
                ys, xs = np.nonzero(pm_arr)
                if xs.size > 0 and ys.size > 0:
                    print_mark_boxes.append([
                        float(bx + xs.min()), float(by + ys.min()),
                        float(bx + xs.max() + 1), float(by + ys.max() + 1),
                    ])

            # Each bag's visibility is reduced by whichever bag was pasted on
            # top of it -- i.e. the *next* bag in z-order -- so we must read
            # back that next bag's own stored overlap_ratio (not a stale
            # shared loop variable) rather than this bag's own overlap with
            # its predecessor.
            next_overlap = bag_records[i + 1]["overlap_ratio"] if i < len(bag_records) - 1 else 0.0
            visible_ratio = 1.0 - 0.5 * next_overlap
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
            # Per-bag, aligned 1:1 with amodal_boxes/amodal_masks (unlike
            # print_marks above, which only has one entry per bag that
            # actually got a print mark and so isn't index-aligned) --
            # lets a caller build a real per-box print_mark classification
            # target instead of training cls_head against an always-0 label.
            "has_print_marks": [rec["has_print"] for rec in bag_records],
            "visible_ratios": visible_ratios,
        }
