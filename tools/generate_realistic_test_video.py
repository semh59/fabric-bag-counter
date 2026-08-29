"""Generate high-fidelity realistic industrial test conveyor video with moving bags (§6.5)."""

import os
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image

from packages.cs_data.synth import SyntheticBagGenerator

ROOT_DIR = Path(__file__).resolve().parent.parent


def generate_realistic_conveyor_video(
    output_path: str = "data/test_conveyor_input.mp4",
    num_frames: int = 120,
    fps: int = 25,
    width: int = 640,
    height: int = 640,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    gen = SyntheticBagGenerator(canvas_size=(width, height))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pre-generate 4 distinct industrial bags with textures & brand labels
    bag_templates = [
        gen.create_bag_template(base_size=(160, 240), color=(225, 218, 198), has_print_mark=True, material_type="woven_pp"),
        gen.create_bag_template(base_size=(155, 235), color=(210, 160, 110), has_print_mark=True, material_type="kraft"),
        gen.create_bag_template(base_size=(165, 245), color=(235, 235, 230), has_print_mark=True, material_type="woven_pp"),
        gen.create_bag_template(base_size=(150, 230), color=(185, 180, 175), has_print_mark=True, material_type="kraft"),
    ]

    # Initialize bag positions entering from left (x < 0) moving towards right
    bags_active = [
        {"tpl": bag_templates[0], "x": -100.0, "y": 320.0, "speed": 4.5, "angle": 3.0},
        {"tpl": bag_templates[1], "x": -320.0, "y": 310.0, "speed": 4.5, "angle": -5.0},
        {"tpl": bag_templates[2], "x": -540.0, "y": 330.0, "speed": 4.5, "angle": 2.0},
        {"tpl": bag_templates[3], "x": -760.0, "y": 315.0, "speed": 4.5, "angle": -2.0},
    ]

    base_conveyor = gen.create_empty_conveyor().convert("RGBA")

    print(f"[Video Generator] Rendering {num_frames} frames of realistic industrial conveyor video...")
    for frame_i in range(num_frames):
        # Create conveyor frame with subtle rubber movement striations
        frame_canvas = base_conveyor.copy()
        w, h = width, height

        for bag in bags_active:
            bx = bag["x"]
            by = bag["y"]
            bag_img, bag_mask, _, _ = bag["tpl"]

            rot_bag = bag_img.rotate(bag["angle"], expand=True, resample=Image.Resampling.BILINEAR)
            rot_mask = bag_mask.rotate(bag["angle"], expand=True, resample=Image.Resampling.NEAREST)

            rw, rh = rot_bag.size
            top_left_x = int(bx - rw // 2)
            top_left_y = int(by - rh // 2)

            if -rw < top_left_x < w + rw:
                # Add shadow
                shadow_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                s_img = Image.new("RGBA", (rw + 20, rh + 20), (0, 0, 0, 0))
                from PIL import ImageDraw, ImageFilter
                sd = ImageDraw.Draw(s_img)
                sd.ellipse([(10, 10), (rw + 10, rh + 10)], fill=(15, 15, 18, 110))
                s_img = s_img.filter(ImageFilter.GaussianBlur(6))
                shadow_canvas.paste(s_img, (top_left_x - 5, top_left_y + 8))
                frame_canvas = Image.alpha_composite(frame_canvas, shadow_canvas)

                # Composite bag
                b_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                b_canvas.paste(rot_bag, (top_left_x, top_left_y))
                frame_canvas = Image.alpha_composite(frame_canvas, b_canvas)

            # Advance bag along conveyor
            bag["x"] += bag["speed"]

        # Convert to BGR for OpenCV VideoWriter
        rgb_frame = np.array(frame_canvas.convert("RGB"))
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    print(f"[Video Generator] Video saved to: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    generate_realistic_conveyor_video()
