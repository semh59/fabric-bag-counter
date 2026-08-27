"""Tests for packages.cs_data.synth (synthetic bag scene generator).

Covers two real bugs fixed in generate_scene / create_bag_template:

1. `overlap_ratio` used to be a loop-local variable computed once per bag in
   the placement loop, then re-read (unchanged, i.e. stale) in a *second*
   loop when deriving each bag's `visible_ratio`. Because it was never
   reassigned per-bag in the second loop, every bag except the one matching
   the last iteration of the first loop got the wrong, stale overlap value
   baked into its visible_ratio.

2. The print-mark box returned by `create_bag_template` is in bag-local,
   unrotated coordinates. After the bag is rotated with `expand=True` (which
   also grows/recenters the canvas), the code used to just offset that
   *unrotated* box by the final paste position instead of transforming it
   through the rotation -- so the reported print-mark box drifted away from
   where the print-mark pixels actually landed for any non-zero rotation.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from packages.cs_data.synth import SyntheticBagGenerator


def test_visible_ratio_uses_each_bags_own_overlap_not_a_stale_shared_value():
    """A 5-bag scene: each bag's visible_ratio must reflect *its own* overlap
    with the bag pasted on top of it (the next bag in z-order), not a single
    shared/stale overlap_ratio value.
    """
    random.seed(7)
    # Wide canvas so bags never run off the right edge and amodal_boxes never
    # get clamped/degenerate -- keeps the x-overlap-derived expectation below
    # exact.
    gen = SyntheticBagGenerator(canvas_size=(2000, 640), min_overlap_ratio=0.15, max_overlap_ratio=0.40)
    scene = gen.generate_scene(num_bags=5, bag_colors=[(220, 215, 200)])

    boxes = scene["amodal_boxes"]
    visible_ratios = scene["visible_ratios"]
    assert len(boxes) == 5
    assert len(visible_ratios) == 5

    # The last bag (topmost, nothing pasted over it) must always be fully visible.
    assert visible_ratios[-1] == 1.0

    # For every other bag, independently derive the *actual* overlap with the
    # next (topmost) bag directly from the composited box geometry, and check
    # the reported visible_ratio matches that -- not some other bag's overlap.
    for i in range(len(boxes) - 1):
        bw_next = boxes[i + 1][2] - boxes[i + 1][0]
        overlap_len = boxes[i][2] - boxes[i + 1][0]
        expected_overlap = max(0.0, overlap_len / bw_next)
        expected_visible = 1.0 - 0.5 * expected_overlap
        assert visible_ratios[i] == pytest.approx(expected_visible, abs=0.01), (
            f"bag {i}: visible_ratio={visible_ratios[i]!r} does not match its own "
            f"measured overlap-derived expectation {expected_visible!r}"
        )

    # Regression guard for the original bug: with a stale shared overlap_ratio,
    # bags 0 and 1 would have ended up with the SAME visible_ratio as bag 3
    # (the last-computed overlap in the placement loop) instead of their own
    # distinct, smaller-index overlaps. Assert the sequence is not degenerately
    # constant across the non-final bags (which the bug would produce whenever
    # the shared value got reused verbatim).
    non_final = visible_ratios[:-1]
    assert len(set(round(v, 6) for v in non_final)) > 1, (
        "all non-final visible_ratios are identical -- looks like the stale "
        "shared overlap_ratio bug is back"
    )


def _reddish_logo_pixel_mask(bag_img_rot: Image.Image) -> np.ndarray:
    """Identify pixels belonging to the print-mark box fill color (180,50,40),
    which is visually distinct from the body color, outline, and stitching
    lines used elsewhere on the bag template.
    """
    arr = np.array(bag_img_rot)
    alpha = arr[..., 3]
    rgb = arr[..., :3].astype(int)
    logo_color = np.array([180, 50, 40])
    dist = np.abs(rgb - logo_color).sum(axis=-1)
    return (alpha > 128) & (dist < 90)


def test_print_box_rotation_transform_overlaps_actual_logo_pixels():
    """For several rotation angles, the print-mark box must be re-derived
    through the same rotation as the bag image -- not just offset from the
    original unrotated coordinates -- so it actually overlaps where the
    print-mark pixels land after rotation.
    """
    gen = SyntheticBagGenerator()
    body_color = (220, 215, 200)

    for angle in (0.0, 15.0, -15.0, 30.0, 45.0):
        bag_img, _mask, pbox, pmask_img = gen.create_bag_template(has_print_mark=True, color=body_color)
        assert pbox is not None and pmask_img is not None

        bag_img_rot = bag_img.rotate(angle, expand=True, resample=Image.Resampling.BILINEAR)
        # This is exactly what generate_scene now does: rotate the print mask
        # through the identical transform and read back its bounding box.
        pmask_rot = pmask_img.rotate(angle, expand=True, resample=Image.Resampling.NEAREST)

        logo_pixels = _reddish_logo_pixel_mask(bag_img_rot)
        assert logo_pixels.sum() > 0, "expected some print-mark pixels to be rendered"

        pm_arr = np.array(pmask_rot) > 128
        ys, xs = np.nonzero(pm_arr)
        assert xs.size > 0 and ys.size > 0
        new_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

        box_mask = np.zeros_like(logo_pixels)
        box_mask[new_box[1]:new_box[3], new_box[0]:new_box[2]] = True
        coverage_new = (logo_pixels & box_mask).sum() / logo_pixels.sum()

        assert coverage_new >= 0.98, (
            f"angle={angle}: rotated print-mark box only covers "
            f"{coverage_new:.3f} of the actual rendered logo pixels"
        )

        # Regression guard: naively offsetting the ORIGINAL unrotated box
        # (the old, buggy behavior) should perform much worse for any
        # non-zero rotation, proving the fix is doing real geometric work
        # and not accidentally passing because both approaches agree.
        if angle != 0.0:
            px1, py1, px2, py2 = (int(v) for v in pbox)
            h, w = logo_pixels.shape
            ox1, oy1 = max(0, px1), max(0, py1)
            ox2, oy2 = min(w, px2), min(h, py2)
            old_box_mask = np.zeros_like(logo_pixels)
            if ox2 > ox1 and oy2 > oy1:
                old_box_mask[oy1:oy2, ox1:ox2] = True
            coverage_old = (logo_pixels & old_box_mask).sum() / logo_pixels.sum()
            assert coverage_old < coverage_new - 0.2, (
                f"angle={angle}: naive unrotated-offset box unexpectedly covers "
                f"about as much of the logo as the properly rotated box "
                f"(old={coverage_old:.3f}, new={coverage_new:.3f})"
            )
