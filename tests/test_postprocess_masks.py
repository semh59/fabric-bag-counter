"""Unit tests for RF-DETR Seg mask decoding (§6.2, §6.3).

Regression test for the mask-coordinate bug: raw model masks are predicted at
a fixed low resolution (e.g. 160x160, see train_rfdetr.MASK_SIZE) over the
*full padded model canvas* (e.g. 640x640, see train_rfdetr.CANVAS_SIZE) -- a
4x downsample that is completely independent of the per-image letterbox
`scale`/`pad` used to invert box coordinates. A previous version of
`postprocess_rfdetr_seg` only inverted the letterbox `scale` on the raw mask
and never accounted for the mask/canvas resolution ratio, which stranded the
decoded mask as a small patch pinned to the top-left corner of the image,
regardless of where the actual detection box was. This test proves the fix by
asserting the decoded full-resolution mask's bounding box actually overlaps
the detection's own (correctly inverted) box.
"""

import numpy as np

from packages.cs_vision.postprocess import postprocess_rfdetr_seg


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def test_decoded_mask_overlaps_detection_box_not_stuck_in_corner():
    """A detection box far from the top-left corner must yield a mask whose
    bounding box lands in the same region as the box -- not near (0, 0)."""
    canvas_size = (640, 640)  # (canvas_h, canvas_w)
    orig_h, orig_w = 320, 640
    scale = 1.0          # letterbox_image(orig 320x640 -> canvas 640x640) => scale=1.0
    pad_w, pad_h = 0, 160  # vertical padding on a wide image

    mask_size = 160
    raw_mask = np.zeros((mask_size, mask_size), dtype=np.float32)

    # Box placed well away from the top-left corner, in canvas coordinates.
    box_canvas = [400.0, 300.0, 550.0, 420.0]
    boxes = np.array([box_canvas], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([0.0], dtype=np.float32)

    # Populate the raw low-res mask at the position corresponding to the box
    # (raw mask grid = canvas / 4, since 160 * 4 == 640).
    rx1, ry1, rx2, ry2 = (int(v / 4) for v in box_canvas)
    raw_mask[ry1:ry2, rx1:rx2] = 1.0
    raw_masks = raw_mask[np.newaxis, ...]  # (1, 160, 160)

    bag_bodies, print_marks = postprocess_rfdetr_seg(
        boxes=boxes,
        scores=scores,
        classes=classes,
        raw_masks=raw_masks,
        orig_shape=(orig_h, orig_w),
        scale=scale,
        pad=(pad_w, pad_h),
        conf_threshold=0.40,
        mask_threshold=0.50,
        canvas_size=canvas_size,
    )

    assert len(bag_bodies) == 1
    bag = bag_bodies[0]
    det_box = bag["box"]  # already inverted to original-image coordinates
    mask = bag["mask"]

    assert mask.shape == (orig_h, orig_w)
    assert mask.any(), "Decoded mask must not be empty"

    mask_bbox = _mask_bbox(mask)
    assert mask_bbox is not None

    # The mask must genuinely overlap the detection's own box -- this is the
    # crux of the regression test. Before the fix, the mask bbox landed near
    # (0, 0) regardless of the detection box, giving ~0 IoU.
    iou = _iou(det_box, mask_bbox)
    assert iou > 0.5, (
        f"Decoded mask bbox {mask_bbox} does not overlap detection box "
        f"{det_box} (IoU={iou:.3f}) -- mask is likely stuck in the corner"
    )

    # Explicitly guard against the old corner-stranding bug: the mask must not
    # be sitting up near the origin while the real box is far away from it.
    assert not (mask_bbox[0] < 50 and mask_bbox[1] < 50), (
        "Decoded mask is pinned near the top-left corner instead of the "
        "detection's actual location"
    )


def test_bbox_fallback_mask_used_when_no_raw_masks():
    """When raw_masks is None, mask should fall back to filling the box region."""
    boxes = np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([0.0], dtype=np.float32)

    bag_bodies, _ = postprocess_rfdetr_seg(
        boxes=boxes,
        scores=scores,
        classes=classes,
        raw_masks=None,
        orig_shape=(640, 640),
        scale=1.0,
        pad=(0, 0),
    )

    assert len(bag_bodies) == 1
    mask = bag_bodies[0]["mask"]
    assert mask[150, 150]  # inside the box
    assert not mask[0, 0]  # outside the box
