"""Unit tests for Temporal Amodal Mask Reconstruction."""

import numpy as np
import pytest
from packages.cs_tracking.amodal_reconstruction import TemporalAmodalReconstructor


def test_amodal_reconstructor_records_and_projects_trajectory():
    reconstructor = TemporalAmodalReconstructor(max_history_frames=10)

    # 1. Unoccluded observation in frame 0
    box0 = [100.0, 100.0, 200.0, 200.0]
    mask0 = np.zeros((640, 640), dtype=bool)
    mask0[100:200, 100:200] = True
    reconstructor.record_observation(track_id=1, frame_index=0, box=box0, mask=mask0, is_isolated=True)

    # 2. Reconstruct mask in frame 5 after bag moved to x=300
    cur_box = [300.0, 100.0, 400.0, 200.0]
    # Partially occluded visible mask
    cur_vis_mask = np.zeros((640, 640), dtype=bool)
    cur_vis_mask[100:150, 300:400] = True  # only top half visible

    recon_mask = reconstructor.reconstruct_amodal_mask(
        track_id=1,
        current_box=cur_box,
        current_visible_mask=cur_vis_mask,
        canvas_shape=(640, 640),
    )

    assert recon_mask.shape == (640, 640)
    # The reconstructed mask restores the bottom half (300 to 400 in X, 100 to 200 in Y)
    assert np.sum(recon_mask[150:200, 300:400]) > 0
