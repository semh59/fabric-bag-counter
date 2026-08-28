"""Unit tests for tracking, BeltMotionModel, and ByteTrack association (§6.6)."""

import numpy as np
import pytest

from packages.cs_tracking.matching import compute_cost_matrix
from packages.cs_tracking.motion import BeltMotionModel
from packages.cs_tracking.tracker import BagTrack, ConveyorByteTracker


def test_belt_motion_model():
    motion = BeltMotionModel(default_speed_px=5.0, default_direction=[1.0, 0.0])
    prior = motion.get_velocity_prior()
    assert prior[0] == 5.0 and prior[1] == 0.0

    # Test update from calibration
    motion.update_from_calibration(speed_px=8.0, direction=[0.0, 1.0])
    prior2 = motion.get_velocity_prior()
    assert abs(prior2[0]) < 0.01 and abs(prior2[1] - 8.0) < 0.01


def test_bag_track_lifecycle():
    BagTrack.reset_counter()
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True

    track = BagTrack(box=[10, 10, 30, 30], score=0.9, mask=mask)
    assert track.track_id == 1
    assert track.state == "tentative"
    assert track.crossing_seq == 0

    # Predict with conveyor prior
    track.predict(belt_velocity_prior=np.array([5.0, 0.0], dtype=np.float32))
    assert track.centroid[0] > 20.0

    # Update with new measurement
    track.update(box=[15, 10, 35, 30], score=0.92, mask=mask)
    assert track.state == "confirmed"
    assert track.hits == 2


def test_conveyor_byte_tracker_association():
    BagTrack.reset_counter()
    motion = BeltMotionModel(default_speed_px=10.0, default_direction=[1.0, 0.0])
    tracker = ConveyorByteTracker(belt_motion=motion)

    # Frame 1: Detection of single bag
    m1 = np.zeros((100, 100), dtype=bool)
    m1[20:40, 20:40] = True
    dets_f1 = [{"box": [20, 20, 40, 40], "score": 0.85, "mask": m1}]
    tracks1 = tracker.update(dets_f1)
    assert len(tracks1) == 1

    # Frame 2: Bag moved 10px forward along X axis
    m2 = np.zeros((100, 100), dtype=bool)
    m2[20:40, 30:50] = True
    dets_f2 = [{"box": [30, 20, 50, 40], "score": 0.88, "mask": m2}]
    tracks2 = tracker.update(dets_f2)

    assert len(tracks2) == 1
    assert tracks2[0].track_id == tracks1[0].track_id  # Continuity verified!
    assert tracks2[0].state == "confirmed"


def test_compute_cost_matrix_weights_have_real_effect():
    """w_mask/w_dist (see CountingEngine.configure()'s tracking_cost_weights
    wiring) must actually change the computed cost, matching the documented
    formula cost = w_mask*mask_cost + w_dist*norm_dist -- not just be
    accepted and ignored."""
    BagTrack.reset_counter()
    track_mask = np.zeros((100, 100), dtype=bool)
    track_mask[0:10, 0:10] = True  # disjoint from det_mask below -> IoU=0, mask_cost=1.0
    track = BagTrack(box=[0, 0, 10, 10], score=0.9, mask=track_mask)

    det_mask = np.zeros((100, 100), dtype=bool)
    det_mask[50:60, 50:60] = True
    # Centroid distance: track at (5,5), detection box centered at (80,5) -> dist=75, norm_dist=0.5 at max_distance_px=150
    det = {"box": [75.0, 0.0, 85.0, 10.0], "mask": det_mask}

    cost_mask_heavy = compute_cost_matrix([track], [det], w_mask=1.0, w_dist=0.0)[0, 0]
    cost_dist_heavy = compute_cost_matrix([track], [det], w_mask=0.0, w_dist=1.0)[0, 0]

    assert cost_mask_heavy == pytest.approx(1.0, abs=1e-3)  # pure mask_cost (IoU=0)
    assert cost_dist_heavy == pytest.approx(0.5, abs=1e-3)  # pure norm_dist
    assert cost_mask_heavy != cost_dist_heavy


def test_max_time_lost_from_config_actually_changes_pruning():
    """latent_track_grace_frames (mapped onto ConveyorByteTracker.
    max_time_lost, see CountingEngine.configure()) must actually change how
    long a lost track survives before being pruned -- not just be stored."""
    BagTrack.reset_counter()
    tracker = ConveyorByteTracker(max_time_lost=2)

    m1 = np.zeros((100, 100), dtype=bool)
    m1[20:40, 20:40] = True
    tracker.update([{"box": [20, 20, 40, 40], "score": 0.9, "mask": m1}])
    assert len(tracker.tracked_tracks) == 1

    # No detections for 3 frames -> track goes lost, then exceeds max_time_lost=2 and is pruned.
    for _ in range(3):
        tracker.update([])

    assert len(tracker.lost_tracks) == 0  # pruned
    assert len(tracker.tracked_tracks) == 0
