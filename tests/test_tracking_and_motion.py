"""Unit tests for tracking, BeltMotionModel, and ByteTrack association (§6.6)."""

import numpy as np
from packages.cs_tracking.matching import associate_detections_to_tracks, compute_cost_matrix
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
