"""Unit tests for GateStateMachine, crossing_seq, and backward slip handling (§5.5, §6.8)."""

from datetime import datetime, timezone
from packages.cs_counting.gate import GateStateMachine
from packages.cs_tracking.tracker import BagTrack


def test_gate_state_machine_forward_crossing():
    BagTrack.reset_counter()
    gate = GateStateMachine(
        gate_id=1,
        axis_origin=(0.0, 0.0),
        axis_vector=(1.0, 0.0),
        gate_position_along_axis=100.0,
        pre_gate_offset=30.0,
        post_gate_offset=30.0,
    )

    track = BagTrack(box=[50, 0, 70, 20], score=0.9)  # cx = 60 (PRE zone)
    t_now = datetime.now(timezone.utc)

    # Step 1: In PRE zone
    events1 = gate.process_tracks([track], frame_index=1, monotonic_ns=1000, wall_clock=t_now)
    assert len(events1) == 0

    # Step 2: Crosses gate into POST zone (cx = 120)
    track.centroid = (120.0, 10.0)
    events2 = gate.process_tracks([track], frame_index=2, monotonic_ns=2000, wall_clock=t_now)
    assert len(events2) == 1
    assert events2[0].direction == 1
    assert events2[0].crossing_seq == 1
    assert track.crossing_seq == 1


def test_gate_backward_slip_net_counting():
    """Verify forward -> backward -> forward slip results in net count of 1 and crossing_seq 1,2,3."""
    BagTrack.reset_counter()
    gate = GateStateMachine(
        gate_id=1,
        axis_origin=(0.0, 0.0),
        axis_vector=(1.0, 0.0),
        gate_position_along_axis=100.0,
        pre_gate_offset=30.0,
        post_gate_offset=30.0,
    )

    track = BagTrack(box=[50, 0, 70, 20], score=0.9)  # cx = 60 (PRE)
    t_now = datetime.now(timezone.utc)
    gate.process_tracks([track], frame_index=1, monotonic_ns=1000, wall_clock=t_now)

    # 1. Forward crossing (cx = 120)
    track.centroid = (120.0, 10.0)
    ev_fwd1 = gate.process_tracks([track], frame_index=2, monotonic_ns=2000, wall_clock=t_now)
    assert len(ev_fwd1) == 1
    assert ev_fwd1[0].direction == 1
    assert ev_fwd1[0].crossing_seq == 1

    # 2. Conveyor slips backward (cx = 70, back in PRE)
    track.centroid = (70.0, 10.0)
    ev_back = gate.process_tracks([track], frame_index=3, monotonic_ns=3000, wall_clock=t_now)
    assert len(ev_back) == 1
    assert ev_back[0].direction == -1
    assert ev_back[0].crossing_seq == 2

    # 3. Conveyor resumes forward (cx = 130, back in POST)
    track.centroid = (130.0, 10.0)
    ev_fwd2 = gate.process_tracks([track], frame_index=4, monotonic_ns=4000, wall_clock=t_now)
    assert len(ev_fwd2) == 1
    assert ev_fwd2[0].direction == 1
    assert ev_fwd2[0].crossing_seq == 3

    # Total net count = +1 - 1 + 1 = 1
    net_count = ev_fwd1[0].direction + ev_back[0].direction + ev_fwd2[0].direction
    assert net_count == 1
