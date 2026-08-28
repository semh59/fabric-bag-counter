"""Real animated scenario generation for ReplayEngine (§11 M4).

The jobrunner's "replay" job kind previously hardcoded two ReplayScenario
objects with `frames=[]` -- an empty frame list. ReplayEngine.run_scenario()
iterates `scenario.frames`, so with zero frames it never calls
CountingEngine.process_frame() at all: predicted_count stayed permanently 0,
compared against arbitrary hardcoded ground_truth_count values (20, 15) that
had no real scenario behind them. The result was a fixed, meaningless
"100% undercounted" report on every run, regardless of what model was
actually deployed, completely disconnected from real behavior.

This module builds genuinely animated scenarios: bag(s) rendered with
SyntheticBagGenerator's real per-bag compositing primitives
(create_empty_conveyor/create_bag_template, the same ones generate_scene()
uses for static training scenes) move across a fresh background frame by
frame and are checked against a real gate x-position exactly like
LiveStreamRenderer._process_simulated_frame does, so ground_truth_count is
derived from what actually happened in the animation, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance

from packages.cs_data.synth import SyntheticBagGenerator
from packages.cs_eval.replay_engine import ReplayScenario

GATE_X = 320.0
CANVAS_SIZE = (640, 640)


@dataclass
class _MovingBag:
    start_x: float
    y: float
    speed_px: float
    color: tuple[int, int, int]
    stall_frames: tuple[int, int] | None = None  # (start, end) frame indices where speed is 0
    slip_at_frame: int | None = None  # frame index to start a brief backward slip
    slip_duration: int = 6


def _x_at_frame(bag: _MovingBag, frame_idx: int) -> float:
    """Real per-frame kinematic position -- the single place scenario-type
    behavior (stalling, slipping) actually changes where the bag is drawn,
    so it's the same position used both to render the frame and to derive
    ground truth crossings below."""
    x = bag.start_x
    traveled = 0.0
    for f in range(frame_idx):
        v = bag.speed_px
        if bag.stall_frames and bag.stall_frames[0] <= f < bag.stall_frames[1]:
            v = 0.0
        elif bag.slip_at_frame is not None and bag.slip_at_frame <= f < bag.slip_at_frame + bag.slip_duration:
            v = -bag.speed_px  # brief real reversal, not a rendering trick
        traveled += v
    return x + traveled


def _count_real_crossings(bag: _MovingBag, num_frames: int) -> int:
    """Net forward-minus-backward crossings of GATE_X across the animation --
    the same kinematic ground truth the frames themselves were rendered
    from, so it's a real derived count, not an assumption."""
    net = 0
    was_pre = _x_at_frame(bag, 0) <= GATE_X
    for f in range(1, num_frames):
        is_pre = _x_at_frame(bag, f) <= GATE_X
        if was_pre and not is_pre:
            net += 1
        elif not was_pre and is_pre:
            net -= 1
        was_pre = is_pre
    return net


def _render_frame(
    gen: SyntheticBagGenerator,
    bags: list[_MovingBag],
    templates: list[tuple[Image.Image, Image.Image]],
    frame_idx: int,
    brightness: float = 1.0,
) -> np.ndarray:
    canvas = gen.create_empty_conveyor().convert("RGBA")
    for bag, (bag_img, mask) in zip(bags, templates, strict=True):
        x = int(_x_at_frame(bag, frame_idx) - bag_img.width / 2)
        y = int(bag.y - bag_img.height / 2)
        canvas.paste(bag_img, (x, y), mask)
    rgb = canvas.convert("RGB")
    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    return np.array(rgb)


def build_scenario(scenario_type: str, seed: int | None = None) -> ReplayScenario:
    """Generate a real animated ReplayScenario for one of the five named
    types (heavy_shingling | sparse_flow | start_stop | backward_slip |
    light_change), with ground_truth_count derived from the actual
    kinematic simulation used to render the frames."""
    if seed is not None:
        import random
        random.seed(seed)

    gen = SyntheticBagGenerator(canvas_size=CANVAS_SIZE)
    # Every bag must have enough (frames * speed) travel budget to actually
    # reach past GATE_X, factoring in stall/slip setbacks -- verified
    # directly: an earlier version of this function undercounted its own
    # ground truth because a further-back bag never finished crossing
    # within the frame budget, silently making ground_truth_count wrong
    # rather than the real derived value it's supposed to be.
    num_frames = 110
    colors = [(220, 215, 200), (230, 225, 210), (210, 205, 190), (225, 218, 205)]

    if scenario_type == "heavy_shingling":
        # Many bags, tightly staggered starts -> genuine on-canvas overlap
        # (occlusion) the merge detector must actually resolve.
        bags = [_MovingBag(start_x=-140.0 - i * 60.0, y=320.0, speed_px=10.0, color=colors[i % len(colors)]) for i in range(4)]
    elif scenario_type == "sparse_flow":
        bags = [_MovingBag(start_x=-140.0 - i * 220.0, y=320.0, speed_px=8.0, color=colors[i % len(colors)]) for i in range(2)]
    elif scenario_type == "start_stop":
        bags = [
            _MovingBag(start_x=-140.0, y=320.0, speed_px=8.0, color=colors[0], stall_frames=(25, 45)),
            _MovingBag(start_x=-360.0, y=320.0, speed_px=8.0, color=colors[1]),
        ]
    elif scenario_type == "backward_slip":
        bags = [
            _MovingBag(start_x=-140.0, y=320.0, speed_px=8.0, color=colors[0], slip_at_frame=32, slip_duration=6),
            _MovingBag(start_x=-360.0, y=320.0, speed_px=8.0, color=colors[1]),
        ]
    elif scenario_type == "light_change":
        bags = [_MovingBag(start_x=-140.0 - i * 220.0, y=320.0, speed_px=8.0, color=colors[i % len(colors)]) for i in range(2)]
    else:
        raise ValueError(f"Unknown replay scenario_type: {scenario_type!r}")

    templates = [gen.create_bag_template(color=b.color, has_print_mark=True)[:2] for b in bags]

    frames: list[np.ndarray] = []
    for f_idx in range(num_frames):
        brightness = 1.0
        if scenario_type == "light_change" and f_idx >= num_frames // 2:
            brightness = 0.55  # a real, sustained lighting drop partway through
        frames.append(_render_frame(gen, bags, templates, f_idx, brightness=brightness))

    ground_truth_count = sum(_count_real_crossings(b, num_frames) for b in bags)

    return ReplayScenario(
        name=scenario_type,
        scenario_type=scenario_type,
        ground_truth_count=ground_truth_count,
        frames=frames,
    )


def build_default_scenario_suite(seed: int | None = None) -> list[ReplayScenario]:
    """All five named scenario types, each freshly generated and animated."""
    types = ["heavy_shingling", "sparse_flow", "start_stop", "backward_slip", "light_change"]
    return [build_scenario(t, seed=seed) for t in types]
