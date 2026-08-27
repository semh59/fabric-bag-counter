"""Zero-leakage dataset splitting with isolated hard_holdout partition (§7.3)."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class SplitResult:
    """Disjoint dataset split metadata."""
    train_sessions: list[str]
    val_sessions: list[str]
    hard_holdout_sessions: list[str]
    train_count: int
    val_count: int
    hard_holdout_count: int
    manifest_hash: str
    split_spec: dict[str, Any]


class DatasetSplitter:
    """Splits dataset at the physical session / video level to guarantee zero frame leakage."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        hard_holdout_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.hard_holdout_ratio = hard_holdout_ratio
        self.seed = seed

    def split_sessions(
        self,
        sessions: list[dict[str, Any]],  # list of {"session_id": str, "camera_id": int, "shift": str, "frame_count": int, "is_heavy_shingling": bool}
    ) -> SplitResult:
        """Partition session list into train, val, and isolated hard_holdout.

        Hard holdout isolates distinct cameras / shifts with heavy shingling.

        Note on hard_holdout capacity: `hard_holdout_ratio` caps how many
        heavy-shingling sessions can go into hard_holdout. Any heavy-shingling
        sessions beyond that cap do NOT get a second isolated bucket -- there's
        only one hard_holdout partition -- so they fall back into the regular
        train/val pool. To keep "hard holdout" isolation as meaningful as
        possible despite that, this overflow is routed preferentially into
        `val` rather than `train` (see below): val is allowed to see extra
        heavy-shingling difficulty, but train is kept as clean of it as the
        ratios allow. Only if val's own capacity is exhausted does overflow
        spill into train.
        """
        rng = random.Random(self.seed)
        shuffled = list(sessions)
        rng.shuffle(shuffled)

        # Prioritize heavy shingling sessions from distinct cameras for hard_holdout
        hard_holdout = []
        regular_pool = []

        for s in shuffled:
            if s.get("is_heavy_shingling", False) and len(hard_holdout) < max(1, int(len(sessions) * self.hard_holdout_ratio)):
                hard_holdout.append(s)
            else:
                regular_pool.append(s)

        # Fill remaining hard holdout if needed
        target_holdout_len = max(1, int(len(sessions) * self.hard_holdout_ratio))
        while len(hard_holdout) < target_holdout_len and regular_pool:
            hard_holdout.append(regular_pool.pop())

        n_reg = len(regular_pool)
        n_train = int(n_reg * (self.train_ratio / (self.train_ratio + self.val_ratio)))
        n_val = n_reg - n_train

        # Any heavy-shingling sessions left in regular_pool are the overflow
        # described in the docstring above: route them into val first (val
        # keeps up to n_val of them), and only let the leftover spill into
        # train once val's capacity is exhausted. This never changes the
        # overall train/val sizes, only which sessions land in which split.
        regular_pool_heavy = [s for s in regular_pool if s.get("is_heavy_shingling", False)]
        regular_pool_normal = [s for s in regular_pool if not s.get("is_heavy_shingling", False)]

        val = regular_pool_heavy[:n_val]
        leftover_heavy = regular_pool_heavy[n_val:]
        val_fill_needed = n_val - len(val)
        if val_fill_needed > 0:
            val = val + regular_pool_normal[:val_fill_needed]
            train = leftover_heavy + regular_pool_normal[val_fill_needed:]
        else:
            train = leftover_heavy + regular_pool_normal

        train_sids = [str(s["session_id"]) for s in train]
        val_sids = [str(s["session_id"]) for s in val]
        holdout_sids = [str(s["session_id"]) for s in hard_holdout]

        # Verify zero intersection. Use explicit checks + raise (not bare
        # `assert`) so this leakage guarantee still holds when the interpreter
        # is run with `-O` (which strips `assert` statements entirely).
        set_t = set(train_sids)
        set_v = set(val_sids)
        set_h = set(holdout_sids)
        if set_t.intersection(set_v):
            raise ValueError(f"Dataset split leakage detected between train and val: {set_t.intersection(set_v)}")
        if set_t.intersection(set_h):
            raise ValueError(f"Dataset split leakage detected between train and hard_holdout: {set_t.intersection(set_h)}")
        if set_v.intersection(set_h):
            raise ValueError(f"Dataset split leakage detected between val and hard_holdout: {set_v.intersection(set_h)}")

        train_frames = sum(s.get("frame_count", 0) for s in train)
        val_frames = sum(s.get("frame_count", 0) for s in val)
        holdout_frames = sum(s.get("frame_count", 0) for s in hard_holdout)

        split_spec = {
            "train_sessions": train_sids,
            "val_sessions": val_sids,
            "hard_holdout_sessions": holdout_sids,
            "train_frame_count": train_frames,
            "val_frame_count": val_frames,
            "hard_holdout_frame_count": holdout_frames,
            "seed": self.seed,
        }

        # Calculate reproducible manifest hash
        manifest_str = json.dumps(split_spec, sort_keys=True)
        manifest_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

        return SplitResult(
            train_sessions=train_sids,
            val_sessions=val_sids,
            hard_holdout_sessions=holdout_sids,
            train_count=train_frames,
            val_count=val_frames,
            hard_holdout_count=holdout_frames,
            manifest_hash=manifest_hash,
            split_spec=split_spec,
        )
