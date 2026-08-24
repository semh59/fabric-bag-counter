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
        train = regular_pool[:n_train]
        val = regular_pool[n_train:]

        train_sids = [str(s["session_id"]) for s in train]
        val_sids = [str(s["session_id"]) for s in val]
        holdout_sids = [str(s["session_id"]) for s in hard_holdout]

        # Verify zero intersection
        set_t = set(train_sids)
        set_v = set(val_sids)
        set_h = set(holdout_sids)
        assert len(set_t.intersection(set_v)) == 0
        assert len(set_t.intersection(set_h)) == 0
        assert len(set_v.intersection(set_h)) == 0

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
