"""Default configuration values and backward-compatible schema migration helpers."""

from __future__ import annotations

from typing import Any

CURRENT_PAYLOAD_SCHEMA_VERSION = 2

# Schema version 1 defaults
SCHEMA_V1_DEFAULTS: dict[str, Any] = {
    "roi_polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    "gate_line": [[0.5, 0.0], [0.5, 1.0]],
    "pre_gate_zone": [[0.3, 0.0], [0.5, 0.0], [0.5, 1.0], [0.3, 1.0]],
    "post_gate_zone": [[0.5, 0.0], [0.7, 0.0], [0.7, 1.0], [0.5, 1.0]],
    "confidence_threshold": 0.45,
    "mask_iou_threshold": 0.40,
    "merge_area_ratio": 1.50,
    "discrepancy_threshold": 0.08,  # 8% discrepancy trigger
    "ring_slots": 8,
    "batch_wait_ms": 30,
    "max_consecutive_drops": 3,
    "warning_light_threshold": 0.90,  # 90% of target
    "gpu_sharing_mode": "strict",
}

# Schema version 2 additions
SCHEMA_V2_DEFAULTS: dict[str, Any] = {
    **SCHEMA_V1_DEFAULTS,
    "merge_signals": {
        "area_enabled": True,
        "shape_enabled": True,
        "temporal_enabled": True,
        "print_mark_enabled": True,
        "min_votes": 2,
    },
    "latent_track_grace_frames": 5,
    "tracking_cost_weights": {
        "mask_iou": 0.7,
        "centroid_distance": 0.3,
    },
    "area_integral": {
        "min_confidence": 0.30,
        "smoothing_window": 10,
    },
    "latency_p95_pause_threshold_ms": 120.0,
}


class UnknownSchemaVersionError(ValueError):
    """Raised when a payload declares a schema version newer than this build understands."""


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overrides` onto `base` at every nesting level.

    Any key present in both where both values are dicts is merged
    recursively (not just shallowly replaced); any other key is simply
    overridden by `overrides`'s value, including when the types differ
    (e.g. overriding a dict default with a non-dict payload value, or vice
    versa) -- in that case the override wins outright, matching how a
    config payload is expected to take precedence over defaults.
    """
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def get_config_with_defaults(
    payload: dict[str, Any] | None,
    schema_version: int = CURRENT_PAYLOAD_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Merge raw config payload with documented schema defaults to guarantee backward compatibility."""
    if schema_version <= 1:
        base = dict(SCHEMA_V1_DEFAULTS)
    elif schema_version == 2:
        base = dict(SCHEMA_V2_DEFAULTS)
    else:
        raise UnknownSchemaVersionError(
            f"Unrecognized payload_schema_version={schema_version!r}; this build only knows "
            f"schema versions up to {CURRENT_PAYLOAD_SCHEMA_VERSION}. Refusing to guess at "
            "defaults for a newer/unknown schema -- upgrade cs_core or pin an older "
            "payload_schema_version."
        )

    if not payload:
        return base

    return _deep_merge(base, payload)
