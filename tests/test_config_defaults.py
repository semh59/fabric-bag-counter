"""Unit tests for config_defaults' recursive merge and schema version handling."""

import pytest
from packages.cs_core.config_defaults import (
    CURRENT_PAYLOAD_SCHEMA_VERSION,
    UnknownSchemaVersionError,
    get_config_with_defaults,
)


def test_recursive_merge_preserves_untouched_nested_keys():
    """A payload that only overrides ONE key inside a nested dict default must
    not wipe out the other sibling keys at that same nesting level (a
    shallow `{**base[key], **override[key]}` merge only handles one level
    deep -- this must hold for deeper nesting too)."""
    payload = {
        "merge_signals": {
            # Only override one of the five default merge_signals keys.
            "min_votes": 5,
        },
    }
    merged = get_config_with_defaults(payload, schema_version=2)

    # The overridden key changed...
    assert merged["merge_signals"]["min_votes"] == 5
    # ...but sibling default keys inside the same nested dict survived.
    assert merged["merge_signals"]["area_enabled"] is True
    assert merged["merge_signals"]["shape_enabled"] is True
    assert merged["merge_signals"]["temporal_enabled"] is True
    assert merged["merge_signals"]["print_mark_enabled"] is True

    # Other top-level nested dicts untouched by the payload survive fully.
    assert merged["tracking_cost_weights"] == {"mask_iou": 0.7, "centroid_distance": 0.3}
    assert merged["area_integral"] == {"min_confidence": 0.30, "smoothing_window": 10}


def test_recursive_merge_handles_two_levels_of_nesting():
    """Merge correctness at a nesting depth a shallow single-level merge would
    already get wrong is what makes this genuinely 'recursive' rather than
    just 'merge one level of dicts'."""
    base_payload = {"area_integral": {"min_confidence": 0.5}}
    merged = get_config_with_defaults(base_payload, schema_version=2)
    assert merged["area_integral"]["min_confidence"] == 0.5
    assert merged["area_integral"]["smoothing_window"] == 10  # untouched sibling survives


def test_non_dict_override_replaces_outright():
    """A scalar override for a top-level key still just replaces the default,
    same as before."""
    merged = get_config_with_defaults({"confidence_threshold": 0.9}, schema_version=2)
    assert merged["confidence_threshold"] == 0.9


def test_unrecognized_future_schema_version_raises_clear_error():
    """A payload_schema_version newer than anything this build knows about
    must raise a clear, specific error instead of silently guessing at
    defaults (e.g. by falling through to the latest known schema)."""
    with pytest.raises(UnknownSchemaVersionError):
        get_config_with_defaults({}, schema_version=CURRENT_PAYLOAD_SCHEMA_VERSION + 1)


def test_known_versions_still_work():
    assert get_config_with_defaults(None, schema_version=1)["ring_slots"] == 8
    assert get_config_with_defaults(None, schema_version=2)["latent_track_grace_frames"] == 5
