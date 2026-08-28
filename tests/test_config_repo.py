"""Tests for ConfigRepository (§5.4) -- previously had zero coverage.

Covers the version/bundle resolution path that CountingEngine.configure()
now depends on: create_config_version, get_effective_config_payload,
create_and_activate_bundle, get_active_bundle.
"""

from packages.cs_storage.db import get_sync_session, init_db_sync
from packages.cs_storage.models_orm import LineORM, ModelVersionORM, SiteORM
from packages.cs_storage.repositories.config_repo import ConfigRepository


def _setup_line() -> tuple[int, int]:
    init_db_sync()
    with get_sync_session() as db:
        site = SiteORM(name="Config Repo Test Site")
        db.add(site)
        db.commit()
        line = LineORM(site_id=site.id, name="Line 1")
        db.add(line)
        db.commit()
        mv = ModelVersionORM(onnx_hash="test-hash", onnx_path="models/test.onnx")
        db.add(mv)
        db.commit()
        return line.id, mv.id


def test_create_config_version_and_get_latest():
    line_id, _ = _setup_line()
    with get_sync_session() as db:
        repo = ConfigRepository(db)
        cfg = repo.create_config_version(line_id=line_id, payload={"confidence_threshold": 0.7}, note="test")
        assert cfg.id is not None

        latest = repo.get_latest_config(line_id)
        assert latest.id == cfg.id
        assert latest.payload == {"confidence_threshold": 0.7}


def test_get_effective_config_payload_merges_onto_defaults():
    line_id, _ = _setup_line()
    with get_sync_session() as db:
        repo = ConfigRepository(db)
        cfg = repo.create_config_version(line_id=line_id, payload={"confidence_threshold": 0.7})
        effective = repo.get_effective_config_payload(cfg)

        assert effective["confidence_threshold"] == 0.7
        # Untouched keys still come from the schema defaults.
        assert effective["merge_area_ratio"] == 1.50
        assert effective["roi_polygon"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_create_and_activate_bundle_deactivates_previous():
    line_id, model_id = _setup_line()
    with get_sync_session() as db:
        repo = ConfigRepository(db)
        cfg1 = repo.create_config_version(line_id=line_id, payload={})
        cfg2 = repo.create_config_version(line_id=line_id, payload={"confidence_threshold": 0.9})

        bundle1 = repo.create_and_activate_bundle(line_id=line_id, model_version_id=model_id, config_version_id=cfg1.id)
        assert repo.get_active_bundle(line_id).id == bundle1.id

        bundle2 = repo.create_and_activate_bundle(line_id=line_id, model_version_id=model_id, config_version_id=cfg2.id)

        db.refresh(bundle1)
        assert bundle1.deactivated_at is not None
        active = repo.get_active_bundle(line_id)
        assert active.id == bundle2.id
        assert active.config_version_id == cfg2.id


def test_get_active_bundle_none_when_never_activated():
    line_id, _ = _setup_line()
    with get_sync_session() as db:
        repo = ConfigRepository(db)
        assert repo.get_active_bundle(line_id) is None


def test_active_bundle_config_version_relationship_resolves_effective_payload():
    """The exact path CountingEngine.configure() wiring depends on:
    get_active_bundle(...).config_version -> get_effective_config_payload(...)."""
    line_id, model_id = _setup_line()
    with get_sync_session() as db:
        repo = ConfigRepository(db)
        cfg = repo.create_config_version(line_id=line_id, payload={"discrepancy_threshold": 0.2})
        repo.create_and_activate_bundle(line_id=line_id, model_version_id=model_id, config_version_id=cfg.id)

        bundle = repo.get_active_bundle(line_id)
        payload = repo.get_effective_config_payload(bundle.config_version)
        assert payload["discrepancy_threshold"] == 0.2
