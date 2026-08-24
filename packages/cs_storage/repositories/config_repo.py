"""Repository for Immutable Config Versions and Deployment Bundles (§5.4)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_core.config_defaults import CURRENT_PAYLOAD_SCHEMA_VERSION, get_config_with_defaults
from packages.cs_storage.models_orm import ConfigVersionORM, DeploymentBundleORM, ModelVersionORM


class ConfigRepository:
    """Manages immutable configuration versions and active deployment bundles."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_config_version(
        self,
        line_id: int,
        payload: dict[str, Any],
        note: str | None = None,
        created_by: str | None = None,
        payload_schema_version: int = CURRENT_PAYLOAD_SCHEMA_VERSION,
    ) -> ConfigVersionORM:
        """Store a new immutable configuration version."""
        cfg = ConfigVersionORM(
            line_id=line_id,
            payload=payload,
            payload_schema_version=payload_schema_version,
            note=note,
            created_by=created_by,
            created_at=datetime.utcnow(),
        )
        self.db.add(cfg)
        self.db.commit()
        self.db.refresh(cfg)
        return cfg

    def get_latest_config(self, line_id: int) -> ConfigVersionORM | None:
        """Fetch the latest config version for a line."""
        stmt = (
            select(ConfigVersionORM)
            .where(ConfigVersionORM.line_id == line_id)
            .order_by(ConfigVersionORM.created_at.desc())
        )
        return self.db.execute(stmt).scalars().first()

    def get_effective_config_payload(self, config_version: ConfigVersionORM) -> dict[str, Any]:
        """Return config payload enriched with schema version defaults."""
        return get_config_with_defaults(
            config_version.payload,
            schema_version=config_version.payload_schema_version,
        )

    def create_and_activate_bundle(
        self,
        line_id: int,
        model_version_id: int,
        config_version_id: int,
        calibration_id: int | None = None,
        git_commit: str | None = None,
        activated_by: str | None = None,
    ) -> DeploymentBundleORM:
        """Activate a new deployment bundle, deactivating previous active bundle."""
        now = datetime.utcnow()
        # Deactivate existing active bundles for this line
        prev_bundles = self.db.execute(
            select(DeploymentBundleORM).where(
                DeploymentBundleORM.line_id == line_id,
                DeploymentBundleORM.deactivated_at == None,  # noqa: E711
            )
        ).scalars().all()
        for b in prev_bundles:
            b.deactivated_at = now

        bundle = DeploymentBundleORM(
            line_id=line_id,
            model_version_id=model_version_id,
            config_version_id=config_version_id,
            calibration_id=calibration_id,
            git_commit=git_commit,
            activated_at=now,
            activated_by=activated_by,
        )
        self.db.add(bundle)
        self.db.commit()
        self.db.refresh(bundle)
        return bundle

    def get_active_bundle(self, line_id: int) -> DeploymentBundleORM | None:
        """Fetch the current active deployment bundle for a line."""
        stmt = (
            select(DeploymentBundleORM)
            .where(
                DeploymentBundleORM.line_id == line_id,
                DeploymentBundleORM.deactivated_at == None,  # noqa: E711
            )
            .order_by(DeploymentBundleORM.activated_at.desc())
        )
        return self.db.execute(stmt).scalars().first()
