"""SQLAlchemy 2.0 ORM models corresponding to the immutable schema specification."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from packages.cs_storage.db import Base


class SiteORM(Base):
    __tablename__ = "site"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), default="Europe/Istanbul")
    locale: Mapped[str] = mapped_column(String(16), default="tr_TR")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    lines = relationship("LineORM", back_populates="site", cascade="all, delete-orphan")
    nodes = relationship("NodeORM", back_populates="site", cascade="all, delete-orphan")
    product_profiles = relationship("ProductProfileORM", back_populates="site", cascade="all, delete-orphan")


class NodeORM(Base):
    __tablename__ = "node"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(Integer, ForeignKey("site.id", ondelete="CASCADE"), nullable=False)
    hostname: Mapped[str] = mapped_column(String(255), nullable=False)
    gpu_info: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="online")
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    site = relationship("SiteORM", back_populates="nodes")
    cameras = relationship("CameraORM", back_populates="node")


class LineORM(Base):
    __tablename__ = "line"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(Integer, ForeignKey("site.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="idle")
    maintenance_window: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    site = relationship("SiteORM", back_populates="lines")
    cameras = relationship("CameraORM", back_populates="line", cascade="all, delete-orphan")
    gates = relationship("GateORM", back_populates="line", cascade="all, delete-orphan")
    calibrations = relationship("LineCalibrationORM", back_populates="line", cascade="all, delete-orphan")
    configs = relationship("ConfigVersionORM", back_populates="line", cascade="all, delete-orphan")
    bundles = relationship("DeploymentBundleORM", back_populates="line", cascade="all, delete-orphan")
    sessions = relationship("SessionORM", back_populates="line")


class CameraORM(Base):
    __tablename__ = "camera"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id: Mapped[int] = mapped_column(Integer, ForeignKey("line.id", ondelete="CASCADE"), nullable=False)
    node_id: Mapped[int] = mapped_column(Integer, ForeignKey("node.id", ondelete="CASCADE"), nullable=False)
    source_driver: Mapped[str] = mapped_column(String(64), default="rtsp")
    source_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    role: Mapped[str] = mapped_column(String(32), default="counting")  # counting | vehicle_watchdog | auxiliary
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    line = relationship("LineORM", back_populates="cameras")
    node = relationship("NodeORM", back_populates="cameras")
    epoch_record = relationship("CameraEpochORM", back_populates="camera", uselist=False, cascade="all, delete-orphan")


class CameraEpochORM(Base):
    """Persistent stream epoch sequence per camera (§5.2)."""
    __tablename__ = "camera_epoch"

    camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("camera.id", ondelete="CASCADE"), primary_key=True)
    current_epoch: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    camera = relationship("CameraORM", back_populates="epoch_record")


class GateORM(Base):
    __tablename__ = "gate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id: Mapped[int] = mapped_column(Integer, ForeignKey("line.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    line = relationship("LineORM", back_populates="gates")


class ProductProfileORM(Base):
    __tablename__ = "product_profile"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(Integer, ForeignKey("site.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    nominal_weight_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    nominal_dims_mm: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    template_images: Mapped[list[str]] = mapped_column(JSON, default=list)
    erp_material_code: Mapped[str | None] = mapped_column(String(128), nullable=True)

    site = relationship("SiteORM", back_populates="product_profiles")
    sessions = relationship("SessionORM", back_populates="product_profile")


class LineCalibrationORM(Base):
    """Two-stage line calibration (§5.3)."""
    __tablename__ = "line_calibration"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id: Mapped[int] = mapped_column(Integer, ForeignKey("line.id", ondelete="CASCADE"), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)  # motion | scale
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Stage 1: motion
    belt_speed_px_per_frame: Mapped[float | None] = mapped_column(Float, nullable=True)
    belt_direction_vector: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    # Stage 2: scale
    px_per_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_bag_gate_area_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    bag_area_stddev_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_video_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_model_version_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    line = relationship("LineORM", back_populates="calibrations")


class DatasetVersionORM(Base):
    __tablename__ = "dataset_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(Integer, ForeignKey("site.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    synthetic_count: Mapped[int] = mapped_column(Integer, default=0)
    split_spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    annotation_guide_version: Mapped[str] = mapped_column(String(32), default="2.0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    training_runs = relationship("TrainingRunORM", back_populates="dataset_version")


class TrainingRunORM(Base):
    __tablename__ = "training_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("dataset_version.id", ondelete="CASCADE"), nullable=False)
    # Nullable: a base ("from scratch") training run has no prior model version to reference.
    # use_alter=True: this FK and ModelVersionORM.training_run_id form a genuine
    # cycle between the two tables, so DDL must create/drop it via a separate
    # ALTER TABLE rather than inline (which SQLAlchemy can't topologically sort).
    base_model_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("model_version.id", ondelete="SET NULL", use_alter=True, name="fk_training_run_base_model_version"),
        nullable=True,
    )
    run_kind: Mapped[str] = mapped_column(String(32), default="base")  # base | site_adaptation
    hyperparams: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    log_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    dataset_version = relationship("DatasetVersionORM", back_populates="training_runs")
    # Two distinct FK paths now link training_run <-> model_version (the run's own
    # produced versions, via ModelVersionORM.training_run_id; and the base model
    # this run started from, via base_model_version_id) -- both relationships must
    # pin their foreign_keys explicitly or SQLAlchemy can't tell them apart.
    model_versions = relationship(
        "ModelVersionORM", back_populates="training_run", foreign_keys="ModelVersionORM.training_run_id"
    )
    base_model_version = relationship("ModelVersionORM", foreign_keys=[base_model_version_id])


class ModelVersionORM(Base):
    __tablename__ = "model_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("training_run.id", ondelete="SET NULL"), nullable=True)
    onnx_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    onnx_path: Mapped[str] = mapped_column(Text, nullable=False)
    eval_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    stage: Mapped[str] = mapped_column(String(32), default="draft")  # draft | shadow | active | retired
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    training_run = relationship(
        "TrainingRunORM", back_populates="model_versions", foreign_keys=[training_run_id]
    )
    bundles = relationship("DeploymentBundleORM", back_populates="model_version")


class ConfigVersionORM(Base):
    __tablename__ = "config_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id: Mapped[int] = mapped_column(Integer, ForeignKey("line.id", ondelete="CASCADE"), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    payload_schema_version: Mapped[int] = mapped_column(Integer, default=2)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    line = relationship("LineORM", back_populates="configs")
    bundles = relationship("DeploymentBundleORM", back_populates="config_version")


class DeploymentBundleORM(Base):
    __tablename__ = "deployment_bundle"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id: Mapped[int] = mapped_column(Integer, ForeignKey("line.id", ondelete="CASCADE"), nullable=False)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_version.id"), nullable=False)
    config_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("config_version.id"), nullable=False)
    calibration_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("line_calibration.id"), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    activated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    activated_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    line = relationship("LineORM", back_populates="bundles")
    model_version = relationship("ModelVersionORM", back_populates="bundles")
    config_version = relationship("ConfigVersionORM", back_populates="bundles")
    calibration = relationship("LineCalibrationORM")


class SessionORM(Base):
    __tablename__ = "session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Indexed: get_active_session()/list_sessions() filter/order on line_id on every
    # session lookup and every poll of the current line status.
    line_id: Mapped[int] = mapped_column(Integer, ForeignKey("line.id", ondelete="CASCADE"), nullable=False, index=True)
    product_profile_id: Mapped[int] = mapped_column(Integer, ForeignKey("product_profile.id"), nullable=False)
    external_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    target_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="open")  # open, counting, paused, degraded, closed, reconcile_required, reconciled
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    counted_total: Mapped[int] = mapped_column(Integer, default=0)
    area_estimate_total: Mapped[float] = mapped_column(Float, default=0.0)
    discrepancy_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    vehicle_plate: Mapped[str | None] = mapped_column(String(32), nullable=True)
    driver_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    carrier_company: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reconciliation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("reconciliation.id", use_alter=True, name="fk_session_reconciliation"), nullable=True
    )

    line = relationship("LineORM", back_populates="sessions")
    product_profile = relationship("ProductProfileORM", back_populates="sessions")
    events = relationship("CountEventORM", back_populates="session", cascade="all, delete-orphan")
    outbox_entries = relationship("OutboxORM", back_populates="session")
    reconciliation = relationship("ReconciliationORM", foreign_keys=[reconciliation_id], post_update=True)


class CountEventORM(Base):
    """Count Event Ledger (§5.5) — Single source of truth for counting."""
    __tablename__ = "count_event"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("session.id", ondelete="CASCADE"), nullable=False)
    line_id: Mapped[int] = mapped_column(Integer, ForeignKey("line.id"), nullable=False)
    camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("camera.id"), nullable=False)
    stream_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    crossing_seq: Mapped[int] = mapped_column(Integer, nullable=False)  # Track-specific monotonic crossing seq (§5.5)
    gate_id: Mapped[int] = mapped_column(Integer, ForeignKey("gate.id"), nullable=False)
    crossing_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    direction: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # +1 forward, -1 backward
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    merge_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    # Set when this event is a defect exclusion (a bag already counted at the
    # gate is later pulled -- e.g. found torn while loading) rather than a
    # normal crossing or a plain miscount rollback. Distinguishes "we removed
    # a damaged bag" from "we corrected a double-count" in the audit trail.
    defect_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # A defect flag can be reviewed and found wrong (the AI misread an intact
    # bag). The ledger is append-only, so a dispute is recorded as a real
    # annotation alongside the original flag rather than by erasing or
    # overwriting defect_reason -- the original detection stays in the audit
    # trail, with who/when/why it was overturned recorded next to it.
    defect_disputed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    defect_disputed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    defect_disputed_note: Mapped[str | None] = mapped_column(String(255), nullable=True)
    defect_disputed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # True only for events generated by the demo/simulated conveyor path
    # (LiveStreamRenderer's synthetic self.bags animation, or the
    # /sessions/{id}/simulate_bag operator control), never for a genuine
    # gate crossing produced by the real CountingEngine pipeline. Lets
    # downstream consumers (ledger views, reconciliation, exports) tell real
    # counted bags apart from demo/test traffic that shares the same ledger.
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deployment_bundle_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("deployment_bundle.id"), nullable=False)
    evidence_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionORM", back_populates="events")

    __table_args__ = (
        UniqueConstraint("session_id", "camera_id", "stream_epoch", "track_id", "gate_id", "crossing_seq", name="uq_count_event_idempotency"),
    )


class ReconciliationORM(Base):
    """Reconciliation record for human review and auditing (§5.7)."""
    __tablename__ = "reconciliation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("session.id", ondelete="CASCADE"), nullable=False)
    trigger_reason: Mapped[str] = mapped_column(String(64), nullable=False)  # degraded_session | count_area_mismatch | erp_conflict | operator_request
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    assigned_role: Mapped[str] = mapped_column(String(32), default="engineer")
    evidence_refs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    resolution: Mapped[str | None] = mapped_column(String(32), nullable=True)  # accept_system | manual_override | void_session
    resolved_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class JobORM(Base):
    """Background job with lease and heartbeat (§5.8)."""
    __tablename__ = "job"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Indexed: acquire_next_job()/list_jobs() filter on status on every poll of the queue.
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)  # queued | running | completed | failed | cancelled
    priority: Mapped[int] = mapped_column(Integer, default=0)
    requires_gpu: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class OutboxORM(Base):
    """Transactional Outbox for ERP dispatch (§5.8, M7)."""
    __tablename__ = "outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("session.id", ondelete="CASCADE"), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending | in_progress | sent | failed | reconcile_required
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    external_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionORM", back_populates="outbox_entries")

    __table_args__ = (
        # fetch_pending_entries()/claim step filters on status + next_attempt_at on every
        # dispatcher poll; composite index matches that WHERE clause directly.
        Index("ix_outbox_status_next_attempt_at", "status", "next_attempt_at"),
    )


class UserAccountORM(Base):
    """User account with RBAC role (§8.1)."""
    __tablename__ = "user_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # operator | engineer | admin
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
