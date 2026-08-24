"""Domain data models and enums for the bag counting system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UserRole(str, Enum):
    OPERATOR = "operator"
    ENGINEER = "engineer"
    ADMIN = "admin"


class CameraRole(str, Enum):
    COUNTING = "counting"
    VEHICLE_WATCHDOG = "vehicle_watchdog"
    AUXILIARY = "auxiliary"


class LineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DEGRADED = "degraded"
    ERROR = "error"


class SessionStatus(str, Enum):
    OPEN = "open"
    COUNTING = "counting"
    PAUSED = "paused"
    DEGRADED = "degraded"
    CLOSED = "closed"
    RECONCILE_REQUIRED = "reconcile_required"
    RECONCILED = "reconciled"


class ReconciliationReason(str, Enum):
    DEGRADED_SESSION = "degraded_session"
    COUNT_AREA_MISMATCH = "count_area_mismatch"
    ERP_CONFLICT = "erp_conflict"
    OPERATOR_REQUEST = "operator_request"


class ReconciliationResolution(str, Enum):
    ACCEPT_SYSTEM = "accept_system"
    MANUAL_OVERRIDE = "manual_override"
    VOID_SESSION = "void_session"


class JobKind(str, Enum):
    EXTRACT_FRAMES = "extract_frames"
    SYNTHESIZE = "synthesize"
    BUILD_DATASET = "build_dataset"
    TRAIN = "train"
    EXPORT_ONNX = "export_onnx"
    EVALUATE = "evaluate"
    REPLAY = "replay"
    CALIBRATE_MOTION = "calibrate_motion"
    CALIBRATE_SCALE = "calibrate_scale"
    MINE_HARD_FRAMES = "mine_hard_frames"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutboxStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SENT = "sent"
    FAILED = "failed"
    RECONCILE_REQUIRED = "reconcile_required"


class CalibrationStage(str, Enum):
    MOTION = "motion"
    SCALE = "scale"


class ModelStage(str, Enum):
    DRAFT = "draft"
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"


class TrainingRunKind(str, Enum):
    BASE = "base"
    SITE_ADAPTATION = "site_adaptation"


class TrainingRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GpuSharingMode(str, Enum):
    STRICT = "strict"
    WINDOW = "window"
    ALWAYS = "always"


# ---------------------------------------------------------------------------
# Pydantic Domain Schemas
# ---------------------------------------------------------------------------

class SiteBase(BaseModel):
    name: str
    timezone: str = "Europe/Istanbul"
    locale: str = "tr_TR"


class Site(SiteBase):
    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NodeBase(BaseModel):
    site_id: int
    hostname: str
    gpu_info: dict[str, Any] = Field(default_factory=dict)
    status: str = "online"


class Node(NodeBase):
    id: int
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)


class LineBase(BaseModel):
    site_id: int
    name: str
    status: LineStatus = LineStatus.IDLE
    maintenance_window: dict[str, Any] | None = None


class Line(LineBase):
    id: int


class CameraBase(BaseModel):
    line_id: int
    node_id: int
    source_driver: str = "rtsp"  # e.g., rtsp, file
    source_config: dict[str, Any] = Field(default_factory=dict)
    role: CameraRole = CameraRole.COUNTING
    enabled: bool = True


class Camera(CameraBase):
    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CameraEpoch(BaseModel):
    camera_id: int
    current_epoch: int = 0


class GateBase(BaseModel):
    line_id: int
    name: str
    order_index: int = 0


class Gate(GateBase):
    id: int


class ProductProfileBase(BaseModel):
    site_id: int
    name: str
    nominal_weight_g: float | None = None
    nominal_dims_mm: dict[str, float] = Field(default_factory=dict)  # length, width, thickness
    template_images: list[str] = Field(default_factory=list)
    erp_material_code: str | None = None


class ProductProfile(ProductProfileBase):
    id: int


class LineCalibrationBase(BaseModel):
    line_id: int
    stage: CalibrationStage
    belt_speed_px_per_frame: float | None = None
    belt_direction_vector: list[float] | None = None  # [vx, vy] unit vector
    px_per_mm: float | None = None
    mean_bag_gate_area_px: float | None = None
    bag_area_stddev_px: float | None = None
    source_video_ref: str | None = None
    source_model_version_id: int | None = None
    is_active: bool = False
    created_by: str | None = None


class LineCalibration(LineCalibrationBase):
    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetVersionBase(BaseModel):
    site_id: int
    name: str
    manifest_hash: str
    frame_count: int
    synthetic_count: int = 0
    split_spec: dict[str, Any] = Field(default_factory=dict)
    annotation_guide_version: str = "2.0"


class DatasetVersion(DatasetVersionBase):
    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingRunBase(BaseModel):
    dataset_version_id: int
    base_model_version_id: int | None = None
    run_kind: TrainingRunKind = TrainingRunKind.BASE
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    status: TrainingRunStatus = TrainingRunStatus.QUEUED
    log_ref: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class TrainingRun(TrainingRunBase):
    id: int
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ModelVersionBase(BaseModel):
    training_run_id: int | None = None
    onnx_hash: str
    onnx_path: str
    eval_scores: dict[str, Any] = Field(default_factory=dict)
    stage: ModelStage = ModelStage.DRAFT


class ModelVersion(ModelVersionBase):
    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConfigVersionBase(BaseModel):
    line_id: int
    payload: dict[str, Any]
    payload_schema_version: int = 2
    note: str | None = None
    created_by: str | None = None


class ConfigVersion(ConfigVersionBase):
    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DeploymentBundleBase(BaseModel):
    line_id: int
    model_version_id: int
    config_version_id: int
    calibration_id: int | None = None
    git_commit: str | None = None
    activated_by: str | None = None


class DeploymentBundle(DeploymentBundleBase):
    id: int
    activated_at: datetime = Field(default_factory=datetime.utcnow)
    deactivated_at: datetime | None = None


class CountEventBase(BaseModel):
    session_id: int
    line_id: int
    camera_id: int
    stream_epoch: int
    track_id: int
    crossing_seq: int
    gate_id: int
    crossing_timestamp: datetime
    frame_index: int
    direction: int  # +1 for forward, -1 for backward
    confidence: float | None = None
    merge_flag: bool = False
    deployment_bundle_id: int
    evidence_ref: str | None = None


class CountEvent(CountEventBase):
    event_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SessionBase(BaseModel):
    line_id: int
    product_profile_id: int
    external_ref: str | None = None
    target_count: int | None = None
    status: SessionStatus = SessionStatus.OPEN


class Session(SessionBase):
    id: int
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    locked_at: datetime | None = None
    counted_total: int = 0
    area_estimate_total: float = 0.0
    discrepancy_flag: bool = False
    reconciliation_id: int | None = None


class ReconciliationBase(BaseModel):
    session_id: int
    trigger_reason: ReconciliationReason
    assigned_role: UserRole = UserRole.ENGINEER
    evidence_refs: dict[str, Any] = Field(default_factory=dict)
    resolution: ReconciliationResolution | None = None
    resolved_count: int | None = None
    resolved_by: str | None = None
    note: str | None = None


class Reconciliation(ReconciliationBase):
    id: int
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None


class JobBase(BaseModel):
    kind: JobKind
    payload: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    priority: int = 0
    requires_gpu: bool = False
    attempts: int = 0
    max_attempts: int = 3
    lease_until: datetime | None = None
    heartbeat_at: datetime | None = None
    last_error: str | None = None


class Job(JobBase):
    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class OutboxBase(BaseModel):
    session_id: int
    payload: dict[str, Any] = Field(default_factory=dict)
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    next_attempt_at: datetime = Field(default_factory=datetime.utcnow)
    external_ref: str | None = None
    last_error: str | None = None


class Outbox(OutboxBase):
    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserAccount(BaseModel):
    id: int
    username: str
    role: UserRole
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AuthToken(BaseModel):
    token: str
    user_id: int
    username: str
    role: UserRole
    expires_at: datetime
