"""Initial schema migration with immutable tables, persistent epoch and ledger constraints

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-08-24 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # site
    op.create_table(
        "site",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("timezone", sa.String(length=64), server_default="Europe/Istanbul"),
        sa.Column("locale", sa.String(length=16), server_default="tr_TR"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # node
    op.create_table(
        "node",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("site_id", sa.Integer(), sa.ForeignKey("site.id", ondelete="CASCADE"), nullable=False),
        sa.Column("hostname", sa.String(length=255), nullable=False),
        sa.Column("gpu_info", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="online"),
        sa.Column("last_heartbeat", sa.DateTime(), server_default=sa.func.now()),
    )

    # line
    op.create_table(
        "line",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("site_id", sa.Integer(), sa.ForeignKey("site.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="idle"),
        sa.Column("maintenance_window", sa.JSON(), nullable=True),
    )

    # camera
    op.create_table(
        "camera",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("line_id", sa.Integer(), sa.ForeignKey("line.id", ondelete="CASCADE"), nullable=False),
        sa.Column("node_id", sa.Integer(), sa.ForeignKey("node.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_driver", sa.String(length=64), server_default="rtsp"),
        sa.Column("source_config", sa.JSON(), nullable=False),
        sa.Column("role", sa.String(length=32), server_default="counting"),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # camera_epoch (§5.2)
    op.create_table(
        "camera_epoch",
        sa.Column("camera_id", sa.Integer(), sa.ForeignKey("camera.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("current_epoch", sa.BigInteger(), server_default="0", nullable=False),
    )

    # gate
    op.create_table(
        "gate",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("line_id", sa.Integer(), sa.ForeignKey("line.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("order_index", sa.Integer(), server_default="0"),
    )

    # product_profile
    op.create_table(
        "product_profile",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("site_id", sa.Integer(), sa.ForeignKey("site.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("nominal_weight_g", sa.Float(), nullable=True),
        sa.Column("nominal_dims_mm", sa.JSON(), nullable=False),
        sa.Column("template_images", sa.JSON(), nullable=False),
        sa.Column("erp_material_code", sa.String(length=128), nullable=True),
    )

    # line_calibration (§5.3)
    op.create_table(
        "line_calibration",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("line_id", sa.Integer(), sa.ForeignKey("line.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("belt_speed_px_per_frame", sa.Float(), nullable=True),
        sa.Column("belt_direction_vector", sa.JSON(), nullable=True),
        sa.Column("px_per_mm", sa.Float(), nullable=True),
        sa.Column("mean_bag_gate_area_px", sa.Float(), nullable=True),
        sa.Column("bag_area_stddev_px", sa.Float(), nullable=True),
        sa.Column("source_video_ref", sa.Text(), nullable=True),
        sa.Column("source_model_version_id", sa.BigInteger(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false")),
    )

    # dataset_version
    op.create_table(
        "dataset_version",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("site_id", sa.Integer(), sa.ForeignKey("site.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=False),
        sa.Column("synthetic_count", sa.Integer(), server_default="0"),
        sa.Column("split_spec", sa.JSON(), nullable=False),
        sa.Column("annotation_guide_version", sa.String(length=32), server_default="2.0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # training_run
    op.create_table(
        "training_run",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("dataset_version_id", sa.Integer(), sa.ForeignKey("dataset_version.id", ondelete="CASCADE"), nullable=False),
        sa.Column("base_model_version_id", sa.Integer(), nullable=True),
        sa.Column("run_kind", sa.String(length=32), server_default="base"),
        sa.Column("hyperparams", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="queued"),
        sa.Column("log_ref", sa.Text(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )

    # model_version
    op.create_table(
        "model_version",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("training_run_id", sa.Integer(), sa.ForeignKey("training_run.id", ondelete="SET NULL"), nullable=True),
        sa.Column("onnx_hash", sa.String(length=64), nullable=False),
        sa.Column("onnx_path", sa.Text(), nullable=False),
        sa.Column("eval_scores", sa.JSON(), nullable=False),
        sa.Column("stage", sa.String(length=32), server_default="draft"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # config_version (§5.4)
    op.create_table(
        "config_version",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("line_id", sa.Integer(), sa.ForeignKey("line.id", ondelete="CASCADE"), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_schema_version", sa.Integer(), server_default="2"),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # deployment_bundle
    op.create_table(
        "deployment_bundle",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("line_id", sa.Integer(), sa.ForeignKey("line.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_version.id"), nullable=False),
        sa.Column("config_version_id", sa.Integer(), sa.ForeignKey("config_version.id"), nullable=False),
        sa.Column("calibration_id", sa.Integer(), sa.ForeignKey("line_calibration.id"), nullable=True),
        sa.Column("git_commit", sa.String(length=64), nullable=True),
        sa.Column("activated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("deactivated_at", sa.DateTime(), nullable=True),
        sa.Column("activated_by", sa.String(length=128), nullable=True),
    )

    # reconciliation (§5.7)
    # NOTE: session_id's FK to session.id is added further below via
    # op.create_foreign_key(), once the "session" table (created after this
    # one) actually exists -- reconciliation and session reference each other
    # (session.reconciliation_id -> reconciliation.id), so one side of this
    # mutual reference must be deferred. This mirrors ReconciliationORM's
    # plain ForeignKey("session.id") in models_orm.py and SessionORM's
    # use_alter=True on its reconciliation_id column.
    op.create_table(
        "reconciliation",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("trigger_reason", sa.String(length=64), nullable=False),
        sa.Column("opened_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("assigned_role", sa.String(length=32), server_default="engineer"),
        sa.Column("evidence_refs", sa.JSON(), nullable=False),
        sa.Column("resolution", sa.String(length=32), nullable=True),
        sa.Column("resolved_count", sa.Integer(), nullable=True),
        sa.Column("resolved_by", sa.String(length=128), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
    )

    # session
    op.create_table(
        "session",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("line_id", sa.Integer(), sa.ForeignKey("line.id", ondelete="CASCADE"), nullable=False),
        sa.Column("product_profile_id", sa.Integer(), sa.ForeignKey("product_profile.id"), nullable=False),
        sa.Column("external_ref", sa.String(length=128), nullable=True),
        sa.Column("target_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="open"),
        sa.Column("opened_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("locked_at", sa.DateTime(), nullable=True),
        sa.Column("counted_total", sa.Integer(), server_default="0"),
        sa.Column("area_estimate_total", sa.Float(), server_default="0.0"),
        sa.Column("discrepancy_flag", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("reconciliation_id", sa.Integer(), sa.ForeignKey("reconciliation.id"), nullable=True),
    )
    op.create_index("ix_session_line_id", "session", ["line_id"])

    # Now that both sides of the mutual session <-> reconciliation reference
    # exist, add the deferred FK from reconciliation.session_id -> session.id
    # (matches ReconciliationORM.session_id in models_orm.py).
    op.create_foreign_key(
        "fk_reconciliation_session",
        "reconciliation",
        "session",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # count_event (§5.5)
    op.create_table(
        "count_event",
        sa.Column("event_id", sa.String(length=36), primary_key=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("session.id", ondelete="CASCADE"), nullable=False),
        sa.Column("line_id", sa.Integer(), sa.ForeignKey("line.id"), nullable=False),
        sa.Column("camera_id", sa.Integer(), sa.ForeignKey("camera.id"), nullable=False),
        sa.Column("stream_epoch", sa.BigInteger(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("crossing_seq", sa.Integer(), nullable=False),
        sa.Column("gate_id", sa.Integer(), sa.ForeignKey("gate.id"), nullable=False),
        sa.Column("crossing_timestamp", sa.DateTime(), nullable=False),
        sa.Column("frame_index", sa.BigInteger(), nullable=False),
        sa.Column("direction", sa.SmallInteger(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("merge_flag", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("defect_reason", sa.String(length=64), nullable=True),
        sa.Column("defect_disputed", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("defect_disputed_by", sa.String(length=128), nullable=True),
        sa.Column("defect_disputed_note", sa.String(length=255), nullable=True),
        sa.Column("defect_disputed_at", sa.DateTime(), nullable=True),
        sa.Column("is_simulated", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("deployment_bundle_id", sa.BigInteger(), sa.ForeignKey("deployment_bundle.id"), nullable=False),
        sa.Column("evidence_ref", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("session_id", "camera_id", "stream_epoch", "track_id", "gate_id", "crossing_seq", name="uq_count_event_idempotency"),
    )

    # job (§5.8)
    op.create_table(
        "job",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="queued"),
        sa.Column("priority", sa.Integer(), server_default="0"),
        sa.Column("requires_gpu", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("attempts", sa.Integer(), server_default="0"),
        sa.Column("max_attempts", sa.Integer(), server_default="3"),
        sa.Column("lease_until", sa.DateTime(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_job_status", "job", ["status"])

    # outbox (§5.8)
    op.create_table(
        "outbox",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("session.id", ondelete="CASCADE"), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending"),
        sa.Column("attempts", sa.Integer(), server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("external_ref", sa.String(length=128), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_outbox_status_next_attempt_at", "outbox", ["status", "next_attempt_at"])

    # user_account
    op.create_table(
        "user_account",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(length=64), unique=True, nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("user_account")
    op.drop_table("outbox")
    op.drop_table("job")
    op.drop_table("count_event")
    # reconciliation.session_id -> session.id was added after both tables existed
    # (mutual reference with session.reconciliation_id -> reconciliation.id), so it
    # must be dropped explicitly before "session" can be dropped.
    op.drop_constraint("fk_reconciliation_session", "reconciliation", type_="foreignkey")
    op.drop_table("session")
    op.drop_table("reconciliation")
    op.drop_table("deployment_bundle")
    op.drop_table("config_version")
    op.drop_table("model_version")
    op.drop_table("training_run")
    op.drop_table("dataset_version")
    op.drop_table("line_calibration")
    op.drop_table("product_profile")
    op.drop_table("gate")
    op.drop_table("camera_epoch")
    op.drop_table("camera")
    op.drop_table("line")
    op.drop_table("node")
    op.drop_table("site")
