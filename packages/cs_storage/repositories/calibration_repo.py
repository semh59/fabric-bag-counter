"""Repository for Line Calibrations (Stage 1 motion & Stage 2 scale) (§5.3)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import LineCalibrationORM


class CalibrationRepository:
    """Manages two-stage line calibrations and active calibration lookup."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_motion_calibration(
        self,
        line_id: int,
        belt_speed_px_per_frame: float,
        belt_direction_vector: list[float],
        created_by: str | None = None,
        is_active: bool = True,
    ) -> LineCalibrationORM:
        """Create Stage 1 motion calibration record."""
        if is_active:
            # Deactivate previous active motion calibrations
            self._deactivate_stage(line_id, "motion")

        calib = LineCalibrationORM(
            line_id=line_id,
            stage="motion",
            belt_speed_px_per_frame=belt_speed_px_per_frame,
            belt_direction_vector=belt_direction_vector,
            is_active=is_active,
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(calib)
        self.db.commit()
        self.db.refresh(calib)
        return calib

    def create_scale_calibration(
        self,
        line_id: int,
        px_per_mm: float,
        mean_bag_gate_area_px: float,
        bag_area_stddev_px: float,
        source_video_ref: str | None = None,
        source_model_version_id: int | None = None,
        created_by: str | None = None,
        is_active: bool = True,
    ) -> LineCalibrationORM:
        """Create Stage 2 scale calibration record."""
        if is_active:
            self._deactivate_stage(line_id, "scale")

        calib = LineCalibrationORM(
            line_id=line_id,
            stage="scale",
            px_per_mm=px_per_mm,
            mean_bag_gate_area_px=mean_bag_gate_area_px,
            bag_area_stddev_px=bag_area_stddev_px,
            source_video_ref=source_video_ref,
            source_model_version_id=source_model_version_id,
            is_active=is_active,
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(calib)
        self.db.commit()
        self.db.refresh(calib)
        return calib

    def get_active_calibration(self, line_id: int, stage: str = "scale") -> LineCalibrationORM | None:
        """Fetch the current active calibration for a specific stage."""
        stmt = (
            select(LineCalibrationORM)
            .where(
                LineCalibrationORM.line_id == line_id,
                LineCalibrationORM.stage == stage,
                LineCalibrationORM.is_active == True,  # noqa: E712
            )
            .order_by(LineCalibrationORM.created_at.desc())
        )
        return self.db.execute(stmt).scalars().first()

    def list_calibrations(self, line_id: int) -> Sequence[LineCalibrationORM]:
        """List all historical calibrations for a line."""
        stmt = select(LineCalibrationORM).where(LineCalibrationORM.line_id == line_id).order_by(LineCalibrationORM.created_at.desc())
        return self.db.execute(stmt).scalars().all()

    def _deactivate_stage(self, line_id: int, stage: str) -> None:
        stmt = select(LineCalibrationORM).where(
            LineCalibrationORM.line_id == line_id,
            LineCalibrationORM.stage == stage,
            LineCalibrationORM.is_active == True,  # noqa: E712
        )
        for cal in self.db.execute(stmt).scalars().all():
            cal.is_active = False
