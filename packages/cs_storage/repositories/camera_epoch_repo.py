"""Repository for persistent camera stream epoch management (§5.2)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import CameraEpochORM


class CameraEpochRepository:
    """Manages persistent monotonically increasing stream epoch counter for cameras."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def increment_and_get_epoch(self, camera_id: int) -> int:
        """Atomically increment and return the new stream epoch for the given camera.
        
        If no epoch record exists yet, it creates one starting at epoch 1.
        """
        record = self.db.execute(
            select(CameraEpochORM).where(CameraEpochORM.camera_id == camera_id).with_for_update()
        ).scalar_one_or_none()

        if record is None:
            record = CameraEpochORM(camera_id=camera_id, current_epoch=1)
            self.db.add(record)
        else:
            record.current_epoch += 1

        self.db.commit()
        return int(record.current_epoch)

    def get_current_epoch(self, camera_id: int) -> int:
        """Read the current stream epoch without incrementing."""
        record = self.db.execute(
            select(CameraEpochORM).where(CameraEpochORM.camera_id == camera_id)
        ).scalar_one_or_none()
        return int(record.current_epoch) if record else 0
