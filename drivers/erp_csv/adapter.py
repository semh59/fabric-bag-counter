"""CSV ERP file-drop adapter (§4.4, §11 M7)."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from packages.cs_core.interfaces.erp_adapter import (
    ErpAdapter,
    ErpResult,
    ErpStatus,
    ErpStatusState,
    SessionPayload,
)


class CsvErpAdapter:
    """Unidirectional CSV export adapter."""

    def __init__(self, export_dir: str = "./data/erp_exports") -> None:
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def submit_session(self, payload: SessionPayload) -> ErpResult:
        """Export session counts into a CSV file."""
        file_name = f"dispatch_session_{payload.session_id}_{payload.external_ref or 'no_ref'}.csv"
        file_path = os.path.join(self.export_dir, file_name)

        try:
            with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "session_id",
                    "line_id",
                    "external_ref",
                    "material_code",
                    "counted_total",
                    "area_estimate_total",
                    "opened_at",
                    "closed_at",
                ])
                writer.writerow([
                    payload.session_id,
                    payload.line_id,
                    payload.external_ref or "",
                    payload.erp_material_code or "",
                    payload.counted_total,
                    f"{payload.area_estimate_total:.2f}",
                    payload.opened_at.isoformat(),
                    payload.closed_at.isoformat() if payload.closed_at else "",
                ])

            return ErpResult(success=True, external_tx_id=file_name)
        except Exception as e:
            return ErpResult(success=False, external_tx_id=None, error_message=str(e), retryable=True)

    def query_status(self, external_ref: str) -> ErpStatus:
        """CSV file-drop adapter does not support remote status querying."""
        return ErpStatus(
            state=ErpStatusState.UNKNOWN,
            external_tx_id=None,
            message="CSV adapter is unidirectional and does not support remote query.",
        )

    @property
    def supports_status_query(self) -> bool:
        return False
