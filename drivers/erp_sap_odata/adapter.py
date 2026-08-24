"""SAP OData ERP integration adapter (§4.4, §11 M7)."""

from __future__ import annotations

from typing import Any
import httpx
from packages.cs_core.interfaces.erp_adapter import (
    ErpAdapter,
    ErpResult,
    ErpStatus,
    ErpStatusState,
    SessionPayload,
)


class SapODataErpAdapter:
    """Bi-directional SAP OData service adapter supporting status queries."""

    def __init__(
        self,
        odata_base_url: str = "https://sap.local:8000/sap/opu/odata/sap/Z_BAG_COUNT_SRV",
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = odata_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout_seconds

    def submit_session(self, payload: SessionPayload) -> ErpResult:
        """Post finalized count event to SAP OData Goods Receipt / Dispatch EntitySet."""
        url = f"{self.base_url}/BagCountPostings"
        body = {
            "SessionId": str(payload.session_id),
            "LineId": str(payload.line_id),
            "DeliveryDocument": payload.external_ref or "",
            "MaterialNumber": payload.erp_material_code or "",
            "ActualCount": payload.counted_total,
            "AreaEstimate": float(payload.area_estimate_total),
            "PostingDate": payload.closed_at.isoformat() if payload.closed_at else payload.opened_at.isoformat(),
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=body, headers=headers)
                if res.status_code in [200, 201]:
                    data = res.json()
                    sap_doc = data.get("d", {}).get("MaterialDocumentNumber", f"SAP_DOC_{payload.session_id}")
                    return ErpResult(success=True, external_tx_id=str(sap_doc))
                else:
                    return ErpResult(
                        success=False,
                        external_tx_id=None,
                        error_message=f"SAP Error HTTP {res.status_code}: {res.text}",
                        retryable=(res.status_code in [500, 502, 503, 504]),
                    )
        except Exception as e:
            return ErpResult(success=False, external_tx_id=None, error_message=str(e), retryable=True)

    def query_status(self, external_ref: str) -> ErpStatus:
        """Query SAP to verify if delivery document has already been posted (§11 M7)."""
        url = f"{self.base_url}/BagCountPostings('{external_ref}')"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(url, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    status_val = data.get("d", {}).get("Status", "posted")
                    doc_num = data.get("d", {}).get("MaterialDocumentNumber")
                    return ErpStatus(
                        state=ErpStatusState.POSTED if status_val == "posted" else ErpStatusState.PENDING,
                        external_tx_id=doc_num,
                    )
                elif res.status_code == 404:
                    return ErpStatus(state=ErpStatusState.UNKNOWN, external_tx_id=None, message="Not found in SAP")
                else:
                    return ErpStatus(state=ErpStatusState.UNKNOWN, external_tx_id=None, message=res.text)
        except Exception as e:
            return ErpStatus(state=ErpStatusState.UNKNOWN, external_tx_id=None, message=str(e))

    @property
    def supports_status_query(self) -> bool:
        return True
