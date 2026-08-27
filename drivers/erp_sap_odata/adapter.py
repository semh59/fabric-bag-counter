"""SAP OData ERP integration adapter (§4.4, §11 M7)."""

from __future__ import annotations

import httpx
from packages.cs_core.interfaces.erp_adapter import (
    ErpResult,
    ErpStatus,
    ErpStatusState,
    SessionPayload,
)

# Unreachable placeholder base URL. This is intentionally never a valid SAP
# endpoint -- every real deployment must supply its own `odata_base_url`.
# submit_session/query_status refuse to make a network call while this
# placeholder is still configured (see _check_base_url_configured) instead
# of failing with an opaque connection error deep inside httpx.
_PLACEHOLDER_BASE_URL = "https://sap.local:8000/sap/opu/odata/sap/Z_BAG_COUNT_SRV"

# Assumed SAP OData entity/response field names. These are specific to one
# SAP service schema (Z_BAG_COUNT_SRV in the reference deployment) -- every
# SAP install can name/rewire its custom OData service differently, so this
# is a configurable default rather than a hardcoded assumption baked into
# the request/response handling code.
DEFAULT_FIELD_MAP: dict[str, str] = {
    # Request body fields (SessionPayload -> SAP entity field name)
    "session_id": "SessionId",
    "line_id": "LineId",
    "delivery_document": "DeliveryDocument",
    "material_number": "MaterialNumber",
    "actual_count": "ActualCount",
    "area_estimate": "AreaEstimate",
    "posting_date": "PostingDate",
    # Response parsing fields (OData envelope conventions)
    "response_envelope": "d",
    "response_material_document_number": "MaterialDocumentNumber",
    "response_status": "Status",
}


class SapConfigurationError(RuntimeError):
    """Raised when the SAP OData adapter is invoked with unresolved placeholder configuration."""


class SapODataErpAdapter:
    """Bi-directional SAP OData service adapter supporting status queries."""

    def __init__(
        self,
        odata_base_url: str = _PLACEHOLDER_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        field_map: dict[str, str] | None = None,
    ) -> None:
        self.base_url = odata_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout_seconds
        # Merge caller overrides onto the documented defaults so a site can
        # override just the field names its SAP service actually uses.
        self.field_map: dict[str, str] = {**DEFAULT_FIELD_MAP, **(field_map or {})}

    def _check_base_url_configured(self) -> None:
        if self.base_url == _PLACEHOLDER_BASE_URL.rstrip("/"):
            raise SapConfigurationError(
                "SapODataErpAdapter.base_url is still the unreachable placeholder "
                f"({_PLACEHOLDER_BASE_URL!r}). Pass a real SAP OData service base URL "
                "via odata_base_url before submitting/querying live ERP data."
            )

    def submit_session(self, payload: SessionPayload) -> ErpResult:
        """Post finalized count event to SAP OData Goods Receipt / Dispatch EntitySet."""
        fm = self.field_map
        url = f"{self.base_url}/BagCountPostings"
        body = {
            fm["session_id"]: str(payload.session_id),
            fm["line_id"]: str(payload.line_id),
            fm["delivery_document"]: payload.external_ref or "",
            fm["material_number"]: payload.erp_material_code or "",
            fm["actual_count"]: payload.counted_total,
            fm["area_estimate"]: float(payload.area_estimate_total),
            fm["posting_date"]: payload.closed_at.isoformat() if payload.closed_at else payload.opened_at.isoformat(),
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # Raised (and caught below into a retryable ErpResult, same as any
            # other submission failure) rather than allowed to fall through to
            # an opaque httpx connection error against an address that was
            # never going to resolve.
            self._check_base_url_configured()
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=body, headers=headers)
                if res.status_code in [200, 201]:
                    data = res.json()
                    envelope = data.get(fm["response_envelope"], {})
                    sap_doc = envelope.get(fm["response_material_document_number"], f"SAP_DOC_{payload.session_id}")
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
        fm = self.field_map
        url = f"{self.base_url}/BagCountPostings('{external_ref}')"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            self._check_base_url_configured()
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(url, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    envelope = data.get(fm["response_envelope"], {})
                    status_val = envelope.get(fm["response_status"], "posted")
                    doc_num = envelope.get(fm["response_material_document_number"])
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
