"""Standard SAP S/4HANA OData ERP Integration Adapter (§4.4, §11 M7).

Supports official standard SAP S/4HANA OData services:
1. `API_MATERIAL_DOCUMENT_SRV`: Goods Movement & Material Postings (MIGO 601 Goods Issue / 101 Goods Receipt).
2. `API_OUTBOUND_DELIVERY_SRV`: Outbound Delivery Processing (VL02N Picking & Goods Issue Confirmation).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import httpx

from packages.cs_core.interfaces.erp_adapter import (
    ErpResult,
    ErpStatus,
    ErpStatusState,
    SessionPayload,
)


class SapServiceType(str, Enum):
    """Standard SAP S/4HANA OData service endpoint types."""

    MATERIAL_DOCUMENT = "API_MATERIAL_DOCUMENT_SRV"
    OUTBOUND_DELIVERY = "API_OUTBOUND_DELIVERY_SRV"


# Standard default base URLs for SAP S/4HANA OData services
_DEFAULT_SAP_HOST = "https://sap.local:8000/sap/opu/odata/sap"
_PLACEHOLDER_BASE_URL = f"{_DEFAULT_SAP_HOST}/API_MATERIAL_DOCUMENT_SRV"


class SapConfigurationError(RuntimeError):
    """Raised when the SAP OData adapter is invoked with unresolved placeholder configuration."""


class SapODataErpAdapter:
    """Bi-directional standard SAP S/4HANA OData service adapter."""

    def __init__(
        self,
        odata_base_url: str = _PLACEHOLDER_BASE_URL,
        service_type: SapServiceType | str = SapServiceType.MATERIAL_DOCUMENT,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        sap_client: str = "100",
        plant: str = "1000",
        storage_location: str = "1010",
        goods_movement_type: str = "601",  # 601: Goods Issue for Delivery
        timeout_seconds: float = 10.0,
        field_map: dict[str, str] | None = None,
    ) -> None:
        self.base_url = odata_base_url.rstrip("/")
        self.service_type = SapServiceType(service_type) if isinstance(service_type, str) else service_type
        self.api_key = api_key
        self.username = username
        self.password = password
        self.sap_client = sap_client
        self.plant = plant
        self.storage_location = storage_location
        self.goods_movement_type = goods_movement_type
        self.timeout = timeout_seconds

        # Configurable field mapping with standard SAP S/4HANA defaults
        self.field_map: dict[str, str] = {
            "posting_date": "PostingDate",
            "document_date": "DocumentDate",
            "header_text": "MaterialDocumentHeaderText",
            "material": "Material",
            "plant": "Plant",
            "storage_loc": "StorageLocation",
            "movement_type": "GoodsMovementType",
            "quantity": "QuantityInEntryUnit",
            "unit": "EntryUnit",
            "delivery_doc": "DeliveryDocument",
            "response_envelope": "d",
            "mat_doc_number": "MaterialDocument",
            "mat_doc_year": "MaterialDocumentYear",
            **(field_map or {}),
        }

    def _check_base_url_configured(self) -> None:
        if self.base_url == _PLACEHOLDER_BASE_URL.rstrip("/"):
            raise SapConfigurationError(
                f"SapODataErpAdapter.base_url is configured with placeholder ({_PLACEHOLDER_BASE_URL!r}). "
                "Configure a valid production SAP S/4HANA OData endpoint URL before issuing ERP transactions."
            )

    def _build_headers(self, csrf_token: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "sap-client": self.sap_client,
        }
        if csrf_token:
            headers["x-csrf-token"] = csrf_token
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _fetch_csrf_token(self, client: httpx.Client) -> tuple[str | None, dict[str, str]]:
        """Fetch SAP CSRF token (X-CSRF-Token: Fetch) required for state-modifying POST requests."""
        headers = {"x-csrf-token": "Fetch", "Accept": "application/json", "sap-client": self.sap_client}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        auth = (self.username, self.password) if (self.username and self.password) else None

        try:
            res = client.get(f"{self.base_url}/$metadata", headers=headers, auth=auth)
            token = res.headers.get("x-csrf-token")
            cookies = dict(res.cookies)
            return token, cookies
        except Exception:
            return None, {}

    def _build_material_document_payload(self, payload: SessionPayload) -> dict[str, Any]:
        """Build standard SAP S/4HANA `API_MATERIAL_DOCUMENT_SRV` JSON payload."""
        fm = self.field_map
        posting_dt = (payload.closed_at or payload.opened_at or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%S")

        return {
            fm["posting_date"]: f"/Date({int(datetime.now(timezone.utc).timestamp() * 1000)})/",
            fm["document_date"]: f"/Date({int(datetime.now(timezone.utc).timestamp() * 1000)})/",
            fm["header_text"]: f"Delivery {payload.external_ref or payload.session_id}",
            "to_MaterialDocumentItem": {
                "results": [
                    {
                        fm["material"]: str(payload.erp_material_code or "DEFAULT_BAG_MATERIAL"),
                        fm["plant"]: self.plant,
                        fm["storage_loc"]: self.storage_location,
                        fm["movement_type"]: self.goods_movement_type,
                        fm["quantity"]: str(payload.counted_total),
                        fm["unit"]: "ST",
                        fm["delivery_doc"]: str(payload.external_ref or ""),
                    }
                ]
            },
        }

    def submit_session(self, payload: SessionPayload) -> ErpResult:
        """Post verified bag count to standard SAP S/4HANA OData service."""
        try:
            self._check_base_url_configured()

            auth = (self.username, self.password) if (self.username and self.password) else None
            with httpx.Client(timeout=self.timeout, auth=auth) as client:
                csrf_token, cookies = self._fetch_csrf_token(client)
                headers = self._build_headers(csrf_token=csrf_token)

                if self.service_type == SapServiceType.MATERIAL_DOCUMENT:
                    url = f"{self.base_url}/A_MaterialDocumentHeader"
                    body = self._build_material_document_payload(payload)
                else:
                    url = f"{self.base_url}/A_OutbDeliveryHeader('{payload.external_ref}')/to_DeliveryDocumentItem"
                    body = {
                        "ActualDeliveryQuantity": str(payload.counted_total),
                        "DeliveryQuantityUnit": "ST",
                    }

                res = client.post(url, json=body, headers=headers, cookies=cookies)

                if res.status_code in [200, 201]:
                    data = res.json()
                    envelope = data.get(self.field_map["response_envelope"], {})
                    mat_doc = envelope.get(self.field_map["mat_doc_number"], f"SAP_DOC_{payload.session_id}")
                    mat_year = envelope.get(self.field_map["mat_doc_year"], datetime.now().year)
                    tx_id = f"{mat_doc}/{mat_year}" if mat_doc else f"SAP_TX_{payload.session_id}"
                    return ErpResult(success=True, external_tx_id=str(tx_id))
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
        """Query standard SAP S/4HANA OData service for goods issue / delivery status (§11 M7)."""
        try:
            self._check_base_url_configured()

            auth = (self.username, self.password) if (self.username and self.password) else None
            headers = self._build_headers()

            with httpx.Client(timeout=self.timeout, auth=auth) as client:
                if self.service_type == SapServiceType.MATERIAL_DOCUMENT:
                    url = f"{self.base_url}/A_MaterialDocumentHeader?$filter=MaterialDocumentHeaderText eq '{external_ref}'&$top=1"
                else:
                    url = f"{self.base_url}/A_OutbDeliveryHeader('{external_ref}')"

                res = client.get(url, headers=headers)

                if res.status_code == 200:
                    data = res.json()
                    envelope = data.get(self.field_map["response_envelope"], {})
                    results = envelope.get("results", [envelope]) if isinstance(envelope, dict) else []
                    if results:
                        item = results[0]
                        mat_doc = item.get(self.field_map["mat_doc_number"]) or external_ref
                        return ErpStatus(state=ErpStatusState.POSTED, external_tx_id=str(mat_doc))
                    return ErpStatus(state=ErpStatusState.PENDING, external_tx_id=None, message="Document not yet posted")
                elif res.status_code == 404:
                    return ErpStatus(state=ErpStatusState.UNKNOWN, external_tx_id=None, message="Not found in SAP")
                else:
                    return ErpStatus(state=ErpStatusState.UNKNOWN, external_tx_id=None, message=f"HTTP {res.status_code}: {res.text}")
        except Exception as e:
            return ErpStatus(state=ErpStatusState.UNKNOWN, external_tx_id=None, message=str(e))

    @property
    def supports_status_query(self) -> bool:
        return True
