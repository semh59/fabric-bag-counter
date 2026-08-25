"""ErpAdapter protocol and payload structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ErpStatusState(str, Enum):
    PENDING = "pending"
    POSTED = "posted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


from dataclasses import dataclass, field


@dataclass
class SessionPayload:
    session_id: int
    line_id: int
    product_profile_id: int
    counted_total: int
    area_estimate_total: float = 0.0
    external_ref: str | None = None
    erp_material_code: str | None = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErpResult:
    success: bool
    external_tx_id: str | None
    error_message: str | None = None
    retryable: bool = False


@dataclass
class ErpStatus:
    state: ErpStatusState
    external_tx_id: str | None
    message: str | None = None


@runtime_checkable
class ErpAdapter(Protocol):
    """Protocol for integrating with external ERP systems (SAP, Oracle, CSV, etc.)."""

    def submit_session(self, payload: SessionPayload) -> ErpResult:
        """Submit finalized count session to ERP."""
        ...

    def query_status(self, external_ref: str) -> ErpStatus:
        """Query delivery/posting status from ERP for idempotent recovery."""
        ...

    @property
    def supports_status_query(self) -> bool:
        """Return True if adapter supports asynchronous status polling (e.g. SAP OData).
        
        Returns False for unidirectional adapters (e.g. CSV file drop).
        """
        ...
