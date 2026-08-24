"""ERP webhook / polling session identity driver (§4.4)."""

from __future__ import annotations

from typing import Any
from packages.cs_core.interfaces.session_identity import SessionIdentity, SessionRef


class ErpSessionIdentity:
    """Discovers active delivery orders from ERP dispatch queue."""

    def __init__(self, erp_queue_endpoint: str = "http://localhost:8000/api/erp/active_order") -> None:
        self.endpoint = erp_queue_endpoint
        self.cached_orders: dict[int, SessionRef] = {}

    def push_erp_order(self, line_id: int, ref: SessionRef) -> None:
        self.cached_orders[line_id] = ref

    def acquire(self, line_id: int) -> SessionRef | None:
        return self.cached_orders.pop(line_id, None)
