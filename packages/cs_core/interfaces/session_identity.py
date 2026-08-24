"""SessionIdentity protocol for acquiring shipping session metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class SessionRef:
    external_ref: str
    target_count: int | None
    product_profile_id: int | None
    metadata: dict[str, Any]


@runtime_checkable
class SessionIdentity(Protocol):
    """Protocol for discovering or triggering session creation (via barcode scan, UI, ERP push)."""

    def acquire(self, line_id: int) -> SessionRef | None:
        """Attempt to acquire active shipping order reference for the specified line."""
        ...
