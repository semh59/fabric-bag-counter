"""Operator manual session identity driver (§4.4)."""

from __future__ import annotations

from packages.cs_core.interfaces.session_identity import SessionIdentity, SessionRef


class OperatorSessionIdentity:
    """Acquires session references created directly by shift operators on the UI."""

    def __init__(self) -> None:
        self.pending_refs: dict[int, SessionRef] = {}

    def set_pending(self, line_id: int, ref: SessionRef) -> None:
        self.pending_refs[line_id] = ref

    def acquire(self, line_id: int) -> SessionRef | None:
        return self.pending_refs.pop(line_id, None)
