"""Shared domain exceptions for the storage layer.

Repositories raise these instead of letting callers guess at intent from a
bare `None` return or a raw SQLAlchemy `IntegrityError`. API routes / job
workers can catch a specific, documented exception type here and decide how
to respond (e.g. translate to an HTTP 409) instead of pattern-matching on
storage internals.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for domain-level errors raised by packages.cs_storage repositories."""


class ActiveSessionConflictError(StorageError):
    """Raised when opening a session on a line that already has an active one.

    A line should have at most one non-terminal (open/counting/paused/degraded)
    counting session at a time -- the ledger and area-estimate reconciliation
    logic assume a single active session per line. Callers (e.g. the
    `/sessions` API route) should catch this and surface it as HTTP 409.
    """

    def __init__(self, line_id: int, existing_session_id: int) -> None:
        self.line_id = line_id
        self.existing_session_id = existing_session_id
        super().__init__(
            f"Line {line_id} already has an active session (id={existing_session_id}); "
            "close or pause it before opening a new one."
        )
