"""Audit log — a thin, named-action convenience over the store's hash-chained table.

The honesty discipline mirrors the engine's trial ledger: security-relevant events
(logins, key lifecycle, data access, erasure) are LOGGED to an append-only, per-
workspace hash chain, so the record can't be quietly rewritten. ``verify`` recomputes
the chain to prove it hasn't been tampered with.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AuditEvent
    from .store import Store

# Canonical action names — one place so logs stay greppable and consistent.
LOGIN = "auth.login"
APIKEY_CREATE = "apikey.create"
APIKEY_REVOKE = "apikey.revoke"
RUN_CREATE = "run.create"
DATA_EXPORT = "data.export"
DATA_DELETE = "data.delete"


class AuditLog:
    def __init__(self, store: "Store") -> None:
        self._store = store

    def record(self, action: str, **kwargs) -> "AuditEvent":
        return self._store.append_audit(action, **kwargs)

    def verify(self, workspace_id: str | None) -> bool:
        return self._store.verify_audit_chain(workspace_id)

    def list(self, workspace_id: str | None, limit: int = 200) -> list:
        return self._store.list_audit(workspace_id, limit)
