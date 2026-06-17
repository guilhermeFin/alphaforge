"""AlphaForge accounts & data layer — built secure-by-default *before* the first
customer record exists (the cheapest time to get it right; see SECURITY.md).

What lives here:
  * crypto    — field encryption, blind indexing, API-key hashing (one master key)
  * db        — SQLite schema + a Postgres-ready, parameterized connection
  * models    — typed records + a Role with an ordering for min-role checks
  * store     — tenant-scoped repository (deny-by-default isolation)
  * authz     — per-workspace authorization (BOLA/BFLA defenses)
  * identity  — Principal + API-key auth today, an OIDC seam for Clerk/Auth0 later
  * audit     — tamper-evident, append-only, hash-chained audit log
  * service   — the audited high-level operations the API calls

Nothing here invents its own password storage or touches card data: identity is
delegated to a managed provider (OIDC seam) and payments to Stripe (we store only a
customer id). The two highest-value targets never live on our servers.
"""
from __future__ import annotations

from .crypto import Crypto, CryptoError
from .models import ApiKey, AuditEvent, Membership, Role, Run, User, Workspace
from .store import Store
from .authz import AuthzError, require_workspace
from .identity import Principal, resolve_principal
from .audit import AuditLog
from .service import AccountService

__all__ = [
    "Crypto", "CryptoError", "Store", "AuthzError", "require_workspace",
    "Principal", "resolve_principal", "AuditLog", "AccountService",
    "ApiKey", "AuditEvent", "Membership", "Role", "Run", "User", "Workspace",
]
