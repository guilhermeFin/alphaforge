"""Typed records for the accounts data layer — plain dataclasses the store maps
to/from SQLite rows. ``Role`` carries an ORDERING so authorization can ask for a
MINIMUM role (deny-by-default) instead of enumerating every allowed role."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    VIEWER = "viewer"   # read-only
    MEMBER = "member"   # run backtests, create runs
    ADMIN = "admin"     # manage API keys / members
    OWNER = "owner"     # billing, export, delete the workspace


_ROLE_RANK = {Role.VIEWER: 0, Role.MEMBER: 1, Role.ADMIN: 2, Role.OWNER: 3}


def role_rank(role: Role) -> int:
    return _ROLE_RANK[role]


def role_at_least(role: Role, minimum: Role) -> bool:
    """True iff ``role`` is at least as privileged as ``minimum`` (the deny-by-default
    primitive: a higher role always satisfies a lower minimum, never the reverse)."""
    return role_rank(role) >= role_rank(minimum)


@dataclass
class User:
    id: str
    idp_subject: str      # the external identity-provider 'sub' (we never store passwords)
    email: str            # plaintext in memory; encrypted + blind-indexed at rest
    created_at: str
    last_seen_at: str


@dataclass
class Workspace:
    id: str
    name: str
    owner_user_id: str
    plan: str
    stripe_customer_id: str | None  # a Stripe REFERENCE only — never card data
    created_at: str


@dataclass
class Membership:
    workspace_id: str
    user_id: str
    role: Role
    created_at: str


@dataclass
class ApiKey:
    id: str
    workspace_id: str
    created_by: str
    name: str
    prefix: str               # first chars of the secret, for display (non-secret)
    created_at: str
    last_used_at: str | None
    revoked_at: str | None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


@dataclass
class Run:
    id: str
    workspace_id: str
    user_id: str
    kind: str                 # "backtest" | "ml_compare"
    request: dict             # decrypted in memory; encrypted at rest (strategy IP)
    verdict: str | None
    summary: dict             # decrypted in memory; encrypted at rest
    created_at: str


@dataclass
class AuditEvent:
    id: int
    ts: str
    workspace_id: str | None
    actor_user_id: str | None
    action: str
    target: str | None
    request_id: str | None
    ip: str | None
    metadata: dict
    prev_hash: str
    hash: str
