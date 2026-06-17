"""Who is calling — request-time authentication.

The LIVE mechanism today is the API key: a ``Authorization: Bearer af_live_…`` header
is resolved against the store (constant-time hash check), yielding a ``Principal``
bound to the key's workspace at the creator's role. This needs no external service, so
authentication actually works in the MVP.

Managed-IdP single sign-on (Clerk/Auth0/Cognito) is the production path for human
logins; it plugs in at ``AccountService.login_with_identity`` (which takes already
VERIFIED claims). We deliberately do NOT roll our own password storage anywhere — the
only credential this module ever sees is a high-entropy bearer secret.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .crypto import API_KEY_PREFIX
from .models import Role

if TYPE_CHECKING:
    from .store import Store


@dataclass
class Principal:
    """The authenticated caller, scoped to ONE workspace for this request."""
    user_id: str
    email: str
    workspace_id: str
    role: Role
    auth_method: str            # "api_key" | "session"
    api_key_id: str | None = None


def _bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def resolve_principal(store: "Store", authorization_header: str | None) -> "Principal | None":
    """Resolve an API-key bearer header to a Principal, or None for anonymous.

    Deny-by-default: an unknown/revoked key, or a key whose workspace membership has
    since been removed, authenticates to NOTHING (returns None), never a partial
    identity.
    """
    token = _bearer_token(authorization_header)
    if not token or not token.startswith(API_KEY_PREFIX):
        return None
    key = store.resolve_api_key(token)
    if key is None:
        return None
    member = store.get_membership(key.workspace_id, key.created_by)
    if member is None:
        return None
    user = store.get_user(key.created_by)
    if user is None:
        return None
    return Principal(user.id, user.email, key.workspace_id, member.role,
                     "api_key", api_key_id=key.id)
