"""Per-workspace authorization — the BOLA/BFLA defense (OWASP API1/API5).

Every workspace-scoped operation goes through ``require_workspace``, which RE-CHECKS
membership against the store for the *target* workspace and enforces a MINIMUM role.
It deliberately does not trust a role carried on the principal for some other
workspace: you can only ever act on a workspace you are actually a member of, at the
privilege your membership grants. Deny-by-default — anything unproven is refused.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .models import Role, role_at_least

if TYPE_CHECKING:  # avoid an import cycle (identity imports nothing from authz)
    from .identity import Principal
    from .store import Store


class AuthzError(PermissionError):
    """Authorization denied (the API maps this to HTTP 403)."""


def require_workspace(store: "Store", principal: "Principal | None",
                      workspace_id: str, minimum: Role = Role.MEMBER):
    """Authorize ``principal`` for ``workspace_id`` at >= ``minimum`` role, or raise.

    Returns the live ``Membership`` on success so callers can use the real role.
    """
    if principal is None:
        raise AuthzError("authentication required")
    member = store.get_membership(workspace_id, principal.user_id)
    if member is None:
        raise AuthzError("not a member of this workspace")
    if not role_at_least(member.role, minimum):
        raise AuthzError(f"insufficient role (need >= {minimum.value}, have {member.role.value})")
    return member
