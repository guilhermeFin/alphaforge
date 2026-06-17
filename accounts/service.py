"""The audited, authorized high-level operations the API calls.

Every state change here (1) authorizes the caller against the target workspace and
(2) writes an audit event. The API layer should call THESE methods, not the store
directly, so authorization and audit can never be forgotten.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import audit as A
from .audit import AuditLog
from .authz import require_workspace
from .identity import Principal
from .models import Role

if TYPE_CHECKING:
    from .models import ApiKey, Run
    from .store import Store


class AccountService:
    def __init__(self, store: "Store", audit: AuditLog | None = None) -> None:
        self._store = store
        self._audit = audit or AuditLog(store)

    @property
    def store(self) -> "Store":
        return self._store

    @property
    def audit(self) -> AuditLog:
        return self._audit

    # ---------------------------------------------------------------- login
    def login_with_identity(self, idp_subject: str, email: str, *,
                            workspace_name: str | None = None,
                            request_id: str | None = None, ip: str | None = None) -> Principal:
        """Idempotent login from ALREADY-VERIFIED identity claims (subject + email).

        On first login we provision a personal workspace (the user becomes its OWNER).
        The CALLER must have verified the claims — a managed IdP (OIDC) in production,
        or the dev-login endpoint in development. We never see or store a password.
        """
        user = self._store.upsert_user_by_idp(idp_subject, email)
        memberships = self._store.list_memberships_for_user(user.id)
        if not memberships:
            ws = self._store.create_workspace(workspace_name or f"{email}'s workspace", user.id)
            workspace_id, role = ws.id, Role.OWNER
        else:
            owned = [m for m in memberships if m.role == Role.OWNER]
            chosen = owned[0] if owned else memberships[0]
            workspace_id, role = chosen.workspace_id, chosen.role
        self._audit.record(A.LOGIN, workspace_id=workspace_id, actor_user_id=user.id,
                           request_id=request_id, ip=ip, metadata={"method": "identity"})
        return Principal(user.id, user.email, workspace_id, role, "session")

    # ------------------------------------------------------------- api keys
    def create_api_key(self, principal: Principal, name: str, *,
                       request_id: str | None = None, ip: str | None = None):
        require_workspace(self._store, principal, principal.workspace_id, Role.ADMIN)
        key, secret = self._store.create_api_key(principal.workspace_id, principal.user_id, name)
        self._audit.record(A.APIKEY_CREATE, workspace_id=principal.workspace_id,
                           actor_user_id=principal.user_id, target=key.id,
                           request_id=request_id, ip=ip, metadata={"name": name})
        return key, secret

    def list_api_keys(self, principal: Principal) -> "list[ApiKey]":
        require_workspace(self._store, principal, principal.workspace_id, Role.MEMBER)
        return self._store.list_api_keys(principal.workspace_id)

    def revoke_api_key(self, principal: Principal, key_id: str, *,
                       request_id: str | None = None, ip: str | None = None) -> bool:
        require_workspace(self._store, principal, principal.workspace_id, Role.ADMIN)
        ok = self._store.revoke_api_key(principal.workspace_id, key_id)
        if ok:
            self._audit.record(A.APIKEY_REVOKE, workspace_id=principal.workspace_id,
                               actor_user_id=principal.user_id, target=key_id,
                               request_id=request_id, ip=ip)
        return ok

    # ----------------------------------------------------------------- runs
    def record_run(self, principal: Principal, kind: str, request: dict,
                   verdict: str | None, summary: dict, *,
                   request_id: str | None = None, ip: str | None = None) -> "Run":
        require_workspace(self._store, principal, principal.workspace_id, Role.MEMBER)
        run = self._store.record_run(principal.workspace_id, principal.user_id,
                                     kind, request, verdict, summary)
        self._audit.record(A.RUN_CREATE, workspace_id=principal.workspace_id,
                           actor_user_id=principal.user_id, target=run.id,
                           request_id=request_id, ip=ip, metadata={"kind": kind})
        return run

    def list_runs(self, principal: Principal, limit: int = 50) -> "list[Run]":
        require_workspace(self._store, principal, principal.workspace_id, Role.VIEWER)
        return self._store.list_runs(principal.workspace_id, limit=limit)

    # ----------------------------------------------------- GDPR / LGPD rights
    def export_account(self, principal: Principal, *,
                       request_id: str | None = None, ip: str | None = None) -> dict:
        require_workspace(self._store, principal, principal.workspace_id, Role.OWNER)
        data = self._store.export_workspace(principal.workspace_id)
        self._audit.record(A.DATA_EXPORT, workspace_id=principal.workspace_id,
                           actor_user_id=principal.user_id, request_id=request_id, ip=ip)
        return data

    def delete_account(self, principal: Principal, *,
                       request_id: str | None = None, ip: str | None = None) -> dict:
        require_workspace(self._store, principal, principal.workspace_id, Role.OWNER)
        # Record the intent on the workspace chain first; the store then erases the
        # workspace (incl. these events) and writes a system-chain tombstone so proof
        # the erasure happened survives without retaining the erased data.
        self._audit.record(A.DATA_DELETE, workspace_id=principal.workspace_id,
                           actor_user_id=principal.user_id, request_id=request_id, ip=ip)
        return self._store.delete_workspace(principal.workspace_id)
