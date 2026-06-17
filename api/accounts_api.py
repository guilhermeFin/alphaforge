"""Optional accounts/auth surface for the AlphaForge API.

This is wired in addition to the public compute API and is OFF by default — the
existing anonymous backtest flow is untouched unless accounts are explicitly enabled
(``ALPHAFORGE_ACCOUNTS=1`` or an ``ALPHAFORGE_DB_PATH`` is set). When enabled it adds:

  * ``Authorization: Bearer <api key>`` request-time auth (``current_principal``)
  * key-management, run-history, and GDPR/LGPD export + erasure endpoints
  * a dev-only login that mints a working API key (real OIDC plugs in at the same seam)

Security choices: deny-by-default authorization (403 on any failed check), no secret
ever returned except the one-time API-key reveal, and the dev-login endpoint refuses to
run in production (where a verified managed-IdP login is required instead).
"""
from __future__ import annotations

import os

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from accounts.authz import AuthzError
from accounts.identity import resolve_principal
from api.security import is_production

# --- lazy, env-driven service singleton (so import never builds a DB in tests) ----
_SERVICE = None
_INIT_DONE = False


def _accounts_enabled() -> bool:
    if os.environ.get("ALPHAFORGE_ACCOUNTS", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return bool(os.environ.get("ALPHAFORGE_DB_PATH", "").strip())


def _build_from_env():
    if not _accounts_enabled():
        return None
    from accounts import AccountService, Store
    from accounts.crypto import Crypto
    path = os.environ.get("ALPHAFORGE_DB_PATH", "").strip() or "alphaforge.db"
    return AccountService(Store(path, Crypto()))


def get_service():
    """The configured AccountService, or None if accounts are disabled."""
    global _SERVICE, _INIT_DONE
    if not _INIT_DONE:
        _SERVICE = _build_from_env()
        _INIT_DONE = True
    return _SERVICE


def configure_for_test(service) -> None:
    """Inject a service (tests) — bypasses env so a tmp DB can be used."""
    global _SERVICE, _INIT_DONE
    _SERVICE, _INIT_DONE = service, True


def reset() -> None:
    global _SERVICE, _INIT_DONE
    _SERVICE, _INIT_DONE = None, False


def current_principal(request: Request):
    """Resolve the caller (or None for anonymous). Safe to call when disabled."""
    svc = get_service()
    if svc is None:
        return None
    return resolve_principal(svc.store, request.headers.get("authorization"))


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def _require(request: Request):
    svc = get_service()
    if svc is None:
        raise HTTPException(status_code=503, detail="accounts are not enabled on this server")
    p = current_principal(request)
    if p is None:
        raise HTTPException(status_code=401,
                            detail="authentication required — send 'Authorization: Bearer <api key>'")
    return svc, p


# ------------------------------------ models ------------------------------------
class DevLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(..., min_length=3, max_length=254)


class ApiKeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field("api key", min_length=1, max_length=80)


router = APIRouter()


@router.post("/auth/dev-login")
def dev_login(body: DevLoginRequest, request: Request) -> dict:
    """DEV ONLY: log in by email (no password) and receive a working API key.

    Disabled in production, where a managed-IdP (OIDC) login is required — that path
    calls the SAME ``login_with_identity`` seam with verified claims.
    """
    if is_production():
        raise HTTPException(status_code=404, detail="not found")
    svc = get_service()
    if svc is None:
        raise HTTPException(status_code=503, detail="accounts are not enabled on this server")
    rid = request.headers.get("x-request-id")
    p = svc.login_with_identity(f"dev|{body.email.strip().lower()}", body.email.strip(),
                                request_id=rid, ip=_client_ip(request))
    key, secret = svc.create_api_key(p, "dev-login key", request_id=rid, ip=_client_ip(request))
    return {"email": p.email, "workspace_id": p.workspace_id, "role": p.role.value,
            "api_key": secret, "note": "store this key now — it is shown only once"}


@router.get("/account/me")
def account_me(request: Request) -> dict:
    _svc, p = _require(request)
    return {"user_id": p.user_id, "email": p.email, "workspace_id": p.workspace_id,
            "role": p.role.value, "auth_method": p.auth_method}


@router.post("/api-keys")
def create_api_key(body: ApiKeyCreateRequest, request: Request) -> dict:
    svc, p = _require(request)
    key, secret = svc.create_api_key(p, body.name, request_id=request.headers.get("x-request-id"),
                                     ip=_client_ip(request))
    return {"id": key.id, "name": key.name, "prefix": key.prefix, "created_at": key.created_at,
            "api_key": secret, "note": "store this key now — it is shown only once"}


@router.get("/api-keys")
def list_api_keys(request: Request) -> dict:
    svc, p = _require(request)
    keys = svc.list_api_keys(p)
    return {"api_keys": [{"id": k.id, "name": k.name, "prefix": k.prefix,
                          "created_at": k.created_at, "last_used_at": k.last_used_at,
                          "revoked_at": k.revoked_at, "active": k.active} for k in keys]}


@router.delete("/api-keys/{key_id}")
def revoke_api_key(key_id: str, request: Request) -> dict:
    svc, p = _require(request)
    ok = svc.revoke_api_key(p, key_id, request_id=request.headers.get("x-request-id"),
                            ip=_client_ip(request))
    if not ok:
        raise HTTPException(status_code=404, detail="no such active key in this workspace")
    return {"revoked": key_id}


@router.get("/runs")
def list_runs(request: Request, limit: int = 50) -> dict:
    svc, p = _require(request)
    runs = svc.list_runs(p, limit=max(1, min(int(limit), 500)))
    return {"runs": [{"id": r.id, "kind": r.kind, "verdict": r.verdict,
                      "created_at": r.created_at, "request": r.request,
                      "summary": r.summary} for r in runs]}


@router.get("/account/export")
def export_account(request: Request) -> dict:
    svc, p = _require(request)
    return svc.export_account(p, request_id=request.headers.get("x-request-id"),
                              ip=_client_ip(request))


@router.delete("/account")
def delete_account(request: Request) -> dict:
    svc, p = _require(request)
    return svc.delete_account(p, request_id=request.headers.get("x-request-id"),
                              ip=_client_ip(request))


def install_accounts(app: FastAPI) -> None:
    """Attach the accounts router + a 403 mapping for authorization failures."""
    @app.exception_handler(AuthzError)
    async def _authz_handler(_request: Request, exc: AuthzError):  # noqa: ANN001
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    app.include_router(router)
