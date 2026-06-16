"""HTTP security hardening for the AlphaForge API — defense in depth.

These are the controls that belong INSIDE the app: strict response headers, a
locked-down CORS policy, a request-size cap, an optional rate limiter, a per-request
id, and structured access logging. Edge controls (TLS termination, WAF, CDN/gateway
rate limiting, DDoS protection) belong at the infrastructure layer and are documented
in SECURITY.md.

Nothing here processes customer PII — none is stored yet. The data-protection plan
for when accounts/billing land is in SECURITY.md; this module hardens the public
compute surface that exists today.

All behaviour is env-driven so the same code is safe in dev and strict in prod:
  ALPHAFORGE_ENV=production         -> Secure cookies, HSTS, rate limiting on by default
  ALPHAFORGE_CORS_ORIGINS=a,b,c     -> allow these browser origins (default: none)
  ALPHAFORGE_RATE_LIMIT=N           -> N requests / 60s / IP (default: prod=120, dev=off)
  ALPHAFORGE_MAX_BODY_BYTES=N       -> reject bodies larger than N (default 256 KiB)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

DEFAULT_MAX_BODY_BYTES = 256 * 1024  # 256 KiB — backtest requests are tiny JSON
DEFAULT_PROD_RATE_LIMIT = 120        # req/60s/IP, applied in prod when no explicit value
RATE_WINDOW = 60.0

_access_log = logging.getLogger("alphaforge.access")

# A JSON API renders no HTML, so lock the browser surface down hard: no scripts,
# no framing, no referrer leakage, no ambient device access, no caching of results.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
}


def is_production() -> bool:
    return os.environ.get("ALPHAFORGE_ENV", "").strip().lower() in ("prod", "production")


def secure_cookies() -> bool:
    """Set the Secure flag on session cookies in production (HTTPS-only transport)."""
    return is_production()


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ALPHAFORGE_CORS_ORIGINS", "").strip()
    return [o.strip() for o in raw.split(",") if o.strip()]  # [] => no cross-origin browser access


def _max_body_bytes() -> int:
    try:
        return int(os.environ.get("ALPHAFORGE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES))
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES


def _rate_limit() -> int:
    """Requests/60s/IP. Explicit env wins; otherwise on (120) in prod, off in dev/tests."""
    raw = os.environ.get("ALPHAFORGE_RATE_LIMIT")
    if raw is None or raw == "":
        return DEFAULT_PROD_RATE_LIMIT if is_production() else 0
    try:
        return int(raw)
    except ValueError:
        return 0


class _FixedWindowLimiter:
    """Minimal in-process per-key fixed-window limiter. A BACKSTOP, not the primary
    control — a single process behind a load balancer can't rate-limit globally.
    Production rate limiting belongs at the edge (gateway/CDN). See SECURITY.md."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = RATE_WINDOW) -> bool:
        now = time.time()
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and dq[0] <= now - window:
                dq.popleft()
            if len(dq) >= limit:
                return False
            dq.append(now)
            return True


_limiter = _FixedWindowLimiter()


def _apply_headers(resp, request_id: str, production: bool):
    for k, v in _SECURITY_HEADERS.items():
        resp.headers.setdefault(k, v)
    if production:
        resp.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
    resp.headers["X-Request-ID"] = request_id
    return resp


def _log_request(request: Request, status: int, started: float, rid: str) -> None:
    # one structured line per request — method/path/status/latency/id/ip only.
    # NEVER the body, query secrets, cookies, or auth headers.
    _access_log.info(json.dumps({
        "rid": rid, "method": request.method, "path": request.url.path,
        "status": status, "dur_ms": round((time.time() - started) * 1000, 1),
        "ip": request.client.host if request.client else None,
    }))


def install_security(app: FastAPI) -> None:
    """Attach CORS + the security middleware to a FastAPI app."""
    origins = _allowed_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_credentials=True,
            allow_methods=["GET", "POST"], allow_headers=["*"], max_age=600,
        )

    @app.middleware("http")
    async def _harden(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.time()
        prod = is_production()

        # 1) request-size cap (cheap Content-Length guard; blunts payload-DoS)
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > _max_body_bytes():
                    resp = _apply_headers(
                        JSONResponse(status_code=413, content={"detail": "request body too large"}),
                        rid, prod)
                    _log_request(request, 413, started, rid)
                    return resp
            except ValueError:
                resp = _apply_headers(
                    JSONResponse(status_code=400, content={"detail": "invalid Content-Length"}),
                    rid, prod)
                _log_request(request, 400, started, rid)
                return resp

        # 2) optional in-process rate limit (prod default on; edge limiter is primary)
        limit = _rate_limit()
        if limit > 0:
            ip = request.client.host if request.client else "unknown"
            if not _limiter.allow(ip, limit):
                resp = _apply_headers(
                    JSONResponse(status_code=429, content={"detail": "rate limit exceeded — slow down"}),
                    rid, prod)
                resp.headers["Retry-After"] = str(int(RATE_WINDOW))
                resp.headers["X-RateLimit-Limit"] = str(limit)
                resp.headers["X-RateLimit-Remaining"] = "0"
                _log_request(request, 429, started, rid)
                return resp

        # 3) process the request, then attach hardening headers
        resp = await call_next(request)
        _apply_headers(resp, rid, prod)
        _log_request(request, resp.status_code, started, rid)
        return resp
