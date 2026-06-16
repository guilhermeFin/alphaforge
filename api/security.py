"""HTTP security hardening for the AlphaForge API — defense in depth.

These are the controls that belong INSIDE the app: strict response headers, a
locked-down CORS policy, a request-size cap, and an optional in-process rate
limiter. Edge controls (TLS termination, WAF, CDN/gateway rate limiting, DDoS
protection) belong at the infrastructure layer and are documented in SECURITY.md.

Nothing here processes customer PII — none is stored yet. The data-protection
plan for when accounts/billing land is in SECURITY.md; this module hardens the
public compute surface that exists today.

All behaviour is env-driven so the same code is safe in dev and strict in prod:
  ALPHAFORGE_ENV=production         -> Secure cookies + HSTS
  ALPHAFORGE_CORS_ORIGINS=a,b,c     -> allow these browser origins (default: none)
  ALPHAFORGE_RATE_LIMIT=N           -> N requests / 60s / IP (default 0 = off)
  ALPHAFORGE_MAX_BODY_BYTES=N       -> reject bodies larger than N (default 256 KiB)
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

DEFAULT_MAX_BODY_BYTES = 256 * 1024  # 256 KiB — backtest requests are tiny JSON

# A JSON API renders no HTML, so lock the browser surface down hard: no scripts,
# no framing, no referrer leakage, no ambient device access.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
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
    try:
        return int(os.environ.get("ALPHAFORGE_RATE_LIMIT", "0"))
    except ValueError:
        return 0


class _FixedWindowLimiter:
    """Minimal in-process per-key fixed-window limiter. A BACKSTOP, not the primary
    control — a single process behind a load balancer can't rate-limit globally.
    Production rate limiting belongs at the edge (gateway/CDN). See SECURITY.md."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
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


def install_security(app: FastAPI) -> None:
    """Attach CORS + the security middleware to a FastAPI app (idempotent per app)."""
    origins = _allowed_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_credentials=True,
            allow_methods=["GET", "POST"], allow_headers=["*"], max_age=600,
        )

    @app.middleware("http")
    async def _harden(request: Request, call_next):
        # 1) request-size cap (cheap Content-Length guard; blunts payload-DoS)
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > _max_body_bytes():
                    return JSONResponse(status_code=413, content={"detail": "request body too large"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})

        # 2) optional in-process rate limit (default off; edge limiter is primary)
        limit = _rate_limit()
        if limit > 0:
            ip = request.client.host if request.client else "unknown"
            if not _limiter.allow(ip, limit):
                return JSONResponse(status_code=429, content={"detail": "rate limit exceeded — slow down"})

        # 3) process the request, then attach hardening headers to the response
        resp = await call_next(request)
        for k, v in _SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        if is_production():
            resp.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains")
        return resp
