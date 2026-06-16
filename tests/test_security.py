"""Security controls are executable, not aspirational — each maps to api/security.py."""
from fastapi.testclient import TestClient

import api.main as main
from api import security

client = TestClient(main.app)


def test_security_headers_present_on_every_response():
    h = client.get("/health").headers
    assert h.get("x-content-type-options") == "nosniff"
    assert h.get("x-frame-options") == "DENY"
    assert h.get("referrer-policy") == "no-referrer"
    assert h.get("cache-control") == "no-store"
    assert "default-src 'none'" in h.get("content-security-policy", "")


def test_request_id_header_present():
    rid = client.get("/health").headers.get("x-request-id")
    assert rid and len(rid) >= 16


def test_unknown_field_is_rejected_422():
    # extra='forbid' blocks mass-assignment / unknown-field injection
    r = client.post("/backtest", json={"provider": "synthetic", "factor": "momentum",
                                       "periods": 400, "lookback": 126, "skip": 21,
                                       "n_trials": 5, "seed": 1, "is_admin": True})
    assert r.status_code == 422


def test_oversized_body_is_rejected_413():
    # blocked by the Content-Length cap BEFORE routing/validation/engine work
    r = client.post("/backtest", json={"provider": "synthetic", "junk": "x" * 300_000})
    assert r.status_code == 413


def test_cors_denies_cross_origin_by_default():
    # no ALPHAFORGE_CORS_ORIGINS at import => CORS middleware not installed => no allow-origin
    r = client.get("/health", headers={"origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_rate_limiter_trips_when_enabled(monkeypatch):
    security._limiter._hits.clear()
    monkeypatch.setenv("ALPHAFORGE_RATE_LIMIT", "2")
    c = TestClient(main.app)
    assert c.get("/health").status_code == 200
    assert c.get("/health").status_code == 200
    r = c.get("/health")
    assert r.status_code == 429   # 3rd within the window is throttled
    assert r.headers.get("retry-after") == "60"
    assert r.headers.get("x-ratelimit-limit") == "2"


def test_rate_limiter_off_by_default():
    # default (no env) => unlimited; the rest of the suite hammering the API must not 429
    assert security._rate_limit() == 0


def test_production_hardens_cookie_and_adds_hsts(monkeypatch):
    monkeypatch.setenv("ALPHAFORGE_ENV", "production")
    c = TestClient(main.app)
    r = c.post("/backtest", json={"provider": "synthetic", "factor": "momentum",
                                  "periods": 400, "lookback": 126, "skip": 21,
                                  "n_trials": 5, "seed": 1})
    assert r.status_code == 200, r.text
    setcookie = r.headers.get("set-cookie", "")
    assert "Secure" in setcookie and "HttpOnly" in setcookie
    assert "strict-transport-security" in {k.lower() for k in r.headers}
