# AlphaForge — Security Gap Audit & Remediation

> Generated 2026-06-16. Method: AlphaForge's actual security surface was audited against
> best-practice control checklists from a framework-mapped cybersecurity-skills knowledge base
> (Apache-2.0; API-security-posture, gateway controls, rate-limiting, schema-validation, API-key
> controls, DevSecOps, supply-chain/SBOM workflows). Read-only — no third-party scripts executed.
> Controls map to OWASP API Top 10, NIST CSF 2.0, MITRE ATT&CK, and D3FEND.

## Verdict

For a **pre-accounts MVP** (no customer PII, passwords, or payment data stored yet), AlphaForge
was already **above average** — strict security headers, deny-by-default CORS, an allow-listed
input validator, full-history secret scanning, and a written threat model already covered most
best-practice controls. **Bandit SAST came back clean.** The gaps were narrow and cheap, and the
big items (auth, encryption-at-rest, IAM) are correctly *deferred* to the accounts roadmap in
`SECURITY.md`, not present holes. This pass closed the cheap gaps; the rest is tracked below.

## Implemented in this pass

| Control | Gap closed | Where | Framework |
|---|---|---|---|
| **Unknown-field rejection** | `BacktestRequest` silently dropped unknown keys | `extra="forbid"` + `symbols` `max_length` (`api/main.py`) | OWASP API3 (mass assignment) |
| **Request-ID + structured access log** | no abuse-detection substrate | per-request `X-Request-ID` + one JSON log line, no body/secrets (`api/security.py`) | NIST DE.CM-01; D3FEND |
| **Rate limiter on by default in prod** | limiter was off unless an env var was set | prod default 120/60s/IP; `Retry-After` + `X-RateLimit-*` on 429 (`api/security.py`) | OWASP API4; ATT&CK T1499 |
| **Cache-Control + HSTS preload** | results cacheable; HSTS not preload-eligible | `Cache-Control: no-store`; `…; preload` (`api/security.py`) | NIST PR.DS-10 |
| **SHA-pinned CI actions** | `@v4`/`@v5` tags = movable → CI RCE risk | both workflows pin 40-char commit SHAs | ATT&CK T1195.002 (supply chain) |
| **Blocking SAST gate** | no static analysis in CI | Bandit job, fails on findings (`security.yml`) | NIST PR.PS-04 |
| **Dependency CVE remediation** | 2026 CVEs in `starlette`/`urllib3` | version floors `starlette>=1.3.1`, `urllib3>=2.7.0` (`pyproject.toml`) — verified installable, suite green | ATT&CK T1195.001 |
| **Pre-commit secret hook** | secrets only caught server-side (CI) | `.pre-commit-config.yaml` runs gitleaks on staged files | NIST GV.SC-07; ATT&CK T1552 |
| **Bounded session store** | unbounded in-process dict (memory-DoS) | `MAX_SESSIONS` cap with oldest-eviction (`api/main.py`) | OWASP API4 |

All verified by `tests/test_security.py` (now covers headers, request-id, unknown-field 422,
413 body cap, CORS deny, rate-limit + headers, prod Secure-cookie + HSTS).

## Known residual (tracked, accepted for now)

- **`tornado` 2026 CVEs** (CVE-2026-49853/4/5, GHSA-pw6j-qg29-8w7f; fixed in 6.5.6/6.5.7) are a
  **transitive dependency of Streamlit**, which currently caps the version — forcing the bump
  breaks the install. Accepted and tracked until Streamlit ships a compatible release; the weekly
  `pip-audit` scan will flag when it's resolvable. (Streamlit is the dev/UI surface, not the API.)
- **`pip-audit` is advisory (non-blocking) by design.** Secrets (gitleaks) and SAST (bandit) are
  hard-fail gates; dependency CVEs are *triaged* — actionable ones remediated via floors, the rest
  surfaced weekly — because hard-failing on every unactionable transitive CVE is noise, not security.
  Making it blocking against a clean CI baseline is a P1 follow-up.

## Data layer built secure-by-default (follow-up, 2026-06-16)

The application-layer half of the deferred list has since been **built and tested** in the
`accounts/` package (OFF by default, additive — the public compute API is unchanged). This
pulls the following from "deferred" to "implemented foundation", verified by
`tests/test_accounts.py`:

| Control | Implementation | Framework |
|---|---|---|
| No homegrown passwords; managed-IdP (OIDC) seam + API-key auth | `accounts/identity.py`, `service.login_with_identity`; keys stored as SHA-256 hash + prefix, constant-time verify | OWASP API2; NIST PR.AA |
| Per-workspace authorization, deny-by-default | `accounts/authz.require_workspace` (re-checks membership for the target workspace) + store-level `workspace_id` filtering | OWASP API1/API5 (BOLA/BFLA) |
| Encryption at rest + field-level encryption | `accounts/crypto.py` (Fernet via one HKDF-derived master key) for email + run params/results; keyed blind index for lookup | NIST PR.DS-01 |
| Payments minimize PCI scope | only a Stripe customer-id reference is stored; no PAN ever | PCI-DSS scope reduction |
| Append-only, tamper-evident audit log | per-workspace SHA-256 hash chain + `verify()` (`accounts/audit.py`, store) | NIST PR.PS / DE.CM; AU-9 |
| PII export + erasure (GDPR / LGPD) | `service.export_account` / `delete_account` (cascade + system-chain tombstone) | GDPR Art. 15/17; LGPD |

What still remains deferred is **deployment/operations, not application design** — see below.

## Deferred to the infrastructure/operations roadmap (do NOT pull forward)

Still correctly out of scope until accounts deploy: a real **managed IdP** (Clerk/Auth0) wired
to the OIDC seam + **MFA** + signed sessions; a hosted DB with **disk encryption** and a real
**KMS/secrets manager** holding `ALPHAFORGE_DATA_KEY` (the app does field encryption; the
platform must protect the key and the volume); scoped IAM + network segmentation; centralized
logging + anomaly alerting + on-call; encrypted, restore-tested backups (RTO/RPO); periodic
third-party penetration test; SOC 2 Type II evidence program. Edge controls (WAF, CDN/gateway +
distributed rate limiting, DAST vs. staging) belong at infra/scale, not in-app.

## The honest bottom line

No system is unhackable, and the highest-leverage decision is structural: **handle identity through
a managed provider and payments through Stripe, so we never store passwords or card data at all** —
removing the two highest-value targets from our servers entirely. That, plus minimizing what we
collect, is the core of the data-protection posture. This audit hardened the live surface; the data
layer gets built secure-by-default *before* the first customer record exists.
