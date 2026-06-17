# AlphaForge — Security & Data Protection

> Owner: Security (acting CISO). Last reviewed: 2026-06-16. Review cadence: quarterly,
> and on any change that introduces customer accounts, payments, or stored PII.

## Posture, stated honestly

AlphaForge is built so that the moment it holds customer data, that data is protected by
**defense in depth** — layered controls so that no single failure exposes anything. We do
**not** claim any system is unhackable; that claim is itself a red flag. What we commit to
is: minimize what we hold, encrypt what we keep, gate every access, watch for abuse, and
fail safe.

**Current scope (important):** AlphaForge in production today is a research engine + API + UI
that stores **no customer PII, no passwords, and no payment data** — no accounts are deployed yet.
The security work splits in two:

1. **Harden the surface that exists now** (the public compute API + secrets handling) — done, see *Controls in place*.
2. **Build the data layer secure-by-default before the first customer record is ever written** — the cheapest time to get this right is before there is anything to lose. This foundation is now **built and tested** in the `accounts/` package (see *Data layer*); it ships OFF by default and is wired in additively, so the anonymous compute flow is unchanged until accounts are deliberately enabled and deployed.

## Data inventory & classification

| Data | Sensitivity | Where it lives | Protection |
|------|-------------|----------------|------------|
| Synthetic + public market data (yfinance) | Public | In-memory / cache | None needed; never redistributed raw |
| Third-party vendor data (SimFin/Sharadar/Compustat) | Licensed/Confidential | Per-tier, key-gated | License-scoped; not redistributed; EOD/PIT historical preferred |
| API keys (Anthropic, Nasdaq, SimFin) | **Secret** | `.env` (gitignored), prod secrets manager | Never in code/logs/VCS; rotated; scanned for in CI |
| Session cookie (`af_session`) | Low (opaque id) | Browser cookie | HttpOnly, SameSite=Lax, Secure in prod; no PII inside |
| Account email | PII | `accounts` SQLite store (when enabled) | **Encrypted at rest** (Fernet) + blind-indexed for lookup; never logged |
| Strategy params / run history | Confidential (user IP) | `accounts` store (when enabled) | **Encrypted at rest**; workspace-scoped; never leaves the tenant |
| API keys | **Secret** | `accounts` store (when enabled) | Only a SHA-256 hash + display prefix stored; secret shown once |
| Passwords | — | **never stored** | Delegated to a managed IdP (OIDC); we never see a password |
| Payment / card data | **Critical** | **never stored** | Stripe-hosted; we keep only a Stripe customer-id reference |

**Principle: data minimization.** The most secure record is the one we never collect. We will
collect the minimum to operate, and payment data will be handled so it never touches our servers.

## Threat model & mitigations (current surface)

| Threat | Vector | Mitigation (today) |
|--------|--------|--------------------|
| **Secret leakage** | Key committed to git, printed in logs, in an error | `.env` gitignored; `.env.example` has no values; **gitleaks** scans every push + full history in CI; engine never logs secrets; errors return type+message, not stack traces or env |
| **Supply-chain / dependency CVE** | Vulnerable transitive package | **pip-audit** in CI (weekly + per-push); pinned ranges in `pyproject.toml`; optional extras isolate heavy deps |
| **Injection** | Malicious `provider`/`factor`/ticker input | Strict allow-listed validation in `api/service._validate` (enum providers/factors, bounded numerics, ticker charset/length); no `eval`, no shell, no SQL yet |
| **Payload / compute DoS** | Huge body or hammering the compute-heavy endpoints | 256 KiB request-body cap; optional in-process rate limiter; edge rate limiting + autoscaling in prod (roadmap) |
| **Clickjacking / XSS / MIME sniff** | Browser-side attacks on API responses | Strict security headers: `CSP default-src 'none'`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, COOP/CORP |
| **Cross-origin abuse** | A malicious page calling the API from a browser | CORS default-denies cross-origin; allowed origins are an explicit env allow-list |
| **SSRF / outbound fetch** | yfinance/LLM/data fetches to attacker-controlled hosts | Outbound calls go only to known vendor endpoints; user input never becomes a fetch URL. (When user-supplied URLs are ever added, they must be allow-listed.) |
| **Cookie theft / fixation** | Session cookie exposure | HttpOnly (no JS access), SameSite=Lax (CSRF blunting), Secure in prod (HTTPS-only), opaque random id, idle expiry |
| **MITM** | Plaintext transport | TLS everywhere in prod (HSTS preload-eligible header); HTTP is dev-only |

These map to real code: input validation in `api/service.py`, HTTP hardening in
`api/security.py`, secret hygiene in `.gitignore` + `.env.example` + `.github/workflows/security.yml`.

## Secrets management policy

- Secrets live in `.env` locally (gitignored) and in a **managed secrets store** in production
  (e.g. cloud secrets manager / platform env vars) — never in source, never in container images.
- **Rotation:** rotate on suspected exposure immediately, and on a routine cadence (≤ 90 days)
  for long-lived keys. The GitHub token, Anthropic key, and data-vendor keys are all rotatable.
- **Least privilege:** each key is scoped to the minimum it needs; separate keys per environment.
- **Detection:** CI gitleaks fails the build on any committed secret; if one ever lands, treat it
  as compromised, rotate, and purge from history.

## Data layer (built foundation — `accounts/`, 2026-06-16)

The secure-by-default data layer is implemented and unit-tested (`tests/test_accounts.py`)
ahead of holding any real customer record. It is OFF by default and additive (the public
compute API is untouched unless `ALPHAFORGE_ACCOUNTS`/`ALPHAFORGE_DB_PATH` is set). What it
enforces in code today:

- **No homegrown passwords.** Identity is delegated to a managed IdP via an OIDC seam
  (`AccountService.login_with_identity` takes already-verified claims). Request-time auth
  uses **API keys** — high-entropy bearer secrets stored only as a SHA-256 hash + prefix,
  verified in constant time, shown once. (A dev-login mints a key locally; it is disabled
  in production, where the OIDC login is required.)
- **Per-workspace authorization, deny-by-default** (`accounts/authz.py`): every scoped
  operation re-checks membership for the *target* workspace at a minimum role — the
  BOLA/BFLA defense. The store also filters every read by `workspace_id` (defense in depth).
- **Encryption at rest** (`accounts/crypto.py`): email and the user's strategy params/results
  are Fernet-encrypted before they hit disk; emails are blind-indexed (keyed HMAC) so lookup
  works without storing plaintext. One master key (`ALPHAFORGE_DATA_KEY`) derives both subkeys
  via HKDF and is **required in production** (fail closed).
- **Payments stay out of scope:** only a Stripe customer-id reference is ever stored.
- **Tamper-evident audit log** (`accounts/audit.py` + store): an append-only, per-workspace
  hash chain of auth/key/run/erasure events — `verify()` detects any silent edit, the same
  honesty discipline as the engine's trial ledger.
- **GDPR / LGPD rights:** full per-workspace **export** and **erasure** (cascade delete + a
  system-chain tombstone proving erasure occurred without retaining the erased data).

What remains before turning this on for real customers is **deployment/operations**, not
application design — see the roadmap below.

## Data-protection roadmap (operational preconditions before accounts go live)

The application foundation above is built; these are the deployment/ops preconditions that
remain before storing real customer data:

1. **Authentication — never roll our own.** Use a managed identity provider (Clerk/Auth0):
   no homegrown password storage; passwords (if any) are the provider's problem, hashed with a
   modern KDF. Offer/strongly encourage **MFA**. Short-lived signed sessions; secure logout.
2. **Authorization.** Per-workspace isolation enforced server-side on every request (a user can
   only ever read/write their own workspaces); deny-by-default.
3. **Encryption.** TLS 1.2+ in transit everywhere; **encryption at rest** for the database and
   all backups; field-level encryption for anything especially sensitive.
4. **Payments via Stripe — minimize PCI scope.** Card data is entered into Stripe's elements and
   **never touches our servers**; we store only Stripe customer/subscription ids. No PAN, ever.
5. **Least-privilege infrastructure.** Scoped IAM roles, no shared admin creds, separate
   prod/staging, secrets in a manager, network segmentation, the DB not publicly reachable.
6. **Audit logging.** Append-only logs of auth events, data access, and admin actions — with the
   same honesty discipline as the engine's trial ledger (logged, not silently mutable).
7. **PII minimization, retention & deletion.** Collect the minimum; document a retention schedule;
   support **data export and deletion** (GDPR / Brazil's LGPD — relevant to the Brazil beachhead).
8. **Backups & DR.** Encrypted, access-controlled, and **restore-tested** backups; a documented RTO/RPO.
9. **Vulnerability management.** Keep the CI scans (deps + secrets); add SAST + container scanning;
   a patching SLA by severity; periodic third-party penetration test before/at scale.
10. **Monitoring & alerting.** Anomaly/abuse alerts (auth failures, rate-limit breaches, unusual
    egress); centralized logs; on-call.
11. **Vendor & data-licensing compliance.** Honor market-data licenses (lean on EOD/PIT historical;
    live = premium); a vendor security-review step before adding a data source.

A **SOC 2 Type II** track is the eventual goal for selling into emerging managers/desks; most of
the controls above are the same evidence, so we build toward it from day one rather than retrofitting.

## Regulatory boundary (a security-adjacent control)

AlphaForge is **research software, not investment advice**, and never emits personalized or
specific buy/sell/price-target output. Keeping that line (enforced in product design, not just
disclaimers) keeps us outside the CVM advisory regimes and shrinks our regulatory + liability
surface — a deliberate risk reduction, not just legal framing.

## Incident response (lightweight, real)

1. **Detect** (CI scan failure, alert, or report) → 2. **Contain** (rotate keys, revoke sessions,
isolate) → 3. **Eradicate & recover** (patch, restore from clean backup) → 4. **Notify** affected
users and any legally-required authority within the required window → 5. **Post-mortem**
(blameless, with a tracked fix list). Keep this runbook current as infrastructure grows.

## Responsible disclosure

Found a vulnerability? Email **security@alphaforge.example** (placeholder until the domain is live)
with steps to reproduce. Please don't open a public issue or access data that isn't yours. We aim
to acknowledge within 72 hours, will keep you updated, and will credit reporters who want it. We do
not pursue good-faith researchers who follow this policy.

---

*This document is the security baseline. It is reviewed quarterly and whenever the data we hold
changes — and the controls it describes are verified by the test suite (`tests/test_security.py`)
and the CI scans, so the policy and the code can't quietly drift apart.*
