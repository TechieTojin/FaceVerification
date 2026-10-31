# FaceVerification 4.0 — Implementation Progress (durable log)

Resume point if work is interrupted. Update after each phase. Nothing is committed or pushed.

## Baseline (2026-10-31, before v4 work)
- Backend + CLI pytest: 233 passed, 8 skipped (opt-in real-model tests).
- New libraries (in `.venv` only): webauthn 2.7.0 (+cbor2, asn1crypto, pyOpenSSL), pyotp 2.9.0,
  aiosmtpd 1.4.6 (local test mail server). Frontend: @simplewebauthn/browser 13, qrcode 1.5.4.
- `npm audit`: react-router (moderate, runtime) and vitest/tinypool (dev-only) advisories pre-exist — review in Phase 7.

## Architecture decisions
- **Tenancy**: `organizations` + `memberships(user, org, role)`. Users are global accounts; the role lives
  on the membership. The active organisation is a signed claim in the access/refresh token and is
  re-validated against an ACTIVE membership on every request (never trusted from the client alone).
  `users.is_platform_admin` gates organisation creation. Existing data → "Default organization".
- **Actor**: every request resolves to an `Actor(kind, org_id, user, role, permissions, ref, amr, ...)`.
  Kinds: USER (cookie/bearer), PAIRING (phone holding a pairing secret), LOGIN (pending passkey login
  doing the face check), API (integration credential), DEVICE (signed device requests).
- **Capture sessions** gain `organization_id`, `owner_kind`, `owner_ref`, `channel`, `camera_id`,
  `pairing_id`, `review_status`. Ownership = (owner_kind, owner_ref) match.
- **Passkeys**: py_webauthn, RP ID/origins from settings, challenges stored server-side, single use, 5 min.
  A face match never issues a session by itself; it can only be an *additional* check after a passkey.
- **MFA**: TOTP (Fernet-encrypted secret, last-used time-step replay guard), email OTP (only when SMTP
  configured), recovery codes (HMAC-hashed, one-time). Step-up claim `su` in the access token.
- **QR / mobile**: opaque one-time pairing token in the QR; the phone exchanges it for a pairing secret,
  shows a 6-digit code, the desktop operator types that code to confirm the pairing before capture.
- **Assertions**: ES256 JWT signed with a DB-stored key (private key Fernet-encrypted), JWKS published.
- **Webhooks**: `t=<ts>,v1=<hmac-sha256>` signature over `"{t}.{body}"`, retries with backoff.
- **Offline kiosk**: separate Python agent (`kiosk/`) reusing the backend engine + providers for local
  inference; device-bound EC P-256 key; bundle encrypted to the device key (ECDH + AES-GCM) and signed by
  the server; local SQLite with encrypted templates and a hash-chained audit queue; idempotent sync.

## Phase checklist
- [x] P1 Tenancy backend: models, migration 0003 (+downgrade, round-trip tested), Actor/deps, all routers/services
  org-scoped, SSE bus org/user scoped, invitations, org service — tests: test_tenancy (8), test_migrations (4). UI pending.
- [x] P2 Auth backend: passkeys (py_webauthn, soft-authenticator tests), passkey→face, MFA (TOTP replay guard, email OTP,
  recovery codes), policy-required MFA enrolment at sign-in, step-up (+face+factor high-risk) — test_passkeys_mfa (10),
  test_email (5, real aiosmtpd). UI pending.
- [x] P3 Cross-device backend: pairing_service (QR, claim, operator code confirm, phone session, SSE progress,
  one final decision) — test_pairing (6). UI pending.
- [x] P4 Ops backend: cameras + devices (signed requests, nonce replay, revocation), review_service (state machine,
  version-guarded decisions, spoof cannot be approved), notifications (+SMTP) — test_reviews (5),
  test_devices_kiosk (6). UI pending.
- [x] P5 Integrations backend: clients/credentials/rotation, hosted sessions, ES256 assertions + JWKS + one-time
  introspection, signed webhooks + retries, idempotency — test_integrations (4); TS SDK `sdk/typescript`
  (4 node:test tests, strict typecheck) + example server.
- [x] P6 Kiosk agent `kiosk/` (DPAPI/passphrase key protection, encrypted local templates, hash-chained MAC'd audit
  queue, offline local inference via backend engine, signed sync, revocation wipe, expiry, clock rollback).
  Kiosk UI pending.
- [x] P6b Frontend: login (passkey, MFA, MFA enrolment, face after passkey), account security, step-up dialog
  (API interceptor), org switcher + organisation page + invite page, notification bell/page, QR desktop + mobile page,
  cameras, devices + detail, review queue + detail, developers (clients, keys, webhooks, deliveries, docs), kiosk UI,
  settings groups. tsc strict clean; vitest 38.
- [x] P7 E2E specs 06–11 (virtual authenticator, desktop + phone contexts, SDK, real kiosk process via toggleable
  proxy, responsive + dark + stress states). Backend+CLI 281 passed / 10 skipped (opt-in). Vitest 39 (vitest 5).
  react-router 7.18.5 + vitest 5.0.5 → npm audit 0. Docs: README, SECURITY, DEPLOYMENT, V4_FEATURES, V4_TEST_REPORT.
- Final: Playwright full run 51/52 (+ the 1 locator fix re-verified 7/7), real-model 9/9, secret scan clean
  (311 files). Nothing committed or pushed. See docs/V4_TEST_REPORT.md.
- [ ] P7 Regression, Playwright (incl. virtual authenticator, two contexts), responsive screenshots, docs

## Notes for resuming
- Migration 0003 create-table section is generated by scratchpad `gen_0003.py` (re-run after model changes; it only
  rewrites the `# <generated-tables>` block).
- SQLite engine now lets SQLAlchemy emit BEGIN (needed for SAVEPOINT in capture terminal hooks).
- Cross-service hooks are registered in `backend/services/wiring.py` (terminal hooks, maintenance tasks).
