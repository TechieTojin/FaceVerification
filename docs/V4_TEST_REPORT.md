# FaceVerification 4.0 — test & visual inspection report

Produced on the development machine (Windows 11, Python 3.10.0, Node 24.13, Chromium via Playwright 1.64),
2026-10-31. Every number below comes from a run on this machine. Nothing here is a biometric accuracy claim.

## Summary

| Suite | Result | Notes |
|---|---|---|
| Backend + CLI (pytest, fake provider) | **281 passed**, 10 skipped | Skipped = opt-in real-model / PostgreSQL tests |
| Backend real models (`RUN_REAL_MODEL_TESTS=1`) | **9 passed** (8 model-pipeline tests + offline kiosk) | incl. the kiosk deciding offline with MediaPipe + MiniFASNet + ArcFace |
| PostgreSQL migrations (`FV_TEST_POSTGRES_URL`) | **not run** | a PostgreSQL service exists on this machine but belongs to another setup; no credentials were used |
| Frontend unit (Vitest) | **39 passed** (Vitest 5.0.5) | |
| Frontend type-check (`tsc -b`, strict) | **0 errors** | |
| Frontend lint (oxlint) | 0 errors, warnings only | |
| Frontend production build | **passes**; `npm audit`: 0 vulnerabilities (after react-router 7.18.5 / vitest 5.0.5) | |
| TypeScript SDK (`node --test`, strict `tsc`) | **4 passed**, type-check clean | webhook signature vector produced by the Python server |
| Playwright deterministic E2E | **52 of 52 passed** in one complete run (11.0 min) after the selector fix; the long review-workflow test was given an explicit 3-minute timeout (it had exceeded the 60 s default on a slower run) | isolated backend, synthetic marker camera, virtual WebAuthn authenticator |
| Docker build / Compose / CI | **not run** | Docker is not installed; CI not triggered (nothing pushed) |
| Physical hardware (camera with a live person, phone, hardware passkey, managed kiosk) | **not performed** | see *Not performed* |

## New backend test files (v4)

| File | Tests | What it proves |
|---|---|---|
| `test_tenancy.py` | 8 | Users, templates, enrolments, sessions, history, CSV, alerts, policy, audit, documents and notifications are invisible/immutable across organisations; cross-tenant access raises an alert in the requesting org; switching requires an active membership; disabled membership / suspended org lose access immediately; shared accounts cannot be renamed, reset or disabled by one org; invitations (new + existing accounts, one-time, email UNAVAILABLE without SMTP); only platform admins create orgs, with step-up |
| `test_migrations.py` | 4 | Fresh schema == models; v2.1 → v4 data mapping; v3 → v4 upgrade with backup, **downgrade and re-upgrade round trip**; downgrade refused with two orgs (found and fixed a downgrade bug that dropped referenced tables) |
| `test_passkeys_mfa.py` | 11 | Real WebAuthn attestations/assertions from a software authenticator: registration needs step-up; usernameless sign-in; invalid, forged, expired and **replayed** challenges; cloned authenticator (non-advancing counter); wrong origin; missing UV; revoked credential; passkey + face (impostor refused, login token cannot touch other sessions, single use); TOTP sign-in, wrong code, **code replay**, recovery code once, factor removal needs strong step-up; attempt limit + alert; policy-required admin MFA with enrolment during sign-in; passkey as second factor; email OTP unavailable without SMTP; face + factor high-risk step-up; admin-assisted factor reset (exclusive accounts only) |
| `test_email.py` | 5 | Against a real local SMTP server (aiosmtpd): opted-in delivery, retries with back-off then FAILED, UNAVAILABLE without SMTP, email OTP sign-in and expiry, invitation email |
| `test_pairing.py` | 6 | Full QR flow with one final decision; desktop receives claimed/confirmed/progress/result events addressed only to the operator; expired, invalid, reused, regenerated and cancelled links; phone- and operator-side cancellation; 3 wrong codes cancel + alert; **4 simultaneous claims → exactly one wins**; operator/org isolation; pairing secret drives only its own session; phone never sees scores |
| `test_reviews.py` | 6 | Borderline match opens a review; approval is a disposition only (technical outcome unchanged); note required; second decision refused; spoof case cannot be approved; conflict of interest; assignment; **concurrent decisions → exactly one wins**; expiry; disabled by policy; cross-org |
| `test_integrations.py` | 4 | Hosted verification end to end; idempotency (replay + key reuse with different body); ES256 assertion verified against JWKS (iss/aud/sub/jti/exp); one-time introspection; tamper rejected; signed webhook verified; retries/back-off/FAILED/manual retry; scopes; rate limit; invalid secret alert; rotation (grace 0) and revocation; tenant isolation; private/http webhook targets refused |
| `test_devices_kiosk.py` | 6 (+1 DPAPI) | Provisioning code single use; signed requests, wrong key, **nonce replay**, clock skew; stale heartbeat → OFFLINE; revocation + alerts + notifications; camera/device permissions and tenant isolation; disabled camera refused; **kiosk: provision → network down → local verification (genuine VERIFIED, impostor NOT_MATCHED, unauthorised person refused) → reconnect → upload once; re-upload acknowledged as duplicates**; expiry, clock rollback, corrupted template, tampered audit log, forged chain flagged by the server; revocation wipes templates; revoked enrolments drop out of the next bundle; DPAPI round trip |

## Defects found by the tests and fixed during v4

| Defect | Found by | Fix |
|---|---|---|
| Migration downgrade dropped `users` and other tables referenced by v4 tables | round-trip migration test | drop only the v4 table set |
| Device clock check compared against `utcnow().timestamp()` (local-time interpretation; IST = 5.5 h skew) | device signature test | compare with `time.time()` |
| Assertion signing committed inside the terminal-hook SAVEPOINT | integration test | hook paths flush only |
| Explicit SQLite `BEGIN` made read-then-write requests fail under concurrency ("database is locked") | concurrent pairing-claim test | reverted to pysqlite's write-time transactions |
| Organisation creation was not marked high-risk | high-risk step-up test | `require_step_up(high_risk=True)` |
| Account-level alerts (refresh reuse, lockout) had no organisation and were invisible | auth test | raised in every organisation of the user |
| Header-credential requests from a signed-in browser would fail CSRF | design review | header transports also send the CSRF token |
| Kiosk "Authorised people" could only be set after provisioning | kiosk E2E | available while the device is pending |
| Ten table headers in the new pages lacked a header row (`<th>` directly in `<thead>`) | kiosk E2E row count | wrapped in `<tr>` |
| Review decision buttons stayed enabled while an assignment was in flight (stale version → refused) | review E2E | buttons disabled while any review action runs; API regression test added |
| Password-login E2E locator matched the new passkey button | E2E | exact locators |

## Visual inspection

Screenshots: `frontend/e2e/.artifacts/screens/` (gitignored) — v4 flows at the top level, `responsive/<width>/` for
every screen at 1440 / 1280 / 768 / 390 px, `v4/` for stress states and dark mode. Inspected by eye and fixed:

| Issue | Fix |
|---|---|
| Both "Live verification" and "QR verification" highlighted in the navigation on `/verify/qr` | exact match for `/verify` |
| QR progress card said "Reconnecting…" after the pairing had finished | shows "Finished" |
| At 390 px the organisation switcher ran under the notification bell (overlap, not overflow, so the automated overflow check passed) | switcher shrinks and truncates; right-hand controls never shrink |
| Review "Expires" and kiosk "valid until" showed the same date twice; raw reason code in history | single date; humanised reasons |

Screens reviewed in this release: QR desktop (waiting, confirm, progress, result, expired), phone (intro, code,
result, expired, cancelled, camera denied, reconnecting), account security (empty, passkey, TOTP set-up, recovery
codes), sign-in (MFA, revoked passkey), review queue/detail (incl. concurrent-decision refusal), cameras, devices
(provisioning code, revoked), developer client (keys, webhooks, deliveries), API docs, organisation, invitation,
notifications (drawer, page, 40+ items at 390 px), kiosk (online, offline verified, revoked), dark mode for the new
pages, and every screen at four widths. Note: the app scrolls inside `<main>`, so "full page" captures of long
lists show only the first screen of the list; horizontal overflow is asserted separately in every responsive test.

## Not performed (and why)

* **Physical camera with a live person**, **physical phone** (QR flow on a real device over HTTPS), **hardware
  passkey / platform authenticator** (only Chromium's virtual authenticator was used), **managed kiosk hardware**:
  not available to this automated environment.
* **Docker build, Compose start-up, CI** — Docker is not installed; nothing was pushed.
* **PostgreSQL migrations** — an opt-in test exists (`test_migrations_postgres.py`); it needs an empty database URL.
* **Biometric accuracy / spoof resistance** — needs representative labelled data (Model performance page).

## Real-model observation (not an accuracy evaluation)

The offline kiosk, provisioned through the real server path and then cut off from the network, ran MediaPipe,
MiniFASNet and ArcFace **locally**. With active liveness disabled for the test (still photos cannot blink or turn),
its local MiniFASNet labelled both still photographs — the enrolled person and an impostor — as presentation
attacks (`LIVENESS_FAILED`), and neither was verified. A real `VERIFIED` on a kiosk needs a live person at a camera.
