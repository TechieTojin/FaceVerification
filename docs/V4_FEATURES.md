# FaceVerification 4.0 — features, design and limits

This document describes the eleven v4 feature groups: how each works, what it guarantees and what it does
not. Test evidence is in [V4_TEST_REPORT.md](V4_TEST_REPORT.md).

## 16. Passwordless sign-in (passkeys)

* WebAuthn via **py_webauthn 2.7** (server) and **@simplewebauthn/browser 13** (client).
* Every ceremony uses a server-generated 32-byte challenge stored in `auth_challenges`, bound to its purpose
  (and user when known), valid for 5 minutes and consumed with a conditional `UPDATE` — a replayed or raced
  assertion fails. Origin and RP ID are checked against `WEBAUTHN_ORIGINS` / `WEBAUTHN_RP_ID`, never against
  values sent by the browser. User verification is **required**. A sign counter that fails to advance is
  rejected (cloned authenticator).
* Sign-in is usernameless (discoverable credentials). Adding or removing a passkey needs step-up.
* **Passkey + face**: when the organisation policy `passkey_login_requires_face` is on and the member is
  enrolled, a successful passkey assertion yields only a short-lived *pending* token. The browser then runs a
  live face check (same server-side challenge engine) with that token; only a VERIFIED result for the same
  user completes the sign-in (`amr = webauthn + face`). **A face check can never start a sign-in on its own.**
* WebAuthn proves possession of the credential and that the *authenticator* verified its user (PIN,
  fingerprint, face unlock…). It does not tell the server which method the authenticator used, and it does not
  mean the platform's face recognition ran.
* Recovery: password + recovery code; or an administrator resets the member's factors
  (`POST /users/{id}/reset-mfa`, step-up required). That reset is refused for accounts that also belong to
  another organisation, so one organisation can never weaken an account shared with others.

## 26. Multi-factor verification

| Factor | Storage | Notes |
|---|---|---|
| Password | bcrypt | Knowledge factor |
| Passkey | public key only | Possession + authenticator UV |
| Authenticator app (TOTP) | seed Fernet-encrypted | ±1 step skew; each time-step accepted once (replay guard) |
| Email one-time code | keyed hash, 10 min | Only offered when SMTP is configured; otherwise reported unavailable |
| Recovery codes | keyed hash | 10 codes, each consumed once; regenerating invalidates the old set |

* A password sign-in requires a second factor when the account has a passkey or authenticator app, or when
  `mfa_required_for_admins` applies; in the latter case an administrator without factors enrols an
  authenticator app *during* sign-in. Five failed attempts invalidate the pending sign-in and raise an alert.
* **Step-up**: sensitive actions answer `403 STEP_UP_REQUIRED`; the UI re-authenticates and retries. Password
  is accepted for step-up only when the account has no stronger factor. Validity: `step_up_max_age_seconds`.
* **High-risk actions** (organisation creation/suspension, integration credentials, kiosk provisioning and
  offline authorisations) can require, via `high_risk_requires_face`, a live face verification of the
  administrator **plus** an independent factor in the same step-up.
* A PIN is not offered as a factor.

## 23. Multi-organisation support

* `organizations` and `memberships(user, organisation, role)`. Accounts are global; roles are per organisation.
* The active organisation is a signed claim in the session token and is **re-validated against an active
  membership in an active organisation on every request** — the client cannot choose an organisation it does
  not belong to. Disabling a membership or suspending an organisation takes effect on the next request.
* Every tenant-scoped table carries `organization_id` and every query filters by the actor's organisation:
  users/memberships, enrolments and templates, consent, requests, sessions and events, history and CSV export,
  alerts, security events, audit log, policies, documents, evaluation runs, cameras, devices, reviews,
  notifications, pairings, integration clients/credentials/webhooks/assertions, offline events. Cross-tenant
  identifiers return 404; referencing another organisation's session raises a `CROSS_TENANT_ACCESS` alert.
* Duplicate-enrolment detection compares templates **within one organisation only**.
* Account-wide actions (rename, password reset, disable/delete account) are allowed only when the account
  belongs exclusively to the acting organisation; otherwise only the membership is changed.
* Organisation creation is reserved to platform administrators (`users.is_platform_admin`; v3 administrators
  become platform administrators on upgrade).
* SSE subscriptions are bound to organisation + user; events without an organisation are never delivered.
* Migration `0003_v4_platform` moves all existing rows into a "Default organization" without deleting
  anything; it backs up SQLite first. Downgrade restores v3 and refuses to run once a second organisation exists.

## 17 & 18. QR and mobile-to-desktop verification

1. The operator creates a pairing; the QR encodes only `PUBLIC_BASE_URL/m/<192-bit one-time token>`.
2. The phone shows organisation, operator and a masked subject name (`Jane D.`) and asks the person to confirm.
3. Claiming is one atomic transition; the phone receives a pairing secret (header credential, kept in memory /
   tab storage) and a 6-digit code. Simultaneous claims: exactly one wins; reuse and expiry return 410.
4. **The operator types the code shown on the phone** before capture can start. A photographed QR claimed by
   someone else cannot proceed without the operator seeing their code. Three wrong codes cancel the pairing and
   raise `PAIRING_ABUSE`.
5. The phone runs the normal server-side challenge; frames carry the pairing secret and get the same checks as
   desktop frames. The desktop receives progress on the user-addressed `pairing` SSE channel (with a polling
   fallback). One pairing produces one final decision; non-final endings allow retries up to
   `pairing_max_sessions`. Cancellation from either side propagates; abandoned pairings expire.
6. Scanning a QR proves nothing about the phone. The phone's camera origin is no more provable than a desktop's.

## 19. Camera management

Cameras are **registered explicitly** (label, location, hardware label, capabilities, optional device). A browser
can list only cameras on its own machine, after permission; the server never discovers, controls or streams
cameras. While an operator verifies with a registered camera, the browser sends heartbeats (no images) and the
session records the camera; status is Online (heartbeat < 60 s) / Offline / Never seen / Disabled. Disabled
cameras cannot be used for new sessions. Every change is audited.

## 24. Device management

* An administrator creates a device (managed kiosk or browser terminal) and gets a one-time provisioning code
  (15 minutes). The device generates an ECDSA P-256 key pair locally and sends only the public key.
* Device requests are signed: `ECDSA-SHA256("METHOD\nPATH\nTIMESTAMP\nNONCE\nSHA256(body)")` with ±5 min
  clock window and single-use nonces. Invalid signatures and revoked devices raise alerts.
* Heartbeats report software version and health; status Online/Offline is derived from the last signed request.
* **This is proof of key possession, not hardware attestation.** Key protection is shown as the device reports it.

## 25. Offline verification kiosk

The kiosk is a managed runtime (`python -m kiosk`) — not a browser feature — that serves its own UI on
127.0.0.1 and verifies **locally**, with the same engine and models as the server (MediaPipe, MiniFASNet,
ArcFace via DeepFace/ONNX).

* Only people an administrator explicitly authorises for that kiosk are provisioned (limit
  `offline_max_subjects`), and only if their enrolment is active and consented.
* Bundles are encrypted to the kiosk's public key (ephemeral ECDH P-256 → HKDF-SHA256 → AES-256-GCM, AAD bound
  to bundle and device) and signed by the server (ES256, verified against the JWKS).
* Locally, templates are AES-GCM encrypted with a data key protected by **Windows DPAPI (user scope)** or a
  scrypt-derived passphrase key. **Neither is hardware-backed**; anyone with the kiosk's OS account can use them.
* Validity is bounded: `min(device setting, offline_validity_hours policy (≤ 72 h), OFFLINE_MAX_VALIDITY_HOURS)`.
  After expiry, or if the clock moves backwards by more than 5 minutes, the kiosk refuses to verify.
* Every decision is appended to a local audit queue: unique event id, sequence, hash chain and a local HMAC.
  Upload is idempotent (`(device, event_id)` unique); the server re-checks the chain and validity window and
  raises `OFFLINE_AUDIT_INTEGRITY` on a mismatch. Tampered or corrupted local data is refused before use.
* **Revocation is not instantaneous while disconnected**: a kiosk learns about revocation at its next contact
  and then erases its templates. The validity period is the upper bound on that delay.

## 20. Verification review workflow

* Cases are raised automatically for: borderline matches (`|distance − threshold| ≤ review_distance_margin`),
  passes with low passive-liveness confidence, presentation attacks detected by passive anti-spoofing, remote
  captures that failed liveness, and document checks whose authenticity was not verified.
* States: `PENDING → IN_REVIEW → APPROVED | REJECTED | RETRY_REQUESTED`, or `EXPIRED`. Decisions require a note,
  use an optimistic version guard (concurrent decisions: exactly one succeeds) and are audited.
* A review is an **administrative disposition**: the session's technical outcome never changes; a case whose
  liveness failed can be rejected or sent for retry but **never approved**; sign-in face checks are never
  reviewed; the operator and the subject of a session cannot review it.

## 21. Notifications

Per-user, per-organisation records for verification results (remote captures), liveness failures, suspicious
activity, enrolment decisions, devices registered/revoked, QR expiry, review assignment/completion and account
security changes. Delivered live over SSE; unread counter, drawer, history, mark read / all, preferences
(account-security notices cannot be turned off in-app). Email is sent only if the recipient opted in **and**
SMTP is configured; otherwise the record says `UNAVAILABLE`. Delivery is retried with back-off (1, 5, 15, 60
minutes; 5 attempts) and is marked `SENT` only after the SMTP server accepts the message. Notifications never
carry biometric data or scores.

## 22. Integration API and SDK

See the in-app *Developers → API docs* page and [`sdk/typescript/README.md`](../sdk/typescript/README.md).
Scoped, rate-limited `fvk_` credentials (HMAC-hashed, shown once, rotation with grace period, revocation);
hosted verification sessions with `Idempotency-Key`; ES256 assertions (iss, aud, sub, jti, iat, exp, outcome,
review status, policy version) with a JWKS and one-time introspection; HMAC-signed webhooks with retries;
private/loopback webhook targets refused in production. No templates, images or scores are exposed.
