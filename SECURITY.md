# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 3.0 (web platform) | Yes |
| 1.3 (desktop CLI) | Bug fixes only |
| 2.x | No — upgrade; v2.x enrolment and settings authorisation flaws are fixed in 3.0 |

## Reporting a vulnerability

Do not open a public issue. Email the repository maintainer (see the GitHub profile) with a description, reproduction steps, affected versions and your severity assessment. You will receive an acknowledgement within 72 hours.

## Threat model

| Threat | Control in 3.0 | Residual risk |
|---|---|---|
| Operator replaces someone's template | Server-side RBAC; operators cannot enrol admins; replacements, duplicates and operator self-enrolment need admin approval (optional four-eyes); every change audited and alerted | A colluding administrator |
| Enrolment without the subject | Consent requires the subject's own password (rate-limited like logins) | Subject coerced or shares password |
| Photo / screen replay | Randomised active challenge validated on the server (direction + movement + blink sequence) plus MiniFASNet; identity matched at start and end of challenge | Not independently measured; see README |
| Uploading stored images instead of camera frames | One-time frame tokens, ordered sequence, single in-flight frame, exact-replay digests, frozen-feed termination, session ownership | **Camera origin cannot be proven by a browser** — virtual cameras / injected streams remain possible |
| Real-time deepfake following instructions | None specific | Out of scope |
| Session IDOR | Sessions are owner-only (404 + HIGH alert on access attempts) | — |
| Brute force | Login + consent attestation lockout, per-operator session rate limit, attempt limits per session | Lockout can be used to annoy (DoS) a known account |
| Token theft | HttpOnly cookies, SameSite=strict, CSRF double-submit, rotating refresh tokens with reuse detection | XSS would still act as the user while the page is open |
| Template theft | Fernet encryption at rest, typed header, never returned by any API | Key compromise; back up and protect `EMBEDDING_ENCRYPTION_KEY` |
| Policy tampering | Admin-only, bounded values (threshold ceiling = published default), weakening changes need confirmation + reason, versioned and logged HIGH | A malicious administrator |
| Data retention | Configurable retention; maintenance job purges events, sessions, digests, login attempts, document portraits | Backups contain data — protect them |

## Hardening checklist

- [ ] HTTPS in front of the app; `COOKIE_SECURE=true`; `CORS_ORIGINS` set to your domain only
- [ ] Strong random `SECRET_KEY`; Fernet `EMBEDDING_ENCRYPTION_KEY` stored in a secret manager **and** backed up
- [ ] Change the initial admin password at first sign-in; create named admin accounts
- [ ] PostgreSQL with encrypted, access-controlled backups; `./backups` treated as sensitive
- [ ] Reverse proxy passes `X-Forwarded-For` and the API trusts only that proxy (`FORWARDED_ALLOW_IPS`)
- [ ] Keep `active_liveness_required` and `passive_liveness_required` enabled
- [ ] Calibrate `match_threshold` with the evaluation module on representative, lawfully obtained data
- [ ] Review alerts daily; resolve with notes
- [ ] Independent penetration test and legal review of biometric processing before go-live

## Privacy

No face images or document images are stored. Templates are encrypted; consent grants and withdrawals are recorded append-only; withdrawal deletes the template; document portrait templates are purged after the comparison or at retention expiry. You are responsible for complying with the law that applies to your deployment.

## FaceVerification 4.0 additions

| Area | Control | Residual risk |
|---|---|---|
| Tenancy | Organisation claim in the session token re-validated against an active membership on every request; every query org-scoped; cross-tenant ids → 404 + alert; SSE bound to org + user | A platform administrator can create organisations; protect those accounts with passkeys and `mfa_required_for_admins` |
| Passkeys | Server challenges (single use, 5 min), origin + RP ID verified server-side, UV required, sign-count regression rejected | The authenticator decides how it verifies the user |
| MFA / step-up | TOTP replay guard, hashed one-time codes, attempt limits + alerts, password step-up only without stronger factors, optional face + factor for high-risk actions | Email codes are as strong as the mailbox |
| QR pairing | Opaque one-time token, atomic claim, operator-typed confirmation code, 3 wrong codes cancel + alert, header credential (not cookies) | The phone's camera origin is not provable |
| Devices / kiosks | Key generated on device, signed requests with nonce replay protection, revocation, offline templates limited to authorised people, encrypted to the device key, bounded validity, hash-chained audit | Keys are OS/passphrase protected, **not hardware-backed**; revocation reaches a disconnected kiosk only on reconnect |
| Reviews | Version-guarded decisions, no approval of failed liveness, conflict-of-interest checks, never affects sign-in | Reviewer judgement |
| Integration API | HMAC-hashed scoped keys shown once, rotation/revocation, per-client rate limit, ES256 assertions + one-time introspection, HMAC webhooks, SSRF guard (no private/loopback targets, no redirects) | Clients must keep keys server-side and verify `aud`/`jti` |
| Notifications | No biometric data in bodies; email only when configured and opted in | — |
