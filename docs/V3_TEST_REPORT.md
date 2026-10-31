# FaceVerification 3.0 — Test & visual inspection report

All results below were produced on the development machine (Windows 11, Python 3.10.0, Node 24, Chromium via Playwright 1.64) on 2026-10-30. Nothing here is a biometric accuracy claim.

## Summary

| Suite | Result | Notes |
|---|---|---|
| Backend (pytest, fake provider) | **158 passed**, 8 skipped | Skipped = opt-in real-model tests below |
| Backend real models (`RUN_REAL_MODEL_TESTS=1`) | **8 passed** | MediaPipe + MiniFASNet + ArcFace on 7 public LFW images |
| v1.3 CLI (pytest `tests/`) | **75 passed** | CLI code unchanged |
| Backend + CLI together (as in CI) | **233 passed**, 8 skipped | |
| Frontend unit (Vitest) | **33 passed** | |
| Frontend type-check (`tsc -b`, `strict`) | **0 errors** | |
| Frontend lint (oxlint) | 0 errors, style warnings only | remaining warnings are the intentional "latest-value ref" pattern |
| Frontend production build | **passes**, no chunk-size warning | route-level code splitting |
| Playwright deterministic E2E | **33 passed** (single full run, 5.5 min) | isolated backend + synthetic marker camera |
| Playwright real-model browser | **5 passed** | production build, real in-browser MediaPipe, real server models |
| Migration of a copy of the real local DB | **passed** | v2.1 → 0002; all rows preserved; original file hash unchanged |
| Start-up with the real `.env` (DB copy) | **ready in ~4 s**, all checks green | |

## What the backend tests prove

* **Decision engine** (`test_engine.py`, 24): genuine → VERIFIED, impostor → NOT_MATCHED, decision only after liveness, bracket match catches a face swap, exact threshold behaviour (distance 0.55 verified at 0.68, rejected at 0.50), retry policy, passive spoof, wrong-direction turn, minimum movement, blink ordering, step timeout, multiple faces and tracking jumps restart the round, quality guidance, template errors are ERROR never a match, enrolment sampling.
* **Policy** (`test_policy.py`, 23): bounds/types/cross-field rules, weakening changes need confirmation, versioning and audit, and **each setting changes real behaviour** (threshold, active liveness, challenge length, session timeout, frame limit, quality gate, passive liveness, operator rate limit); running sessions keep their snapshot.
* **Sessions** (`test_sessions_api.py`, 30): every outcome, expiry, inactivity, maintenance expiry, IDOR (404 + alert), stale token/sequence, cross-session replay, frozen-feed termination, supersession, cancellation, closed-session rejection, missing/revoked/withdrawn/legacy enrolment, dimension mismatch, missing encryption key (503), model failure (3× → ERROR), inference busy (503 + claim released), database failure (500, claim released, no decision), **four concurrent submissions → exactly one accepted**, evaluated-step labelling.
* **Enrolment** (`test_enrolment_api.py`, 15): full capture → active template that verifies; subject-password consent; attestation brute-force alert; operator cannot enrol an admin; viewer cannot modify biometrics; replacement stays PENDING (old template in use) until admin approval; reject; four-eyes; duplicate detection (IDs visible to admins only); self-enrolment approval; admin re-enrolment request; consent withdrawal; revoke; inconsistent samples rejected.
* **Auth & RBAC** (`test_auth.py`, `test_users_rbac.py`): cookies/CSRF, uniform login failures, constant-work login, lockout alert, IP limit, refresh rotation and family revocation on reuse, password policy and token invalidation, device sessions, user CRUD guards, a 14-endpoint × 3-role permission matrix, viewers never receive biometric scores.
* **Platform** (`test_platform.py`, `test_migrations.py`, `test_resync.py`, `test_templates.py`): history scoping, CSV export with formula-injection neutralisation, alert lifecycle, document flow (authenticity NOT_PERFORMED, portrait purged), retention, evaluation maths, analytics/dashboard from real data, health not hardcoded, migrations (fresh schema == models; populated v2.1 DB upgrade with backup and data mapping), network resync, template crypto.

## Real-model observations (not an accuracy evaluation)

| Observation | Value |
|---|---|
| Genuine pairs (n=11) cosine distance | mean 0.472, max **0.701** (> 0.68 default → a false reject at default threshold) |
| Impostor pairs (n=10) | mean 0.943, min 0.866 |
| BGR vs channel-swapped input | embedding distance 0.034 (v2.1 colour-order bug was material) |
| MiniFASNet on a raw scanned photo | REAL (1.00) |
| MiniFASNet on the same photos via browser camera pipeline | **SPOOF** 0.51–0.82 in all 9 evaluated frames across 3 sessions |
| Photo presented to the browser (passive only / passive + active) | LIVENESS_FAILED both times; never VERIFIED |
| Frozen camera feed (identical bytes) | server ends session after 5 replays (found by the real-model run, now enforced and tested) |

A real end-to-end **VERIFIED** through the browser could not be demonstrated: it requires a live person in front of a physical camera (see *Not performed*). Real match/non-match decisions are verified at service level (`test_realmodel.py`).

## Playwright scenarios (deterministic suite, all passing)

Login success/failure (no redirect loop), session restore on reload, logout and protected routes, no tokens in web storage, HttpOnly cookies; admin / operator / viewer navigation and forbidden pages; matching face; non-matching face; spoof → liveness failure; no face, multiple faces and "move closer" guidance; camera permission denied; no camera; network disconnection with recovery to VERIFIED; API failure shown; double-click creates one session; leaving the page cancels the session server-side; session timeout; enrolment with subject consent; wrong subject password; operator blocked from enrolling an admin; replacement → pending → admin approval; revocation with typed confirmation; user create (validation) / edit role / disable / enable; history filter + CSV download + session detail; unauthorised enrolment → alert → acknowledge → resolve; policy weakening with confirmation; document check (no-portrait error, live capture, three separate results, portrait purged); analytics/models/system health; password change; 14 screens × 4 widths without horizontal overflow; mobile navigation drawer; mobile live verification; dark theme.

## Visual inspection

Screenshots: `frontend/e2e/.artifacts/screens/` (gitignored), including `responsive/{desktop-1440,laptop-1280,tablet-768,mobile-390}/*.png`, `dark/*.png` and `real/*.png` (camera and overlay masked). Defects found by inspection and fixed:

| Defect | Fix |
|---|---|
| Latency line chart invisible with a single day of data | point markers when ≤ 3 points |
| Model page overflowed at 390 px (grid item min-width) | `min-w-0` on cards |
| Step badge showed a stale server result while tracking saw two faces | labelled "Server:" |
| "0 fps tracking" during the first second | hidden until measured |
| Frame timeline labelled frames with the *next* step | events now record the evaluated step |
| Capture card below the subject picker on phones | challenge card first while capturing |
| Frozen feed showed a misleading "inactivity" expiry | replay notice + server-side FRAME_REPLAY termination |

## Not performed (and why)

* **Physical camera with a live person** — not possible from this automated environment; needed to observe real blink/turn behaviour, real-world latency and a real VERIFIED through the UI.
* **Docker / Compose / CI execution** — Docker is not installed; GitHub Actions not triggered (no push).
* **PostgreSQL** — no server available; schema and migrations are written to be portable but untested on PostgreSQL.
* **Biometric accuracy and spoof-resistance evaluation** — requires a representative, lawfully obtained labelled dataset (use the Model performance page).
* **Dev server on port 5173** — that port was occupied by another local project during testing; the production build and preview server were used instead.
