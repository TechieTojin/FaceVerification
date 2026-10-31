# Deployment, operations, backup & recovery — FaceVerification 3.0

> The Docker/Compose files were written for this release but **could not be executed on the development machine** (Docker not installed). Validate them in your environment before relying on them.

## Topology

```
Internet ──TLS──▶ load balancer / ingress ──▶ web (nginx, static UI, /api proxy) ──▶ api (FastAPI, 1 worker) ──▶ PostgreSQL
```

* Run **one uvicorn worker per API container**. The inference pool and the SSE event bus are per process; scale out with more containers behind a sticky-free proxy (each container serves its own SSE subscribers; dashboards also refresh on a timer).
* Inference is CPU-bound: size `INFERENCE_WORKERS` to physical cores minus one, and `INFERENCE_QUEUE_LIMIT` to the backlog you accept before returning `503 Retry-After`.

## Steps

1. `python scripts/fetch_models.py` (downloads and SHA-256-checks the MediaPipe landmarker).
2. Copy `.env.example` → `.env`; set at least `SECRET_KEY`, `EMBEDDING_ENCRYPTION_KEY`, `INITIAL_ADMIN_*`, `POSTGRES_PASSWORD`, `CORS_ORIGINS`.
3. `docker compose up --build`. The API applies migrations at start-up (`AUTO_MIGRATE=true`) and warms up models (`/api/v1/health/ready` returns 200 only when the database is reachable, the schema is at head and models are loaded).
4. Terminate TLS in front of `web:8080`; keep `COOKIE_SECURE=true`.
5. Sign in, change the initial admin password, review **Policy settings**.

Without containers: create the venv as in the README, set `APP_ENV=production`, and run
`uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips=<proxy-ip>`
behind nginx using `frontend/nginx.conf` (adjust `proxy_pass`) and the output of `npm run build`.

## Health endpoints

| Endpoint | Meaning |
|---|---|
| `GET /api/v1/health` | Liveness (process up) |
| `GET /api/v1/health/ready` | Readiness: DB ping, Alembic revision == head, models READY (503 otherwise) |
| `GET /api/v1/system/health` | Admin detail: inference pool, latency percentiles, model status, maintenance, SSE subscribers |

## Database & migrations

* SQLite is the default for single-machine use; PostgreSQL is supported via `DATABASE_URL=postgresql+psycopg://…` (driver pinned in `backend/requirements_backend.txt`). **PostgreSQL has not been exercised in this release's test runs** — run the backend test suite against a PostgreSQL `DATABASE_URL` before production use.
* `python -m backend.manage check | migrate | backup`.
* Upgrading from v2.1: the v2.1 schema is detected (no `alembic_version`), stamped at `0001_v21_baseline`, then upgraded. A SQLite file is backed up first to `BACKUP_DIR`. v2.1 templates are kept but marked `REENROLMENT_REQUIRED`; legacy settings are recorded in policy v1's change summary but not applied. Downgrade is not supported — restore the backup instead.

## Backups

| Data | How |
|---|---|
| SQLite | `python -m backend.manage backup` (online, consistent copy including WAL) on a schedule; copy `BACKUP_DIR` off-host, encrypted |
| PostgreSQL | `pg_dump -Fc faceverification > fv-$(date +%F).dump` (or managed snapshots); encrypt at rest |
| `EMBEDDING_ENCRYPTION_KEY` | Store in a secret manager with a separate, access-controlled backup. Without it every template is unrecoverable; with it, a stolen backup exposes templates — keep them apart |
| `SECRET_KEY` | Secret manager; rotating it signs everyone out (no data loss) |

Backups contain encrypted biometric templates and personal data: apply the same retention and access rules as the live system.

## Recovery

1. Stop the API.
2. SQLite: replace the database file with the backup (remove stale `-wal`/`-shm` files). PostgreSQL: `pg_restore --clean -d faceverification fv-YYYY-MM-DD.dump`.
3. Ensure the **same** `EMBEDDING_ENCRYPTION_KEY` is configured.
4. Start the API; it migrates the restored database to head if needed and reports readiness.
5. Check *System health*, then verify one known enrolled user.

## Key rotation (embedding key)

There is no in-place re-encryption tool in 3.0. To rotate: export nothing, plan a re-enrolment window, set the new key, revoke existing enrolments (they become undecryptable), and re-enrol. Document this in your operating procedures.

## Retention

Configured on the Policy page (`retention_*`); the maintenance job runs every `MAINTENANCE_INTERVAL_SECONDS` and deletes old frame events, closed sessions (and their challenge records), login attempts, expired refresh tokens and replay digests, and purges document portrait templates.

## CI

`.github/workflows/ci.yml` runs backend + CLI tests, frontend type-check/lint/unit tests/build and the deterministic Playwright suite on Ubuntu. It has not been executed on GitHub from this machine.

## FaceVerification 4.0

### Upgrade
The start-up migration `0003_v4_platform` backs up SQLite, then moves every existing row into a "Default
organization". To roll back: stop the API and restore the backup from `BACKUP_DIR`
(`backups/<db>-<timestamp>-pre-0003_v4_platform.db`), or run `alembic downgrade 0002_v3_platform` while only one
organisation exists (the downgrade refuses otherwise). To test the migrations on PostgreSQL, point
`FV_TEST_POSTGRES_URL` at an empty throwaway database and run `pytest backend/tests/test_migrations_postgres.py`.

### Passkeys
Passkeys only work on a secure context with a domain name: serve the UI over HTTPS (or `http://localhost` for
development), set `WEBAUTHN_RP_ID` to the registrable domain (e.g. `fv.example.com`) and `WEBAUTHN_ORIGINS` to the
exact origins (e.g. `["https://fv.example.com"]`). Changing the RP ID later invalidates all registered passkeys.

### Phones (QR / hosted links)
`PUBLIC_BASE_URL` must be reachable from the phone over **HTTPS** — browsers allow camera access only on secure
origins. Behind a reverse proxy, keep `/api/` and the SPA on the same origin.

### Email
Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_STARTTLS`, `SMTP_FROM`. Without them email is
reported as unavailable and email sign-in codes are not offered.

### Offline kiosk
On the kiosk machine (Windows: DPAPI protects the local keys under the kiosk's Windows account):

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements_backend.txt
.venv\Scripts\pip install --no-deps -r backend\requirements_mediapipe.txt
.venv\Scripts\python scripts\fetch_models.py
cd frontend; npm ci; npm run build; cd ..
# 1. In the web UI: Devices & kiosks → Register device → Managed kiosk, allow offline → copy the one-time code
# 2. Devices → <kiosk> → Authorised people → add the people who may verify here
.venv\Scripts\python -m kiosk provision --server https://fv.example.com --code <CODE>
.venv\Scripts\python -m kiosk serve --ui-dir frontend\dist      # UI at http://127.0.0.1:8300/kiosk
```

On non-Windows hosts use `--protection passphrase` with `KIOSK_PASSPHRASE` set (≥ 12 characters). The kiosk keeps
its data in `kiosk_data/` (gitignored): `kiosk.json` (device id, protected keys) and `kiosk.db` (encrypted
templates, audit queue). It syncs every 30 s when the server is reachable. `python -m kiosk wipe` erases all local
data. Run it under a dedicated, locked-down OS account; the keys are not hardware-protected.
