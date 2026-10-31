// Starts an isolated backend for Playwright: fresh SQLite DB, test-only fake inference provider.
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomBytes } from 'node:crypto';

const repo = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const python = process.env.PYTHON ?? join(repo, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const dir = join(repo, '.e2e');
const mode = process.env.E2E_MODE ?? 'fake';      // 'fake' (deterministic) | 'real' (DeepFace + MediaPipe)
const port = process.env.E2E_BACKEND_PORT ?? (mode === 'real' ? '8101' : '8100');
const db = join(dir, `e2e-${mode}.db`);

rmSync(db, { force: true });
rmSync(`${db}-wal`, { force: true });
rmSync(`${db}-shm`, { force: true });
mkdirSync(dir, { recursive: true });

// Fernet key: 32 random bytes, url-safe base64.
const fernet = randomBytes(32).toString('base64').replace(/\+/g, '-').replace(/\//g, '_');
const env = {
  ...process.env,
  APP_ENV: mode === 'real' ? 'development' : 'test',
  INFERENCE_PROVIDER: mode === 'real' ? 'deepface' : 'fake',
  DATABASE_URL: `sqlite:///${db.replace(/\\/g, '/')}`,
  SECRET_KEY: randomBytes(32).toString('hex'),
  EMBEDDING_ENCRYPTION_KEY: fernet,
  INITIAL_ADMIN_EMAIL: 'admin@e2e.local',
  INITIAL_ADMIN_PASSWORD: 'E2e-Admin-Pass-1',
  INITIAL_ADMIN_NAME: 'Ada Admin',
  BCRYPT_ROUNDS: mode === 'real' ? '10' : '4',
  BACKUP_DIR: join(dir, 'backups'),
  CORS_ORIGINS: '["http://localhost:4173","http://127.0.0.1:4173","http://localhost:5174","http://127.0.0.1:5174"]',
  LOG_LEVEL: 'WARNING',
  // The whole suite signs in from 127.0.0.1; lift the per-IP/account login limits for this
  // isolated test backend only (rate limiting itself is covered by backend/tests/test_auth.py).
  LOGIN_RATE_LIMIT_PER_IP: '10000',
  LOGIN_RATE_LIMIT_PER_ACCOUNT: '10000',
  // WebAuthn needs a registrable domain: the browser suite runs on http://localhost:<port>.
  WEBAUTHN_RP_ID: 'localhost',
  WEBAUTHN_ORIGINS: JSON.stringify([`http://localhost:${mode === 'real' ? '4174' : '4173'}`]),
  PUBLIC_BASE_URL: `http://localhost:${mode === 'real' ? '4174' : '4173'}`,
  // The suite runs its own webhook receiver on 127.0.0.1 (private targets are refused in production).
  WEBHOOK_ALLOW_PRIVATE_TARGETS: 'true',
  MAINTENANCE_INTERVAL_SECONDS: '3',
  PYTHONUNBUFFERED: '1',
};

// Throwaway per-run test settings (gitignored .e2e/) so test-only seed scripts can reach the same DB/key.
writeFileSync(join(dir, `backend-env-${mode}.json`), JSON.stringify({
  DATABASE_URL: env.DATABASE_URL, EMBEDDING_ENCRYPTION_KEY: env.EMBEDDING_ENCRYPTION_KEY, SECRET_KEY: env.SECRET_KEY,
  APP_ENV: env.APP_ENV, INFERENCE_PROVIDER: env.INFERENCE_PROVIDER, BACKEND_PORT: port,
}));

if (mode === 'fake') {
  const seed = spawnSync(python, ['-m', 'backend.tests.e2e_seed'], { cwd: repo, env, stdio: 'inherit' });
  if (seed.status !== 0) process.exit(seed.status ?? 1);
}

const child = spawn(python, ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', port,
  '--no-access-log', '--log-level', 'warning'],
  { cwd: repo, env, stdio: 'inherit' });
const stop = () => { child.kill(); process.exit(0); };
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
child.on('exit', (code) => process.exit(code ?? 0));
