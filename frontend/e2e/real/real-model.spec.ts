/**
 * Real-model browser tests: production build + real in-browser MediaPipe + real server models.
 * The camera is Chromium's fake device playing a *photo* (with sensor noise), i.e. a presentation
 * attack. These tests therefore assert that real liveness stops it; real match/non-match decisions
 * are covered at service level in backend/tests/test_realmodel.py.
 */
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, expect, test, type Page } from '@playwright/test';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const VIDEOS = join(REPO, '.e2e', 'videos');
const PYTHON = join(REPO, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const BASE = 'http://localhost:4174';
const ADMIN = { email: 'admin@e2e.local', password: 'E2e-Admin-Pass-1' };
const PROBE = { email: 'probe@e2e.local', password: 'Pr0be-Subject-Pass!', name: 'Probe Subject' };
const SEEDED = { email: 'seeded@e2e.local', password: 'Seeded-Subject-Pass!', name: 'Seeded Subject' };

test.skip(!existsSync(join(VIDEOS, 'genuine-a.y4m')), 'run `python scripts/make_test_videos.py` first');
test.describe.configure({ mode: 'serial' });

async function withCamera<T>(clip: string, fn: (page: Page) => Promise<T>): Promise<T> {
  const browser = await chromium.launch({ args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-video-capture=${join(VIDEOS, `${clip}.y4m`)}`] });
  try {
    const ctx = await browser.newContext({ baseURL: BASE, permissions: ['camera'], viewport: { width: 1440, height: 900 } });
    return await fn(await ctx.newPage());
  } finally {
    await browser.close();
  }
}

async function login(page: Page) {
  await page.goto('/login');
  await page.getByTestId('user-menu').or(page.getByLabel('Email')).first().waitFor();
  if (await page.getByTestId('user-menu').isVisible()) return;          // already signed in via this context
  await page.getByLabel('Email').fill(ADMIN.email);
  await page.getByLabel('Password').fill(ADMIN.password);
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await expect(page.getByTestId('user-menu')).toBeVisible();
}

async function apiHeaders(page: Page) {
  const r = await page.request.post('/api/v1/auth/login', { data: ADMIN });
  const token = (await r.json()).access_token as string;
  const csrf = /fv_csrf=([^;]+)/.exec(r.headersArray().filter((h) => h.name.toLowerCase() === 'set-cookie').map((h) => h.value).join('\n'))?.[1] ?? '';
  return { Authorization: `Bearer ${token}`, 'X-CSRF-Token': decodeURIComponent(csrf) };
}

const masked = (page: Page, name: string) => page.screenshot({ path: `e2e/.artifacts/screens/real/${name}.png`, fullPage: true,
  mask: [page.locator('video'), page.locator('[data-testid="camera-stage"] canvas')] });

/** Server-side evidence of the last session: outcome, reason and passive-check labels. */
async function lastSessionEvidence(page: Page, purpose: string) {
  const h = await apiHeaders(page);
  const list = await (await page.request.get(`/api/v1/history/sessions?purpose=${purpose}&page_size=1`, { headers: h })).json();
  const d = await (await page.request.get(`/api/v1/history/sessions/${list.sessions[0].session_id}`, { headers: h })).json();
  const passive = d.events.filter((e: { passive_label?: string }) => e.passive_label)
    .map((e: { passive_label: string; passive_confidence: number }) => `${e.passive_label}@${e.passive_confidence}`);
  return { outcome: d.outcome as string, reason: d.reason_code as string, frames: d.frames as number, passive };
}

async function setPolicy(page: Page, changes: Record<string, unknown>) {
  const h = await apiHeaders(page);
  const r = await page.request.put('/api/v1/policy', { headers: h, data: { changes, confirm: true, reason: 'real-model e2e' } });
  expect(r.status(), await r.text()).toBe(200);
}

test('setup: models ready and test subjects exist', async ({ page }) => {
  await expect.poll(async () => (await page.request.get('/api/v1/health/ready')).status(), { timeout: 180_000 }).toBe(200);
  const h = await apiHeaders(page);
  for (const u of [PROBE, SEEDED]) {
    const r = await page.request.post('/api/v1/users', { headers: h, data: { email: u.email, full_name: u.name, password: u.password, role: 'OPERATOR' } });
    expect([201, 409]).toContain(r.status());
  }
  const seed = spawnSync(PYTHON, ['-m', 'backend.tests.real_seed', SEEDED.email, 'ref_bush_1.jpg'],
    { cwd: REPO, env: { ...process.env, E2E_REAL_SEED: '1' }, encoding: 'utf8' });
  expect(seed.status, seed.stderr).toBe(0);
});

test('enrolment from a photo is rejected by real passive anti-spoofing', async () => {
  await withCamera('genuine-a', async (page) => {
    await setPolicy(page, { active_liveness_required: false });
    await login(page);
    await page.goto('/enrol', { waitUntil: 'domcontentloaded' });
    await page.getByRole('option', { name: /Probe Subject/ }).click();
    await page.getByRole('button', { name: 'Continue' }).click();
    await page.getByLabel(/have read and accept/).check();
    await page.getByLabel(/password \(entered by the subject\)/).fill(PROBE.password);
    await page.getByRole('button', { name: 'Record consent' }).click();
    await page.getByTestId('start-capture').click();
    await expect(page.getByText('Live tracking')).toBeVisible({ timeout: 60_000 });   // real MediaPipe in the browser
    await page.waitForTimeout(1200);
    await masked(page, 'enrol-capturing');
    await expect(page.getByTestId('enrol-result')).toBeVisible({ timeout: 150_000 });
    await masked(page, 'enrol-photo-rejected');
    const ev = await lastSessionEvidence(page, 'ENROL');
    console.log(`photo enrolment → ${ev.outcome}/${ev.reason}, frames=${ev.frames}, passive=${ev.passive.join(' ')}`);
    expect(ev.outcome).toBe('LIVENESS_FAILED');
    expect(ev.passive.some((p: string) => p.startsWith('SPOOF'))).toBe(true);
    const st = await (await page.request.get('/api/v1/users/directory?enrolled_only=true', { headers: await apiHeaders(page) })).json();
    expect(st.users.map((u: { email: string }) => u.email)).not.toContain(PROBE.email);
  });
});

for (const [label, active] of [['passive only', false], ['passive + active challenge', true]] as const) {
  test(`a photo of an enrolled person is not verified (${label})`, async () => {
    await withCamera('genuine-b', async (page) => {
      await setPolicy(page, { active_liveness_required: active });
      await login(page);
      await page.goto('/verify');
      await page.getByRole('option', { name: /Seeded Subject/ }).click();
      await page.getByTestId('start-capture').click();
      await expect(page.getByText('Live tracking')).toBeVisible({ timeout: 60_000 });
      await expect(page.getByTestId('session-outcome')).toBeVisible({ timeout: 150_000 });
      await masked(page, `verify-photo-${active ? 'active' : 'passive'}`);
      const ev = await lastSessionEvidence(page, 'VERIFY');
      console.log(`photo verification (${label}) → ${ev.outcome}/${ev.reason}, frames=${ev.frames}, passive=${ev.passive.join(' ')}`);
      await expect(page.getByTestId('session-outcome')).not.toHaveText('Verified');
      expect(ev.outcome).not.toBe('VERIFIED');
    });
  });
}

test('restore default policy', async ({ page }) => {
  await setPolicy(page, { active_liveness_required: true });
});
