import { generateKeyPairSync } from 'node:crypto';
import type { Browser } from '@playwright/test';
import { apiLogin, expect, installFakeCamera, login, shot, test, USERS } from '../fixtures';
import { confirmStepUp } from '../v4';

const RITA = { email: 'rita@e2e.local', password: 'E2e-User-Pass-1' };

test('cameras: register, rename, heartbeat while in use, disable', async ({ page, problems }) => {
  await installFakeCamera(page, { mode: 'ok' });
  await login(page, USERS.admin);
  await page.goto('/cameras');
  await shot(page, 'cameras-empty');
  await page.getByTestId('register-camera').click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('Label').fill('Front desk');
  await dialog.getByLabel('Location or area').fill('Lobby');
  await dialog.getByRole('button', { name: 'Detect' }).click();
  await dialog.getByLabel('Physical camera').selectOption('Synthetic test camera');
  await dialog.getByRole('button', { name: 'Save' }).click();
  const row = page.getByRole('row', { name: /Front desk/ });
  await expect(row).toContainText('Never seen');

  await row.getByRole('button', { name: 'Edit' }).click();
  await page.getByRole('dialog').getByLabel('Label').fill('Front desk A');
  await page.getByRole('dialog').getByRole('button', { name: 'Save' }).click();
  await expect(page.getByRole('row', { name: /Front desk A/ })).toBeVisible();

  // Use it for a verification: the browser sends heartbeats and the session records the camera.
  await page.goto('/verify');
  await page.getByLabel('Registered camera (optional)').selectOption({ label: 'Front desk A — Lobby' });
  await page.getByRole('option', { name: /Alice Enrolled/ }).click();
  await page.getByTestId('start-capture').click();
  await expect(page.getByTestId('capture-running')).toBeVisible();
  await page.getByTestId('cancel-capture').click();
  await page.goto('/cameras');
  await expect(page.getByRole('row', { name: /Front desk A/ })).toContainText('Online');
  await shot(page, 'cameras-online');
  await page.getByRole('row', { name: /Front desk A/ }).getByRole('button', { name: 'Disable' }).click();
  await expect(page.getByRole('row', { name: /Front desk A/ })).toContainText('Disabled');
  await page.getByRole('row', { name: /Front desk A/ }).click();
  await expect(page.getByTestId('camera-detail')).toContainText('Camera disabled');
  await shot(page, 'camera-detail');
  expect(problems).toEqual([]);
});

test('operators cannot manage cameras or devices', async ({ request }) => {
  const h = await apiLogin(request, USERS.operator);
  expect((await request.post('/api/v1/cameras', { headers: h, data: { label: 'x' } })).status()).toBe(403);
  expect((await request.post('/api/v1/devices', { headers: h, data: { name: 'x', kind: 'KIOSK' } })).status()).toBe(403);
});

test('devices: register with a one-time code, view status, revoke', async ({ page, request, problems }) => {
  await login(page, USERS.admin);
  await page.goto('/devices');
  await page.getByTestId('register-device').click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('Name').fill('Reception kiosk');
  await dialog.getByLabel('Location').fill('Reception');
  await dialog.getByRole('button', { name: 'Register' }).click();
  await confirmStepUp(page, { password: USERS.admin.password });
  const code = await page.getByTestId('provisioning-code').getByTestId('copy-value').inputValue();
  expect(code).toMatch(/^[A-Z0-9]{12}$/);
  await shot(page, 'device-provisioning-code');
  await page.getByRole('button', { name: 'Done' }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'Reception kiosk' })).toBeVisible();
  await expect(page.getByText('Pending')).toBeVisible();
  const deviceId = page.url().split('/').pop()!;

  // An invalid code is refused (signed device requests themselves are covered by backend tests).
  const pem = generateKeyPairSync('ec', { namedCurve: 'P-256' }).publicKey.export({ type: 'spki', format: 'pem' }).toString();
  expect((await request.post('/api/v1/device-api/provision', { data: { code: 'WRONGWRONG12', public_key_pem: pem } })).status()).toBe(401);
  await page.getByTestId('revoke-device').click();
  await page.getByRole('dialog').getByLabel(/Reason/).fill('Lost in transit');
  await page.getByRole('dialog').getByRole('button', { name: 'Revoke device' }).click();
  await expect(page.getByText('Lost in transit')).toBeVisible();
  await shot(page, 'device-revoked');
  // A revoked device cannot be provisioned with its old code either.
  expect((await request.post('/api/v1/device-api/provision', { data: { code, public_key_pem: pem } })).status()).toBe(401);
  expect(deviceId).toBeTruthy();
  expect(problems).toEqual([]);
});

async function borderlineVerification(browser: Browser) {
  // Identity 15 is a deterministic "near" variant of identity 1 (distance 0.55); a 0.57 threshold makes it borderline.
  const ctx = await browser.newContext({ permissions: ['camera'] });
  const page = await ctx.newPage();
  await installFakeCamera(page, { mode: 'ok', autopilot: true, scene: { identity: 15 } });
  await login(page, USERS.operator);
  await page.goto('/verify');
  await page.getByRole('option', { name: /Alice Enrolled/ }).click();
  await page.getByTestId('start-capture').click();
  await expect(page.getByTestId('session-outcome')).toHaveText('Verified', { timeout: 40_000 });
  await ctx.close();
}

test('review workflow: borderline match → notification → assign → decide; concurrent decision refused', async ({ page, browser, request, problems }) => {
  test.setTimeout(180_000);              // two complete verifications plus two reviewer sessions
  const admin = await apiLogin(request, USERS.admin);
  const r = await request.put('/api/v1/policy', { headers: admin, data: { changes: { match_threshold: 0.57 }, reason: 'E2E review' } });
  expect(r.ok()).toBeTruthy();
  try {
    await login(page, RITA);
    const before = Number((await page.getByTestId('notification-bell').getAttribute('aria-label'))?.match(/(\d+) unread/)?.[1] ?? 0);
    await borderlineVerification(browser);
    await borderlineVerification(browser);
    await expect(page.getByTestId('unread-count')).not.toHaveText(String(before), { timeout: 15_000 });
    await page.getByTestId('notification-bell').click();
    await shot(page, 'notifications-drawer');
    await page.getByTestId('notification-item').filter({ hasText: 'New verification to review' }).first().click();
    await expect(page.getByRole('heading', { level: 1, name: /Review: Alice Enrolled/ })).toBeVisible();
    await shot(page, 'review-detail');

    await page.goto('/reviews');
    // Other specs legitimately raise cases too (a spoof attempt, document checks); count the borderline ones.
    const borderline = page.getByTestId('review-row').filter({ hasText: 'borderline' });
    await expect(borderline).toHaveCount(2);
    await shot(page, 'review-queue');
    await borderline.first().click();
    await page.getByRole('button', { name: 'Assign to me' }).click();
    await expect(page.getByText('In review').first()).toBeVisible();          // assignment landed (new version)
    await page.getByLabel('Reason for the decision (required)').fill('Compared with the ID card on file');
    await page.getByTestId('approve-review').click();
    await expect(page.getByText('Approved').first()).toBeVisible();

    // Concurrent decision: two reviewers open the same case; the second decision is refused.
    await page.goto('/reviews');
    await page.getByTestId('review-row').filter({ hasText: 'borderline' }).first().click();
    const url = page.url();
    const other = await browser.newContext();
    const p2 = await other.newPage();
    await login(p2, USERS.admin);
    await p2.goto(new URL(url).pathname);
    await page.getByLabel('Reason for the decision (required)').fill('Mismatch in hairline');
    await p2.getByLabel('Reason for the decision (required)').fill('Looks fine');
    await page.getByTestId('reject-review').click();
    await expect(page.getByText('Rejected').first()).toBeVisible();
    await p2.getByTestId('approve-review').click();
    await expect(p2.getByText(/decided this case first|already rejected/)).toBeVisible();
    await shot(p2, 'review-concurrent-refused');
    await other.close();

    // Operators cannot open the queue.
    const op = await apiLogin(request, USERS.operator);
    expect((await request.get('/api/v1/reviews', { headers: op })).status()).toBe(403);
  } finally {
    await request.put('/api/v1/policy', { headers: admin, data: { changes: { match_threshold: 0.68 }, confirm: true, reason: 'E2E restore' } });
  }
  expect(problems).toEqual([]);
});

test('notification centre: mark read, preferences, reconnect', async ({ page, problems }) => {
  await login(page, RITA);
  await page.goto('/notifications');
  await expect(page.getByRole('heading', { level: 1, name: 'Notifications' })).toBeVisible();
  await shot(page, 'notifications-page');
  await page.getByRole('button', { name: 'Mark all read' }).first().click();
  await expect(page.getByTestId('unread-count')).toHaveCount(0);
  await page.getByLabel('Device revoked by email').check();
  await page.getByTestId('save-preferences').click();
  await expect(page.getByText('Preferences saved')).toBeVisible();
  await page.reload();
  await expect(page.getByLabel('Device revoked by email')).toBeChecked();
  // Live stream: drop it while the server is unreachable → offline indicator; when reachable again it reconnects.
  await expect(page.getByTestId('live-indicator')).toContainText('Live', { timeout: 20_000 });
  await page.route('**/api/v1/stream**', (route) => route.abort('internetdisconnected'));
  await page.evaluate(() => (window as unknown as { __FV_SSE_DROP__: () => void }).__FV_SSE_DROP__());
  await expect(page.getByTestId('live-indicator')).toContainText(/Offline|Connecting/, { timeout: 10_000 });
  await shot(page, 'notifications-stream-offline');
  await page.unroute('**/api/v1/stream**');
  await expect(page.getByTestId('live-indicator')).toContainText('Live', { timeout: 45_000 });
  expect(problems).toEqual([]);
});
