import type { Page } from '@playwright/test';
import { apiLogin, expect, installFakeCamera, login, setAutopilot, setScene, shot, test, USERS } from '../fixtures';

async function openVerify(page: Page, subject = 'Alice Enrolled') {
  await page.goto('/verify');
  await page.getByRole('option', { name: new RegExp(subject) }).click();
}

const outcome = (page: Page) => page.getByTestId('session-outcome');

test.describe('live verification', () => {
  test('matching face completes the server challenge and is verified', async ({ page, problems }) => {
    await installFakeCamera(page, { mode: 'ok', autopilot: false, scene: { identity: 1 } });
    await login(page, USERS.operator);
    await openVerify(page);
    await page.getByTestId('start-capture').click();
    await expect(page.getByTestId('capture-running')).toBeVisible();
    await expect(page.getByRole('list', { name: 'Liveness challenge' }).getByRole('listitem')).toHaveCount(4);
    await page.waitForTimeout(800);
    await shot(page, 'verify-running');
    await setAutopilot(page, true);
    await expect(outcome(page)).toHaveText('Verified', { timeout: 30_000 });
    await expect(page.getByText(/Cosine distance 0\.0\d+ vs\. threshold 0\.680/)).toBeVisible();
    await shot(page, 'verify-verified');
    // camera is released after the decision
    await expect(page.getByText('Camera is off')).toBeVisible();
    expect(problems).toEqual([]);
  });

  test('non-matching face is not matched', async ({ page, problems }) => {
    await installFakeCamera(page, { mode: 'ok', autopilot: true, scene: { identity: 2 } });
    await login(page, USERS.operator);
    await openVerify(page);
    await page.getByTestId('start-capture').click();
    await expect(outcome(page)).toHaveText('Not matched', { timeout: 30_000 });
    await shot(page, 'verify-not-matched');
    expect(problems).toEqual([]);
  });

  test('spoof is rejected by liveness', async ({ page, problems }) => {
    await installFakeCamera(page, { mode: 'ok', autopilot: true, scene: { identity: 1, real: false } });
    await login(page, USERS.operator);
    await openVerify(page);
    await page.getByTestId('start-capture').click();
    await expect(outcome(page)).toHaveText('Liveness failed', { timeout: 40_000 });
    await shot(page, 'verify-liveness-failed');
    expect(problems).toEqual([]);
  });

  test('no face and multiple faces produce guidance without verifying', async ({ page, problems }) => {
    await installFakeCamera(page, { mode: 'ok', scene: { faces: 0 } });
    await login(page, USERS.operator);
    await openVerify(page);
    await page.getByTestId('start-capture').click();
    await expect(page.getByTestId('guidance-headline')).toHaveText(/No face detected/);
    await shot(page, 'verify-no-face');
    await setScene(page, { faces: 2 });
    await expect(page.getByTestId('guidance-headline')).toHaveText('Only one person should be visible');
    await shot(page, 'verify-multiple-faces');
    await setScene(page, { faces: 1, widthCode: 1 });
    await expect(page.getByTestId('guidance-headline')).toHaveText('Move closer');
    await expect(page.getByTestId('capture-running')).toBeVisible();
    await page.getByTestId('cancel-capture').click();
    await expect(outcome(page)).toHaveText('Cancelled');
    expect(problems).toEqual([]);
  });

  test('camera permission denied', async ({ page, problems }) => {
    await installFakeCamera(page, { mode: 'denied' });
    await login(page, USERS.operator);
    await openVerify(page);
    await page.getByTestId('start-capture').click();
    await expect(page.getByTestId('camera-error')).toContainText('permission was denied');
    await shot(page, 'verify-camera-denied');
    const sessions = await page.request.get('/api/v1/history/sessions?page_size=100');
    expect((await sessions.json()).sessions.every((s: { status: string }) => s.status !== 'ACTIVE')).toBe(true);
    expect(problems).toEqual([]);
  });

  test('no camera available', async ({ page, problems }) => {
    await installFakeCamera(page, { mode: 'notfound' });
    await login(page, USERS.operator);
    await openVerify(page);
    await page.getByTestId('start-capture').click();
    await expect(page.getByTestId('camera-error')).toContainText('No camera was found');
    expect(problems).toEqual([]);
  });

  test('network disconnection is survived and recovered', async ({ page, context, problems }) => {
    await installFakeCamera(page, { mode: 'ok', autopilot: false, scene: { identity: 1 } });
    await login(page, USERS.operator);
    await openVerify(page);
    await page.getByTestId('start-capture').click();
    await expect(page.getByTestId('capture-running')).toBeVisible();
    await page.waitForTimeout(700);
    await context.setOffline(true);
    await expect(page.getByText('Connection lost — retrying…')).toBeVisible({ timeout: 10_000 });
    await shot(page, 'verify-network-lost');
    await context.setOffline(false);
    await setAutopilot(page, true);
    await expect(outcome(page)).toHaveText('Verified', { timeout: 40_000 });
    expect(problems).toEqual([]);
  });

  test('API failure on session start is shown, not swallowed', async ({ page, problems }) => {
    await installFakeCamera(page, { mode: 'ok' });
    await login(page, USERS.operator);
    await openVerify(page);
    await page.route('**/api/v1/sessions', (r) => r.fulfill({ status: 500, body: '{"detail":"boom"}', contentType: 'application/json' }));
    await page.getByTestId('start-capture').click();
    await expect(page.getByText('Could not continue')).toBeVisible();
    await shot(page, 'verify-api-error');
    expect(problems.filter((p) => !p.includes('HTTP 500'))).toEqual([]);
  });

  test('double-click start creates exactly one session; leaving the page cancels it', async ({ page, request, problems }) => {
    await installFakeCamera(page, { mode: 'ok', scene: { faces: 0 } });
    await login(page, USERS.operator);
    await openVerify(page);
    let creates = 0;
    page.on('request', (r) => { if (r.method() === 'POST' && r.url().endsWith('/api/v1/sessions')) creates++; });
    await page.getByTestId('start-capture').dblclick();
    await expect(page.getByTestId('capture-running')).toBeVisible();
    expect(creates).toBe(1);
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'History' }).click();
    await expect(page.getByRole('heading', { name: 'Verification history' })).toBeVisible();
    const h = await apiLogin(request, USERS.operator);
    await expect.poll(async () => {
      const r = await request.get('/api/v1/history/sessions?page_size=1', { headers: h });
      return (await r.json()).sessions[0].outcome;
    }).toBe('SESSION_CANCELLED');
    expect(problems).toEqual([]);
  });

  test('session timeout ends the session', async ({ page, request, problems }) => {
    test.setTimeout(90_000);
    const admin = await apiLogin(request, USERS.admin);
    const put = await request.put('/api/v1/policy', { headers: admin, data: { changes: { session_timeout_seconds: 30 } } });
    expect(put.status(), await put.text()).toBe(200);
    try {
      await installFakeCamera(page, { mode: 'ok', scene: { faces: 0 } });
      await login(page, USERS.operator);
      await openVerify(page);
      await page.getByTestId('start-capture').click();
      await expect(outcome(page)).toHaveText('Expired', { timeout: 45_000 });
      await shot(page, 'verify-expired');
    } finally {
      await request.put('/api/v1/policy', { headers: admin, data: { changes: { session_timeout_seconds: 120 }, confirm: true, reason: 'e2e restore' } });
    }
    expect(problems).toEqual([]);
  });
});
