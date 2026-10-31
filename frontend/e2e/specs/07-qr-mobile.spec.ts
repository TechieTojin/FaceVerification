import { devices, type Browser, type Page } from '@playwright/test';
import { expect, installFakeCamera, login, shot, test, USERS } from '../fixtures';
import { e2eOp } from '../v4';

async function startPairing(page: Page, subject: RegExp) {
  await login(page, USERS.operator);
  await page.goto('/verify/qr');
  await page.getByRole('option', { name: subject }).click();
  const created = page.waitForResponse((r) => r.url().endsWith('/api/v1/pairings') && r.request().method() === 'POST');
  await page.getByTestId('create-pairing').click();
  const pairing = await (await created).json() as { id: string; qr_url: string };
  await expect(page.getByTestId('qr-image')).toBeVisible();
  const link = (await page.getByTestId('pairing-link').textContent())!.trim();
  expect(link).toBe(pairing.qr_url);
  return { id: pairing.id, path: new URL(link).pathname };
}

async function phone(browser: Browser, opts: Parameters<typeof installFakeCamera>[1]) {
  const ctx = await browser.newContext({ ...devices['Pixel 7'], permissions: ['camera'] });
  const page = await ctx.newPage();
  await installFakeCamera(page, opts);
  return { ctx, page };
}

test('desktop shows a QR, the phone verifies, both see the result', async ({ page, browser, problems }) => {
  const { path } = await startPairing(page, /Fiona Phone/);
  await shot(page, 'qr-desktop-waiting');
  const m = await phone(browser, { mode: 'ok', autopilot: true, scene: { identity: 5 } });
  await m.page.goto(path);
  await expect(m.page.getByTestId('mobile-intro')).toContainText('Fiona P.');
  await shot(m.page, 'mobile-intro');
  await m.page.getByLabel(/I am Fiona P\./).check();
  await m.page.getByTestId('claim-pairing').click();
  const code = (await m.page.getByTestId('confirmation-code').textContent())!.trim();
  await shot(m.page, 'mobile-code');

  await expect(page.getByLabel('Code shown on the phone')).toBeVisible();
  await expect(page.getByTestId('pairing-stages')).toContainText('Phone connected');
  await shot(page, 'qr-desktop-confirm');
  await page.getByLabel('Code shown on the phone').fill(code);
  await page.getByRole('button', { name: 'Confirm pairing' }).click();

  await m.page.getByTestId('start-capture').click();
  await expect(page.getByText('Verification in progress on the phone')).toBeVisible({ timeout: 15_000 });
  await shot(page, 'qr-desktop-progress');
  await expect(m.page.getByTestId('mobile-result')).toContainText('Verified', { timeout: 40_000 });
  await shot(m.page, 'mobile-result');
  await expect(page.getByTestId('pairing-result')).toContainText('Verified', { timeout: 15_000 });
  await shot(page, 'qr-desktop-result');
  await m.ctx.close();
  expect(problems).toEqual([]);
});

test('expired, reused and cancelled QR codes', async ({ page, browser }) => {
  const first = await startPairing(page, /Fiona Phone/);
  e2eOp('expire-pairing', first.id);
  await expect(page.getByTestId('pairing-expired')).toBeVisible({ timeout: 15_000 });
  await shot(page, 'qr-desktop-expired');
  const late = await phone(browser, { mode: 'ok' });
  await late.page.goto(first.path);
  await expect(late.page.getByText('This QR code has expired. Ask for a new one.')).toBeVisible();
  await shot(late.page, 'mobile-expired');

  // Regenerate → a new code; claiming it twice fails the second time.
  const regenerated = page.waitForResponse((r) => /\/pairings\/[^/]+\/regenerate$/.test(r.url()));
  await page.getByRole('button', { name: 'Generate new QR' }).click();
  const second = await (await regenerated).json() as { qr_url: string };
  const path = new URL(second.qr_url).pathname;
  const a = await phone(browser, { mode: 'ok' });
  await a.page.goto(path);
  await a.page.getByLabel(/I am Fiona P\./).check();
  await a.page.getByTestId('claim-pairing').click();
  await expect(a.page.getByTestId('confirmation-code')).toBeVisible();
  const b = await phone(browser, { mode: 'ok' });
  await b.page.goto(path);
  await expect(b.page.getByText('This QR code has already been used.')).toBeVisible();

  // Operator cancels → the phone learns about it.
  await page.getByTestId('cancel-pairing').click();
  await expect(a.page.getByTestId('mobile-ended')).toContainText('cancelled', { timeout: 10_000 });
  await shot(a.page, 'mobile-cancelled');
  for (const c of [late, a, b]) await c.ctx.close();
});

test('mobile camera denied, then a network interruption during capture recovers', async ({ page, browser }) => {
  const { path } = await startPairing(page, /Fiona Phone/);
  const m = await phone(browser, { mode: 'denied' });
  await m.page.goto(path);
  await m.page.getByLabel(/I am Fiona P\./).check();
  await m.page.getByTestId('claim-pairing').click();
  const code = (await m.page.getByTestId('confirmation-code').textContent())!.trim();
  await page.getByLabel('Code shown on the phone').fill(code);
  await page.getByRole('button', { name: 'Confirm pairing' }).click();
  await m.page.getByTestId('start-capture').click();
  await expect(m.page.getByText(/Camera permission was denied/i).first()).toBeVisible();
  await shot(m.page, 'mobile-camera-denied');
  await m.ctx.close();

  // Same pairing secret is bound to that browser; start a fresh pairing for the network test.
  await page.getByTestId('cancel-pairing').click();
  await page.getByTestId('create-pairing').click();
  const link = (await page.getByTestId('pairing-link').textContent())!.trim();
  const n = await phone(browser, { mode: 'ok', autopilot: true, scene: { identity: 5 } });
  await n.page.goto(new URL(link).pathname);
  await n.page.getByLabel(/I am Fiona P\./).check();
  await n.page.getByTestId('claim-pairing').click();
  const code2 = (await n.page.getByTestId('confirmation-code').textContent())!.trim();
  await page.getByLabel('Code shown on the phone').fill(code2);
  await page.getByRole('button', { name: 'Confirm pairing' }).click();
  await n.page.getByTestId('start-capture').click();
  await expect(n.page.getByTestId('capture-running')).toBeVisible();
  await n.ctx.setOffline(true);
  await expect(n.page.getByText(/Connection lost — retrying/)).toBeVisible({ timeout: 10_000 });
  await shot(n.page, 'mobile-offline');
  await n.ctx.setOffline(false);
  await expect(n.page.getByTestId('mobile-result')).toContainText('Verified', { timeout: 45_000 });
  await n.ctx.close();
});
