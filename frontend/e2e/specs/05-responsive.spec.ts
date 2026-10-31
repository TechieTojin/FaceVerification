import type { Page } from '@playwright/test';
import { expect, installFakeCamera, login, test, USERS } from '../fixtures';

const WIDTHS = [
  { name: 'desktop-1440', width: 1440, height: 900 },
  { name: 'laptop-1280', width: 1280, height: 800 },
  { name: 'tablet-768', width: 768, height: 1024 },
  { name: 'mobile-390', width: 390, height: 844 },
];

const PAGES: [string, string][] = [
  ['dashboard', '/'], ['verify', '/verify'], ['enrol', '/enrol'], ['documents', '/documents'], ['history', '/history'],
  ['users', '/users'], ['approvals', '/approvals'], ['alerts', '/alerts'], ['security', '/security'],
  ['analytics', '/analytics'], ['models', '/models'], ['system', '/system'], ['settings', '/settings'], ['profile', '/profile'],
  ['qr-verify', '/verify/qr'], ['reviews', '/reviews'], ['organization', '/organization'], ['cameras', '/cameras'],
  ['devices', '/devices'], ['developers', '/developers'], ['api-docs', '/developers/docs'], ['notifications', '/notifications'],
  ['account-security', '/account/security'],
];

async function noHorizontalOverflow(page: Page) {
  const { scroll, client } = await page.evaluate(() => {
    const main = document.querySelector('main');
    return {
      scroll: Math.max(document.documentElement.scrollWidth, main?.scrollWidth ?? 0),
      client: Math.max(document.documentElement.clientWidth, main?.clientWidth ?? 0),
    };
  });
  expect(scroll, 'page must not scroll horizontally').toBeLessThanOrEqual(client + 1);
}

for (const vp of WIDTHS) {
  test(`all screens render without overflow at ${vp.name}`, async ({ page, problems }) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await installFakeCamera(page, { mode: 'ok', scene: { identity: 1 } });
    await login(page, USERS.admin);
    for (const [name, path] of PAGES) {
      await page.goto(path);
      await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
      await page.waitForLoadState('networkidle').catch(() => undefined);
      await expect(page.getByRole('status', { name: /Loading/ })).toHaveCount(0, { timeout: 10_000 });
      await noHorizontalOverflow(page);
      await page.screenshot({ path: `e2e/.artifacts/screens/responsive/${vp.name}/${name}.png`, fullPage: true });
    }
    if (vp.width < 1024) {
      await page.getByRole('button', { name: 'Open navigation' }).click();
      await expect(page.getByRole('dialog', { name: 'Navigation' })).toBeVisible();
      await page.screenshot({ path: `e2e/.artifacts/screens/responsive/${vp.name}/nav-drawer.png` });
      await page.getByRole('dialog', { name: 'Navigation' }).getByRole('link', { name: 'History' }).click();
      await expect(page.getByRole('heading', { name: 'Verification history' })).toBeVisible();
    }
    expect(problems).toEqual([]);
  });
}

test('mobile live verification is usable with the camera running', async ({ page, problems }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installFakeCamera(page, { mode: 'ok', scene: { identity: 1 }, autopilot: false });
  await login(page, USERS.operator);
  await page.goto('/verify');
  await page.getByRole('option', { name: /Alice Enrolled/ }).click();
  await page.getByTestId('start-capture').click();
  await expect(page.getByTestId('capture-running')).toBeVisible();
  await noHorizontalOverflow(page);
  await page.waitForTimeout(600);
  await page.screenshot({ path: 'e2e/.artifacts/screens/responsive/mobile-390/verify-running.png', fullPage: true });
  await page.getByTestId('cancel-capture').click();
  expect(problems).toEqual([]);
});

test('dark theme renders dashboard and verification', async ({ page, problems }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await installFakeCamera(page, { mode: 'ok', scene: { identity: 1 } });
  await login(page, USERS.admin);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await page.screenshot({ path: 'e2e/.artifacts/screens/dark/dashboard.png', fullPage: true });
  await page.goto('/analytics');
  await expect(page.getByText('Sessions per day')).toBeVisible();
  await page.screenshot({ path: 'e2e/.artifacts/screens/dark/analytics.png', fullPage: true });
  await page.goto('/verify');
  await page.getByRole('option', { name: /Alice Enrolled/ }).click();
  await page.getByTestId('start-capture').click();
  await expect(page.getByTestId('capture-running')).toBeVisible();
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'e2e/.artifacts/screens/dark/verify.png', fullPage: true });
  await page.getByTestId('cancel-capture').click();
  expect(problems).toEqual([]);
});
