import type { Page } from '@playwright/test';
import { apiLogin, expect, installFakeCamera, login, test, USERS } from '../fixtures';
import { e2eOp } from '../v4';

async function noHorizontalOverflow(page: Page) {
  const { scroll, client } = await page.evaluate(() => ({
    scroll: Math.max(document.documentElement.scrollWidth, document.querySelector('main')?.scrollWidth ?? 0),
    client: Math.max(document.documentElement.clientWidth, document.querySelector('main')?.clientWidth ?? 0),
  }));
  expect(scroll, 'page must not scroll horizontally').toBeLessThanOrEqual(client + 1);
}

const shotTo = (page: Page, name: string) => page.screenshot({ path: `e2e/.artifacts/screens/v4/${name}.png`, fullPage: true });

test.beforeAll(async ({ request }) => {
  const admin = await apiLogin(request, USERS.admin);
  // Large table + long names (synthetic accounts only).
  for (let i = 0; i < 35; i++) {
    await request.post('/api/v1/users', { headers: admin, data: {
      email: `bulk${i}@e2e.local`, password: 'Bulk-User-Pass-1', role: 'VIEWER',
      full_name: i === 0 ? 'Maximiliana Konstantinopoulou-Vanderbilt-Featherstonehaugh de la Cruz' : `Bulk Person ${i}`,
    } });
  }
  e2eOp('notify-many', 'admin@e2e.local', '40');
});

for (const vp of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 390, height: 844 }]) {
  test(`stress states render cleanly (${vp.name})`, async ({ page, problems }) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await login(page, USERS.admin);

    await page.goto('/users');
    await expect(page.getByRole('row')).toHaveCount(21);                     // header + a full page of 20
    await noHorizontalOverflow(page);
    await shotTo(page, `${vp.name}-users-large-table`);
    await page.getByLabel('Search users').fill('Konstantinopoulou');
    await expect(page.getByText(/Konstantinopoulou/)).toBeVisible();
    await noHorizontalOverflow(page);
    await shotTo(page, `${vp.name}-users-large-long-names`);

    await page.goto('/notifications');
    await expect(page.getByTestId('notification-item').first()).toBeVisible();
    await noHorizontalOverflow(page);
    await shotTo(page, `${vp.name}-notifications-long-list`);
    await page.getByTestId('notification-bell').click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toBeVisible();
    await noHorizontalOverflow(page);
    await shotTo(page, `${vp.name}-notification-drawer`);
    await page.keyboard.press('Escape');

    for (const [name, path] of [['reviews', '/reviews'], ['devices', '/devices'], ['cameras', '/cameras'],
      ['developers', '/developers'], ['organization', '/organization'], ['account-security', '/account/security']] as const) {
      await page.goto(path);
      await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
      await expect(page.getByRole('status', { name: /Loading/ })).toHaveCount(0, { timeout: 10_000 });
      await noHorizontalOverflow(page);
      await shotTo(page, `${vp.name}-${name}`);
    }
    expect(problems).toEqual([]);
  });
}

test('dark theme: new v4 screens', async ({ page, problems }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await installFakeCamera(page, { mode: 'ok' });
  await login(page, USERS.admin);
  for (const [name, path] of [['qr-verify', '/verify/qr'], ['reviews', '/reviews'], ['devices', '/devices'],
    ['notifications', '/notifications'], ['account-security', '/account/security'], ['developers', '/developers']] as const) {
    await page.goto(path);
    await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
    await expect(page.getByRole('status', { name: /Loading/ })).toHaveCount(0, { timeout: 10_000 });
    await shotTo(page, `dark-${name}`);
  }
  await page.goto('/login');
  expect(problems).toEqual([]);
});

test('signed-out pages: sign-in options, invalid pairing link, kiosk without agent', async ({ page }) => {
  await page.goto('/login');
  await expect(page.getByTestId('passkey-signin')).toBeVisible();
  await shotTo(page, 'login-v4');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/m/not-a-valid-token');
  await expect(page.getByText('This verification link is not valid')).toBeVisible();
  await shotTo(page, 'mobile-invalid-link');
});
