import { expect, login, shot, test, USERS } from '../fixtures';

test('login failure shows a clear error and no redirect loop', async ({ page, problems }) => {
  const navigations: string[] = [];
  page.on('framenavigated', (f) => { if (f === page.mainFrame()) navigations.push(f.url()); });
  await page.goto('/');
  await expect(page).toHaveURL(/\/login$/);
  await page.waitForTimeout(1500);
  expect(navigations.length).toBeLessThan(4);          // regression: v2.1 reloaded /login forever
  await page.getByLabel('Email').fill(USERS.admin.email);
  await page.getByLabel('Password').fill('wrong-password');
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await expect(page.getByText('Incorrect email or password.')).toBeVisible();
  await shot(page, 'login-error');
  expect(problems).toEqual([]);
});

test('login, session restore on reload, logout', async ({ page, problems, context }) => {
  await login(page, USERS.admin);
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  const storage = await page.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }));
  expect(storage).not.toMatch(/eyJ|token/i);              // no tokens in web storage
  const cookies = await context.cookies();
  expect(cookies.find((c) => c.name === 'fv_access')?.httpOnly).toBe(true);
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await page.getByTestId('user-menu').click();
  await page.getByRole('menuitem', { name: 'Sign out' }).click();
  await expect(page).toHaveURL(/\/login$/);
  await page.goto('/users');
  await expect(page).toHaveURL(/\/login$/);
  expect(problems).toEqual([]);
});

test('admin navigation reaches every module', async ({ page, problems }) => {
  await login(page, USERS.admin);
  const nav = page.getByRole('navigation', { name: 'Main' });
  for (const [link, heading] of [
    ['Live verification', 'Live verification'], ['QR verification', 'QR verification'], ['Enrolment', 'Enrolment'],
    ['Document check', 'Document check'], ['History', 'Verification history'], ['Review queue', 'Review queue'],
    ['Enrolment approvals', 'Enrolment approvals'], ['Alerts', 'Security alerts'], ['Security log', 'Security log'],
    ['Users', 'Users'], ['Organisation', 'Organisation'], ['Cameras', 'Cameras'], ['Devices & kiosks', 'Devices'],
    ['Developers', 'API clients'], ['Analytics', 'Analytics'], ['Model performance', 'Model performance'],
    ['System health', 'System health'], ['Policy settings', 'Policy settings'],
  ] as const) {
    await nav.getByRole('link', { name: link, exact: true }).click();
    await expect(page.getByRole('heading', { level: 1, name: heading })).toBeVisible();
  }
  expect(problems).toEqual([]);
});

test('operator sees only operator modules and is blocked from admin pages', async ({ page, problems }) => {
  await login(page, USERS.operator);
  const nav = page.getByRole('navigation', { name: 'Main' });
  await expect(nav.getByRole('link', { name: 'Live verification' })).toBeVisible();
  await expect(nav.getByRole('link', { name: 'Enrolment', exact: true })).toBeVisible();
  for (const hidden of ['Users', 'Alerts', 'Security log', 'System health', 'Analytics', 'Review queue', 'Developers']) {
    await expect(nav.getByRole('link', { name: hidden, exact: true })).toHaveCount(0);
  }
  await page.goto('/users');
  await expect(page.getByText('You don’t have access to this page')).toBeVisible();
  await shot(page, 'operator-forbidden');
  expect(problems).toEqual([]);
});

test('viewer has read-only navigation', async ({ page, problems }) => {
  await login(page, USERS.viewer);
  const nav = page.getByRole('navigation', { name: 'Main' });
  await expect(nav.getByRole('link', { name: 'Analytics' })).toBeVisible();
  await expect(nav.getByRole('link', { name: 'History' })).toBeVisible();
  for (const hidden of ['Live verification', 'Enrolment', 'Users', 'Policy settings', 'Document check']) {
    await expect(nav.getByRole('link', { name: hidden, exact: true })).toHaveCount(0);
  }
  await page.goto('/verify');
  await expect(page.getByText('You don’t have access to this page')).toBeVisible();
  expect(problems).toEqual([]);
});
