import type { Page } from '@playwright/test';
import { apiLogin, expect, installFakeCamera, login, shot, test, USERS } from '../fixtures';

/** Synthetic "document" image carrying the FakeProvider marker grid (no real face or document). */
async function markerJpeg(page: Page, opts: { identity: number; faces?: number; widthCode?: number }): Promise<Buffer> {
  const b64 = await page.evaluate(async (o) => {
    const c = document.createElement('canvas');
    c.width = 1000;
    c.height = 700;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = 'rgb(150,150,150)';
    ctx.fillRect(0, 0, 1000, 700);
    for (let i = 0; i < 900; i++) {
      const g = 40 + Math.floor(Math.random() * 160);
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(Math.random() * 1000, 50 + Math.random() * 650, 12, 12);
    }
    [10, o.faces ?? 1, 8, 0, o.identity, 12, 14, o.widthCode ?? 4].forEach((v, i) => {
      const g = v * 17;
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(i * 40, 0, 40, 40);
    });
    const blob: Blob = await new Promise((r) => c.toBlob((b) => r(b!), 'image/jpeg', 0.92));
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = '';
    buf.forEach((x) => { s += String.fromCharCode(x); });
    return btoa(s);
  }, opts);
  return Buffer.from(b64, 'base64');
}

async function runVerification(page: Page, identity: number, expected: string) {
  await installFakeCamera(page, { mode: 'ok', autopilot: true, scene: { identity } });
  await login(page, USERS.operator);
  await page.goto('/verify');
  await page.getByRole('option', { name: /Alice Enrolled/ }).click();
  await page.getByTestId('start-capture').click();
  await expect(page.getByTestId('session-outcome')).toHaveText(expected, { timeout: 40_000 });
}

test.beforeAll(async ({ browser }) => {
  // This spec creates its own history so it does not depend on other spec files.
  for (const [identity, expected] of [[1, 'Verified'], [2, 'Not matched']] as const) {
    const ctx = await browser.newContext({ permissions: ['camera'] });
    await runVerification(await ctx.newPage(), identity, expected);
    await ctx.close();
  }
});

test('admin creates, edits, disables and re-enables a user', async ({ page, problems }) => {
  await login(page, USERS.admin);
  await page.goto('/users');
  await page.getByRole('button', { name: 'New user' }).click();
  await page.getByRole('button', { name: 'Create user' }).click();
  await expect(page.getByText('Enter a valid email address')).toBeVisible();
  await shot(page, 'users-create-validation');
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('Full name').fill('Frank Fresh');
  await dialog.getByLabel('Email').fill('frank@e2e.local');
  await dialog.getByLabel('Initial password').fill('Fr3sh-Start-Pass!');
  await dialog.getByLabel('Role').selectOption('VIEWER');
  await page.getByRole('button', { name: 'Create user' }).click();
  await expect(page.getByRole('heading', { name: 'Frank Fresh' })).toBeVisible();
  await page.getByRole('button', { name: 'Edit' }).click();
  await page.getByRole('dialog').getByLabel('Role').selectOption('OPERATOR');
  await page.getByRole('button', { name: 'Save changes' }).click();
  await expect(page.getByRole('region', { name: 'Notifications' }).getByText('User updated')).toBeVisible();
  await page.getByRole('button', { name: 'Disable' }).click();
  await page.getByRole('button', { name: 'Disable account' }).click();
  await expect(page.getByText(/Disabled/).first()).toBeVisible();
  await page.getByRole('button', { name: 'Enable' }).click();
  await page.getByRole('button', { name: 'Enable account' }).click();
  await expect(page.getByText('Account enabled')).toBeVisible();
  await expect(page.getByRole('cell', { name: 'User enabled' }).or(page.getByText('User enabled')).first()).toBeVisible();
  await shot(page, 'user-detail');
  await page.goto('/users');
  await page.getByLabel('Role').selectOption('OPERATOR');
  await expect(page.getByRole('row', { name: /Frank Fresh/ })).toBeVisible();
  await shot(page, 'users-list');
  expect(problems).toEqual([]);
});

test('history filters and CSV export', async ({ page, problems }) => {
  await login(page, USERS.admin);
  await page.goto('/history');
  await expect(page.getByRole('table', { name: 'Verification sessions' })).toBeVisible();
  await page.getByLabel('Outcome').selectOption('VERIFIED');
  await expect(page.getByRole('table', { name: 'Verification sessions' }).getByText('Verified').first()).toBeVisible();
  await expect(page.getByRole('table', { name: 'Verification sessions' }).getByText('Not matched')).toHaveCount(0);
  await shot(page, 'history-filtered');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export CSV' }).click();
  const file = await download;
  const path = await file.path();
  const fs = await import('node:fs');
  const csv = fs.readFileSync(path!, 'utf8');
  expect(csv.split('\n')[0]).toContain('session_id');
  expect(csv).toContain('VERIFIED');
  expect(csv).not.toContain('NOT_MATCHED');
  await page.getByRole('table', { name: 'Verification sessions' }).getByRole('row').nth(1).click();
  await expect(page.getByRole('heading', { name: /session/ })).toBeVisible();
  await expect(page.getByText('Frame timeline')).toBeVisible();
  await shot(page, 'session-detail');
  expect(problems).toEqual([]);
});

test('suspicious activity raises an alert that admins acknowledge and resolve', async ({ page, request, problems }) => {
  const op = await apiLogin(request, USERS.operator);
  const me = await (await request.get('/api/v1/auth/me', { headers: op })).json();
  const admins = await request.get('/api/v1/users/directory?search=Ada', { headers: op });
  const adminId = (await admins.json()).users[0].id;
  const r = await request.post(`/api/v1/enrolments/${adminId}/sessions`, { headers: op });
  expect(r.status()).toBe(403);
  await login(page, USERS.admin);
  await page.goto('/alerts');
  const row = page.getByTestId('alert-row').filter({ hasText: 'Unauthorised enrolment attempt' });
  await expect(row).toBeVisible();
  await expect(row).toContainText(me.email);
  await shot(page, 'alerts-open');
  await row.getByRole('button', { name: 'Acknowledge' }).click();
  await page.getByRole('button', { name: 'Acknowledge' }).last().click();
  await expect(page.getByText('Alert acknowledged')).toBeVisible();
  await row.getByRole('button', { name: 'Resolve' }).click();
  await expect(page.getByRole('dialog').getByRole('button', { name: 'Resolve' })).toBeDisabled();
  await page.getByLabel('Resolution note').fill('Operator was testing; reminded of policy');
  await page.getByRole('dialog').getByRole('button', { name: 'Resolve' }).click();
  await expect(page.getByText('Alert resolved')).toBeVisible();
  await page.getByRole('tab', { name: /Resolved/ }).click();
  await expect(page.getByTestId('alert-row').filter({ hasText: 'reminded of policy' })).toBeVisible();
  await page.goto('/security');
  await expect(page.getByRole('table', { name: 'Security events' }).getByText('Unauthorized enrolment attempt').first()).toBeVisible();
  await shot(page, 'security-log');
  expect(problems).toEqual([]);
});

test('policy change: weakening requires confirmation and creates a version', async ({ page, request, problems }) => {
  await login(page, USERS.admin);
  await page.goto('/settings');
  await expect(page.getByText(/Active policy/)).toBeVisible();
  await shot(page, 'settings');
  const input = page.getByLabel(/Maximum attempts per session/);
  await input.fill('99');
  await expect(page.getByText('Maximum 10')).toBeVisible();
  await expect(page.getByTestId('save-policy')).toBeDisabled();
  await input.fill('5');
  await expect(page.getByText('Some changes weaken security')).toBeVisible();
  await page.getByTestId('save-policy').click();
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByText('Confirm security-weakening changes')).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Confirm and save' })).toBeDisabled();
  await dialog.getByLabel(/Reason/).fill('Pilot with elderly users');
  await shot(page, 'settings-confirm');
  await dialog.getByRole('button', { name: 'Confirm and save' }).click();
  await expect(page.getByText(/Policy saved as version \d+/)).toBeVisible();
  const admin = await apiLogin(request, USERS.admin);
  const pol = await (await request.get('/api/v1/policy', { headers: admin })).json();
  expect(pol.values.max_attempts).toBe(5);
  const restore = await request.put('/api/v1/policy', { headers: admin, data: { changes: { max_attempts: 3 } } });
  expect(restore.status()).toBe(200);
  expect(problems).toEqual([]);
});

test('document check: portrait match is reported separately from authenticity', async ({ page, problems }) => {
  await installFakeCamera(page, { mode: 'ok', autopilot: true, scene: { identity: 11 } });
  await login(page, USERS.operator);
  await page.goto('/documents');
  await page.getByRole('button', { name: 'New check' }).click();
  await expect(page.getByText('Document authenticity will not be verified')).toBeVisible();
  await page.getByLabel('Subject reference').fill('Visitor 4711');
  await page.getByLabel('Document type').selectOption('PASSPORT');
  await page.getByRole('button', { name: 'Create check' }).click();
  await expect(page.getByText('The subject must consent')).toBeVisible();
  await page.getByLabel(/consents to their document portrait/).check();
  await page.getByRole('button', { name: 'Create check' }).click();
  await expect(page.getByRole('heading', { name: 'Visitor 4711' })).toBeVisible();
  await page.locator('#doc-file').setInputFiles({ name: 'blank.jpg', mimeType: 'image/jpeg', buffer: await markerJpeg(page, { identity: 11, faces: 0 }) });
  await expect(page.getByText('No portrait face was found')).toBeVisible();
  await page.locator('#doc-file').setInputFiles({ name: 'doc.jpg', mimeType: 'image/jpeg', buffer: await markerJpeg(page, { identity: 11 }) });
  await expect(page.getByText('2. Live capture')).toBeVisible();
  await page.getByTestId('start-capture').click();
  const results = page.getByTestId('document-results');
  await expect(results).toContainText('Match', { timeout: 40_000 });   // page switches to the completed record
  await expect(page.getByRole('definition').filter({ hasText: /^Completed$/ })).toBeVisible();
  await expect(results).toContainText('Not performed');
  await expect(results).toContainText('Face matches — document not authenticated');
  await expect(page.getByText(/Purged/)).toBeVisible();
  await shot(page, 'document-result');
  expect(problems).toEqual([]);
});

test('analytics, model performance and system health render real data', async ({ page, problems }) => {
  await login(page, USERS.admin);
  await page.goto('/analytics');
  await expect(page.getByText('Sessions per day')).toBeVisible();
  await expect(page.locator('.recharts-surface').first()).toBeVisible();
  await shot(page, 'analytics');
  await page.getByRole('button', { name: 'Table' }).first().click();
  await expect(page.getByRole('cell', { name: 'Verified', exact: true }).or(page.locator('th', { hasText: 'Verified' })).first()).toBeVisible();
  await page.goto('/models');
  await expect(page.getByRole('table', { name: 'Models' })).toContainText('FakeProvider');
  await shot(page, 'models');
  await page.goto('/system');
  await expect(page.getByText('All readiness checks pass')).toBeVisible();
  await shot(page, 'system-health');
  await page.goto('/');
  await expect(page.getByTestId('health-panel')).toContainText('Reachable');
  await expect(page.getByTestId('live-indicator')).toContainText('Live');
  await shot(page, 'dashboard');
  expect(problems).toEqual([]);
});

test('profile: change password signs out, new password works', async ({ page, problems }) => {
  const erin = { email: 'erin@e2e.local', password: 'E2e-User-Pass-1' };
  await login(page, erin);
  await page.goto('/profile');
  await shot(page, 'profile');
  await page.getByLabel('Current password').fill(erin.password);
  await page.getByLabel('New password', { exact: true }).fill('Changed-Pass-2026!');
  await page.getByLabel('Confirm new password').fill('Changed-Pass-2026!');
  await page.getByRole('button', { name: 'Change password' }).click();
  await expect(page).toHaveURL(/\/login$/);
  await login(page, { email: erin.email, password: 'Changed-Pass-2026!' });
  expect(problems).toEqual([]);
});
