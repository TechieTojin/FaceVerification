import { spawn, type ChildProcess } from 'node:child_process';
import { createConnection, createServer, type Socket } from 'node:net';
import { rmSync } from 'node:fs';
import { join } from 'node:path';
import { apiLogin, expect, installFakeCamera, login, shot, test, USERS } from '../fixtures';
import { backendPort, confirmStepUp, PYTHON, REPO } from '../v4';

test('platform admin creates an organisation, switches to it, invites a member; data stays isolated', async ({ page, browser, request, playwright, problems }) => {
  await login(page, USERS.admin);
  await page.goto('/organization');
  await shot(page, 'organization');
  await page.getByTestId('create-org').click();
  await page.getByRole('dialog').getByLabel('Name').fill('Beta Corp');
  await page.getByRole('dialog').getByRole('button', { name: 'Create' }).click();
  await confirmStepUp(page, { password: USERS.admin.password });
  await expect(page.getByRole('row', { name: /Beta Corp/ })).toBeVisible();

  await page.getByTestId('org-switcher').click();
  await page.getByRole('option', { name: /Beta Corp/ }).click();
  await expect(page.getByTestId('org-switcher')).toContainText('Beta Corp');
  await page.goto('/users');
  await expect(page.getByRole('cell', { name: 'Ada Admin' })).toBeVisible();
  await expect(page.getByText('Alice Enrolled')).toHaveCount(0);            // default-org members are not here
  await page.goto('/history');
  await expect(page.getByText(/No sessions/i)).toBeVisible();
  await shot(page, 'org-beta-history-empty');

  await page.goto('/organization');
  await page.getByTestId('invite-member').click();
  await page.getByRole('dialog').getByLabel('Email').fill('beta.member@e2e.local');
  await page.getByRole('dialog').getByLabel('Role').selectOption('OPERATOR');
  await page.getByRole('dialog').getByRole('button', { name: 'Create invitation' }).click();
  const link = await page.getByTestId('copy-value').inputValue();
  await page.getByRole('button', { name: 'Done' }).click();

  const guest = await browser.newContext();
  const g = await guest.newPage();
  await g.goto(new URL(link).pathname);
  await expect(g.getByTestId('invite-page')).toContainText('Join Beta Corp');
  await shot(g, 'invite-accept');
  await g.getByLabel('Your full name').fill('Bea Member');
  await g.getByLabel('Choose a password').fill('Beta-Member-Pass-1');
  await g.getByRole('button', { name: 'Create account and join' }).click();
  await expect(g.getByTestId('org-switcher')).toContainText('Beta Corp');
  await guest.close();

  // Cross-tenant access through the API is refused for a Beta member (separate cookie jar per identity).
  const betaRequest = await playwright.request.newContext({ baseURL: 'http://localhost:4173' });
  const beta = await apiLogin(betaRequest, { email: 'beta.member@e2e.local', password: 'Beta-Member-Pass-1' });
  const admin = await apiLogin(request, USERS.admin);
  await request.post('/api/v1/auth/switch-org', { headers: admin, data: { organization_id: (await (await request.get('/api/v1/auth/me', { headers: admin })).json()).organizations.find((o: { name: string }) => o.name === 'Default organization').id } });
  const directory = await (await request.get('/api/v1/users/directory', { headers: admin })).json();
  const alice = directory.users.find((u: { full_name: string }) => u.full_name === 'Alice Enrolled');
  // An operator has no users:read at all (403); either way nothing about the other organisation is revealed.
  expect([403, 404]).toContain((await betaRequest.get(`/api/v1/users/${alice.id}`, { headers: beta })).status());
  expect((await betaRequest.get(`/api/v1/enrolments/${alice.id}`, { headers: beta })).status()).toBe(404);
  expect((await betaRequest.post('/api/v1/sessions', { headers: beta, data: { target_user_id: alice.id } })).status()).toBe(404);
  expect((await betaRequest.get('/api/v1/history/sessions', { headers: beta })).ok()).toBeTruthy();
  expect((await (await betaRequest.get('/api/v1/history/sessions', { headers: beta })).json()).total).toBe(0);
  await betaRequest.dispose();

  await page.getByTestId('org-switcher').click();
  await page.getByRole('option', { name: /Default organization/ }).click();
  await expect(page.getByTestId('org-switcher')).toContainText('Default organization');
  expect(problems).toEqual([]);
});

/** Run a command without blocking the event loop (the proxy below must keep serving while it runs). */
function run(cmd: string, args: string[], env: NodeJS.ProcessEnv): Promise<{ status: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { cwd: REPO, env });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('close', (status) => resolve({ status, stdout, stderr }));
  });
}

/** TCP proxy between the kiosk agent and the server; `online = false` drops every connection. */
async function toggleProxy(targetPort: number) {
  const state = { online: true };
  const sockets = new Set<Socket>();
  const server = createServer((client) => {
    if (!state.online) { client.destroy(); return; }
    const upstream = createConnection(targetPort, '127.0.0.1');
    sockets.add(client); sockets.add(upstream);
    client.pipe(upstream).pipe(client);
    const done = () => { client.destroy(); upstream.destroy(); sockets.delete(client); sockets.delete(upstream); };
    client.on('error', done); upstream.on('error', done); client.on('close', done); upstream.on('close', done);
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', () => r()));
  return {
    port: (server.address() as { port: number }).port,
    set(online: boolean) { state.online = online; if (!online) for (const s of sockets) s.destroy(); },
    close: () => new Promise<void>((r) => { for (const s of sockets) s.destroy(); server.close(() => r()); }),
  };
}

test('offline kiosk: provision, disconnect, verify locally, reconnect and sync once; revocation wipes it', async ({ page, browser, problems }) => {
  test.setTimeout(240_000);
  const dataDir = join(REPO, '.e2e', 'kiosk');
  rmSync(dataDir, { recursive: true, force: true });
  const proxy = await toggleProxy(Number(backendPort()));
  let agent: ChildProcess | null = null;
  try {
    await login(page, USERS.admin);
    await page.goto('/devices');
    await page.getByTestId('register-device').click();
    const dialog = page.getByRole('dialog');
    await dialog.getByLabel('Name').fill('Warehouse kiosk');
    await dialog.getByRole('switch', { name: /Allow offline verification/ }).click();
    await dialog.getByRole('button', { name: 'Register' }).click();
    await confirmStepUp(page, { password: USERS.admin.password });
    const code = await page.getByTestId('provisioning-code').getByTestId('copy-value').inputValue();
    await page.getByRole('button', { name: 'Done' }).click();
    await page.getByTestId('edit-offline-subjects').click();
    await page.getByRole('dialog').getByRole('option', { name: /Gina Kiosk/ }).click();
    await page.getByRole('dialog').getByRole('button', { name: 'Add', exact: true }).click();
    await page.getByTestId('save-offline-subjects').click();
    await expect(page.getByText('Offline subjects updated')).toBeVisible();

    const env = { ...process.env, KIOSK_PASSPHRASE: 'e2e-only-kiosk-passphrase' };
    const prov = await run(PYTHON, ['-m', 'kiosk', 'provision', '--data-dir', dataDir, '--server', `http://127.0.0.1:${proxy.port}`,
      '--code', code, '--inference', 'fake', '--protection', 'passphrase'], env);
    expect(prov.status, prov.stderr).toBe(0);
    expect(prov.stdout).toContain('"bundle_refreshed": true');
    agent = spawn(PYTHON, ['-m', 'kiosk', 'serve', '--data-dir', dataDir, '--inference', 'fake', '--port', '8310',
      '--ui-dir', join(REPO, 'frontend', 'dist-e2e'), '--sync-interval', '2'], { cwd: REPO, env, stdio: 'inherit' });

    const kctx = await browser.newContext({ permissions: ['camera'] });
    const k = await kctx.newPage();
    await installFakeCamera(k, { mode: 'ok', autopilot: true, scene: { identity: 8 } });
    await expect.poll(async () => (await k.goto('http://127.0.0.1:8310/kiosk').catch(() => null))?.status() ?? 0, { timeout: 60_000 }).toBe(200);
    await expect(k.getByTestId('kiosk-page')).toContainText('Connected to server', { timeout: 30_000 });
    await shot(k, 'kiosk-online');

    proxy.set(false);                                                  // network outage
    await k.getByTestId('kiosk-sync').click();
    await expect(k.getByText('Offline mode')).toBeVisible({ timeout: 15_000 });
    await k.getByLabel('Who is being verified?').selectOption({ label: 'Gina Kiosk' });
    await k.getByTestId('start-capture').click();
    await expect(k.getByTestId('session-outcome')).toHaveText('Verified', { timeout: 40_000 });
    await expect(k.getByTestId('kiosk-events')).toContainText('Pending', { timeout: 10_000 });
    await shot(k, 'kiosk-offline-verified');

    proxy.set(true);                                                    // reconnect → sync
    await k.getByTestId('kiosk-sync').click();
    await expect(k.getByTestId('kiosk-events')).toContainText('Synced', { timeout: 20_000 });
    await expect(k.getByText('Online').first()).toBeVisible();
    await k.getByTestId('kiosk-sync').click();                          // a second sync uploads nothing new
    await page.reload();
    await expect(page.getByRole('table', { name: 'Offline events' }).getByRole('row')).toHaveCount(2);   // header + 1 event
    await expect(page.getByRole('table', { name: 'Offline events' })).toContainText('Gina Kiosk');
    await shot(page, 'device-offline-log');

    await page.getByTestId('revoke-device').click();
    await page.getByRole('dialog').getByLabel(/Reason/).fill('Decommissioned');
    await page.getByRole('dialog').getByRole('button', { name: 'Revoke device' }).click();
    await k.getByTestId('kiosk-sync').click();
    await expect(k.getByText('This kiosk has been revoked')).toBeVisible({ timeout: 15_000 });
    await shot(k, 'kiosk-revoked');
    await kctx.close();
  } finally {
    agent?.kill();
    await proxy.close();
  }
  expect(problems).toEqual([]);
});
