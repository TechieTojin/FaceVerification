import { devices } from '@playwright/test';
import { expect, installFakeCamera, login, shot, test, USERS } from '../fixtures';
import { backendPort, confirmStepUp, webhookReceiver } from '../v4';
import {
  FaceVerificationClient, FaceVerificationError, constructWebhookEvent, verifyWebhookSignature,
} from '../../../sdk/typescript/src/index.ts';

const ISSUER = 'http://localhost:4173/api/v1/integrations';

test('API client + webhook via the UI, hosted verification via the SDK, signed assertion and webhook', async ({ page, browser, problems }) => {
  const hook = await webhookReceiver();
  try {
    await login(page, USERS.admin);
    await page.goto('/developers');
    await shot(page, 'developers-empty');
    await page.getByTestId('new-client').click();
    await page.getByRole('dialog').getByLabel('Name').fill('Shop backend');
    await page.getByRole('dialog').getByRole('button', { name: 'Create client' }).click();
    await confirmStepUp(page, { password: USERS.admin.password });
    const apiKey = await page.getByTestId('api-key').getByTestId('copy-value').inputValue();
    expect(apiKey).toMatch(/^fvk_[a-z0-9]+_/);
    await shot(page, 'developers-key-once');
    await page.getByRole('button', { name: 'Done' }).click();
    const clientId = page.url().split('/').pop()!;

    await page.getByTestId('add-webhook').click();
    await page.getByRole('dialog').getByLabel('URL').fill(hook.url);
    await page.getByRole('dialog').getByRole('button', { name: 'Add endpoint' }).click();
    const secret = await page.getByTestId('one-time-secret').getByTestId('copy-value').inputValue();
    await page.getByRole('button', { name: 'Done' }).click();
    await page.getByRole('button', { name: 'Send test' }).click();
    await expect(page.getByText('Test event delivered')).toBeVisible();
    expect(hook.received.length).toBe(1);
    expect(await verifyWebhookSignature(secret, hook.received[0].body, String(hook.received[0].headers['fv-signature']))).toBe(true);

    // SDK: start a hosted verification, complete it on a phone, read and verify the assertion.
    const fv = new FaceVerificationClient({ baseUrl: `http://127.0.0.1:${backendPort()}`, apiKey });
    const v = await fv.createVerification({ subjectEmail: 'hank@e2e.local', externalRef: 'order-e2e-1', idempotencyKey: 'order-e2e-1' });
    const again = await fv.createVerification({ subjectEmail: 'hank@e2e.local', externalRef: 'order-e2e-1', idempotencyKey: 'order-e2e-1' });
    expect(again.id).toBe(v.id);                                   // idempotent retry
    const phoneCtx = await browser.newContext({ ...devices['Pixel 7'], permissions: ['camera'] });
    const phone = await phoneCtx.newPage();
    await installFakeCamera(phone, { mode: 'ok', autopilot: true, scene: { identity: 4 } });
    await phone.goto(new URL(v.verification_url!).pathname);
    await phone.getByLabel(/I am Hank H\./).check();
    await phone.getByTestId('claim-pairing').click();               // hosted links need no operator code
    await phone.getByTestId('start-capture').click();
    await expect(phone.getByTestId('mobile-result')).toContainText('Verified', { timeout: 40_000 });
    await phoneCtx.close();

    let status = await fv.getVerification(v.id);
    for (let i = 0; i < 20 && status.status !== 'COMPLETED'; i++) {
      await page.waitForTimeout(500);
      status = await fv.getVerification(v.id);
    }
    expect(status.outcome).toBe('VERIFIED');
    const claims = await fv.verifyAssertion(status.assertion!, { audience: clientId, issuer: ISSUER });
    expect(claims).toMatchObject({ sub: 'order-e2e-1', outcome: 'VERIFIED', liveness: 'passed', aud: clientId });
    expect((await fv.introspectAssertion(status.assertion!)).active).toBe(true);
    expect(await fv.introspectAssertion(status.assertion!)).toMatchObject({ active: false, reason: 'already_used' });

    await expect.poll(() => hook.received.filter((r) => r.headers['fv-event-type'] === 'verification.completed').length,
      { timeout: 20_000 }).toBe(1);
    const delivery = hook.received.find((r) => r.headers['fv-event-type'] === 'verification.completed')!;
    const event = await constructWebhookEvent(secret, delivery.body, String(delivery.headers['fv-signature']));
    expect(event.data.outcome).toBe('VERIFIED');
    await expect(constructWebhookEvent(secret, delivery.body.replace('VERIFIED', 'VERIFIEd'), String(delivery.headers['fv-signature'])))
      .rejects.toThrow(FaceVerificationError);

    // Rotate (no grace) → the old key stops working, the new one works; revoking the new one stops it too.
    await page.reload();
    await expect(page.getByRole('cell', { name: /verification\.completed/ }).first()).toBeVisible();
    await shot(page, 'developer-client-detail');
    await page.getByTestId('rotate-key').click();
    await page.getByRole('dialog').getByRole('button', { name: 'Rotate' }).click();
    const newKey = await page.getByTestId('one-time-secret').getByTestId('copy-value').inputValue();
    await page.getByRole('button', { name: 'Done' }).click();
    await expect(fv.getVerification(v.id)).rejects.toMatchObject({ status: 401, code: 'INVALID_API_KEY' });
    const fv2 = new FaceVerificationClient({ baseUrl: `http://127.0.0.1:${backendPort()}`, apiKey: newKey });
    expect((await fv2.getVerification(v.id)).status).toBe('COMPLETED');
    await page.getByRole('row', { name: /Active/ }).getByRole('button', { name: 'Revoke' }).click();
    await expect(page.getByRole('row', { name: /Active/ })).toHaveCount(0);
    await expect(fv2.getVerification(v.id)).rejects.toMatchObject({ status: 401 });
  } finally {
    await hook.close();
  }
  expect(problems).toEqual([]);
});

test('API docs page renders and operators cannot reach developer settings', async ({ page }) => {
  await login(page, USERS.admin);
  await page.goto('/developers/docs');
  await expect(page.getByRole('heading', { level: 1, name: 'Integration API' })).toBeVisible();
  await shot(page, 'api-docs');
  await page.getByTestId('user-menu').click();
  await page.getByRole('menuitem', { name: 'Sign out' }).click();
  await expect(page).toHaveURL(/\/login$/);
  await login(page, USERS.operator);
  await page.goto('/developers');
  await expect(page.getByText('You don’t have access to this page')).toBeVisible();
});
