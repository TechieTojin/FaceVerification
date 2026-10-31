import { expect, login, shot, test } from '../fixtures';
import { addVirtualAuthenticator, confirmStepUp, credentialsOf, totp } from '../v4';

const PAT = { email: 'pat@e2e.local', password: 'E2e-User-Pass-1' };
const TOM = { email: 'tom@e2e.local', password: 'E2e-User-Pass-1' };

async function signOut(page: import('@playwright/test').Page) {
  await page.getByTestId('user-menu').click();
  await page.getByRole('menuitem', { name: 'Sign out' }).click();
  await expect(page).toHaveURL(/\/login$/);
}

test('register a passkey, sign in passwordless, then revoke it', async ({ page, problems }) => {
  const auth = await addVirtualAuthenticator(page);
  await login(page, PAT);
  await page.goto('/account/security');
  await expect(page.getByRole('heading', { level: 1, name: 'Account security' })).toBeVisible();
  await shot(page, 'account-security-empty');
  await page.getByTestId('add-passkey').click();
  await page.getByLabel('Name').fill('E2E laptop');
  await page.getByTestId('create-passkey').click();
  await confirmStepUp(page, { password: PAT.password });          // adding a factor needs step-up
  await expect(page.getByTestId('passkey-row')).toContainText('E2E laptop');
  expect(await credentialsOf(auth)).toHaveLength(1);
  await shot(page, 'account-security-passkey');

  await signOut(page);
  await page.getByTestId('passkey-signin').click();
  await expect(page.getByTestId('user-menu')).toContainText('Pat Passkey');      // signed in without a password

  // Password sign-in now asks for the second factor.
  await signOut(page);
  await page.getByLabel('Email').fill(PAT.email);
  await page.getByLabel('Password').fill(PAT.password);
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Two-step verification' })).toBeVisible();
  await shot(page, 'login-mfa-passkey');
  await page.getByRole('button', { name: 'Use passkey' }).click();
  await expect(page.getByTestId('user-menu')).toBeVisible();

  // Revoke: the fresh passkey sign-in counts as step-up.
  await page.goto('/account/security');
  await page.getByRole('button', { name: 'Remove E2E laptop' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Remove' }).click();
  const stepUp = page.getByTestId('step-up-dialog');
  if (await stepUp.isVisible().catch(() => false)) await confirmStepUp(page, { method: 'Passkey' });
  await expect(page.getByTestId('passkey-row')).toHaveCount(0);
  await signOut(page);
  // The authenticator still holds the credential, but the server no longer accepts it.
  await page.getByTestId('passkey-signin').click();
  await expect(page.getByText('This passkey is not registered or has been removed')).toBeVisible();
  await shot(page, 'login-passkey-revoked');
  expect(problems).toEqual([]);
});

test('passkey endpoints reject invalid challenges', async ({ request }) => {
  const r = await request.post('/api/v1/auth/passkey/verify', {
    data: { challenge_id: '00000000-0000-0000-0000-000000000000', credential: { id: 'x', rawId: 'x', type: 'public-key', response: {} } },
  });
  expect(r.status()).toBe(400);
  expect((await r.json()).code).toBe('INVALID_CHALLENGE');
  const failed = await request.post('/api/v1/auth/login/face/session', { data: { pending_token: 'not-a-real-pending-token' } });
  expect(failed.status()).toBe(401);           // a face check can never start a sign-in on its own
});

test('authenticator app: enrol, sign in, reject wrong and reused codes, recovery code, remove', async ({ page, problems }) => {
  await login(page, TOM);
  await page.goto('/account/security');
  await page.getByTestId('setup-totp').click();
  await confirmStepUp(page, { password: TOM.password });
  const secret = (await page.getByTestId('totp-secret').textContent())!.trim();
  await shot(page, 'totp-setup');
  await page.getByLabel('Enter the 6-digit code from the app').fill(totp(secret));
  await page.getByRole('button', { name: 'Enable' }).click();
  await expect(page.getByTestId('recovery-code-list').locator('li')).toHaveCount(10);
  const codes = await page.getByTestId('recovery-code-list').locator('li').allTextContents();
  await shot(page, 'recovery-codes');
  await page.getByRole('button', { name: 'I have saved them' }).click();
  await signOut(page);

  await page.getByLabel('Email').fill(TOM.email);
  await page.getByLabel('Password').fill(TOM.password);
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await page.getByLabel('6-digit code').fill('000000');
  await page.getByRole('button', { name: 'Verify' }).click();
  await expect(page.getByText('Verification failed. Check the code and try again.')).toBeVisible();
  const code = totp(secret, 1);                    // next time-step: newer than the one used at enrolment
  await page.getByLabel('6-digit code').fill(code);
  await page.getByRole('button', { name: 'Verify' }).click();
  await expect(page.getByTestId('user-menu')).toBeVisible();
  await signOut(page);

  // The same code is refused on the next sign-in (replay); a recovery code works once.
  await page.getByLabel('Email').fill(TOM.email);
  await page.getByLabel('Password').fill(TOM.password);
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await page.getByLabel('6-digit code').fill(code);
  await page.getByRole('button', { name: 'Verify' }).click();
  await expect(page.getByText('Verification failed. Check the code and try again.')).toBeVisible();
  await page.getByRole('tab', { name: 'Recovery code' }).click();
  await page.getByLabel('Recovery code').fill(codes[0]);
  await page.getByRole('button', { name: 'Verify' }).click();
  await expect(page.getByTestId('user-menu')).toBeVisible();

  // Removing the authenticator needs a recent strong factor. Signing in with a recovery code a moment ago counts;
  // if that window has passed the step-up dialog asks for another recovery code.
  await page.goto('/account/security');
  await page.getByRole('button', { name: 'Remove' }).first().click();
  await page.getByRole('dialog').getByRole('button', { name: 'Remove' }).click();
  if (await page.getByTestId('step-up-dialog').isVisible().catch(() => false)) {
    await confirmStepUp(page, { method: 'Recovery code', code: codes[1] });
  }
  await expect(page.getByText('Two-step verification is off')).toBeVisible();
  expect(problems).toEqual([]);
});
