import type { Page } from '@playwright/test';
import { expect, installFakeCamera, login, shot, SUBJECT_PASSWORD, test, USERS } from '../fixtures';

async function pickAndConsent(page: Page, name: string, password = SUBJECT_PASSWORD) {
  await page.goto('/enrol');
  await page.getByRole('option', { name: new RegExp(name) }).click();
  await page.getByRole('button', { name: 'Continue' }).click();
  await page.getByLabel(/have read and accept/).check();
  await page.getByLabel(/password \(entered by the subject\)/).fill(password);
  await page.getByRole('button', { name: 'Record consent' }).click();
}

test('operator enrols a new person with subject consent and liveness capture', async ({ page, problems }) => {
  await installFakeCamera(page, { mode: 'ok', autopilot: true, scene: { identity: 7 } });
  await login(page, USERS.operator);
  await page.goto('/enrol');
  await page.getByRole('option', { name: /Carol New/ }).click();
  await expect(page.getByText('Authorisation check')).toBeVisible();
  await shot(page, 'enrol-select');
  await page.getByRole('button', { name: 'Continue' }).click();
  await expect(page.getByRole('heading', { name: 'Informed consent' })).toBeVisible();
  await shot(page, 'enrol-consent');
  await page.getByLabel(/have read and accept/).check();
  await page.getByLabel(/password \(entered by the subject\)/).fill(SUBJECT_PASSWORD);
  await page.getByRole('button', { name: 'Record consent' }).click();
  await page.getByTestId('start-capture').click();
  await expect(page.getByTestId('capture-running')).toBeVisible();
  await shot(page, 'enrol-capture');
  await expect(page.getByTestId('enrol-result')).toContainText('Enrolment complete', { timeout: 40_000 });
  await shot(page, 'enrol-complete');
  await page.getByRole('button', { name: 'View enrolment' }).click();
  await expect(page.getByText('Usable for verification')).toBeVisible();
  await expect(page.getByText('Active', { exact: true })).toBeVisible();
  expect(problems).toEqual([]);
});

test('wrong subject password blocks consent', async ({ page, problems }) => {
  await login(page, USERS.operator);
  await pickAndConsent(page, 'Erin New', 'not-the-password');
  await expect(page.getByText('The subject\'s password was not accepted')).toBeVisible();
  expect(problems).toEqual([]);
});

test('operator cannot enrol an administrator account', async ({ page, problems }) => {
  await login(page, USERS.operator);
  await page.goto('/enrol');
  await page.getByRole('option', { name: /Ada Admin/ }).click();
  await expect(page.getByText('Only administrators may enrol administrator accounts.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Continue' })).toBeDisabled();
  expect(problems).toEqual([]);
});

test('template replacement requires admin approval, then activates', async ({ browser, problems, page }) => {
  await installFakeCamera(page, { mode: 'ok', autopilot: true, scene: { identity: 8 } });
  await login(page, USERS.operator);
  await page.goto('/enrol');
  await page.getByRole('option', { name: /Dave Replace/ }).click();
  await expect(page.getByText('Replacement needs approval')).toBeVisible();
  await page.getByRole('button', { name: 'Continue' }).click();
  await page.getByLabel(/have read and accept/).check();
  await page.getByLabel(/password \(entered by the subject\)/).fill(SUBJECT_PASSWORD);
  await page.getByRole('button', { name: 'Record consent' }).click();
  await page.getByTestId('start-capture').click();
  await expect(page.getByTestId('enrol-result')).toContainText('Awaiting administrator approval', { timeout: 40_000 });
  await expect(page.getByTestId('enrol-result')).toContainText('Replaces an existing template');
  await shot(page, 'enrol-pending');

  const adminCtx = await browser.newContext();
  const admin = await adminCtx.newPage();
  await login(admin, USERS.admin);
  await admin.goto('/approvals');
  const row = admin.getByRole('row', { name: /Dave Replace/ });
  await expect(row).toBeVisible();
  await shot(admin, 'approvals-pending');
  await row.getByRole('button', { name: 'Approve' }).click();
  await admin.getByLabel('Review note (optional)').fill('Identity confirmed in person');
  await admin.getByRole('button', { name: 'Approve and activate' }).click();
  await expect(admin.getByText('Request approved — template is now active')).toBeVisible();
  await admin.getByRole('tab', { name: 'Approved', exact: true }).click();
  await expect(admin.getByRole('row', { name: /Dave Replace/ })).toBeVisible();
  await adminCtx.close();
  expect(problems).toEqual([]);
});

test('admin revokes an enrolment with typed confirmation', async ({ page, problems }) => {
  await login(page, USERS.admin);
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Users', exact: true }).click();
  await page.getByLabel('Search users').fill('Bob');
  await page.getByRole('row', { name: /Bob Enrolled/ }).click();
  await page.getByRole('link', { name: 'Enrolment record' }).click();
  await page.getByRole('button', { name: 'Revoke' }).click();
  await page.getByLabel('Reason').fill('Left the organisation');
  const confirm = page.getByRole('button', { name: 'Revoke template' });
  await expect(confirm).toBeDisabled();
  await page.getByLabel(/to confirm/).fill('DELETE');
  await shot(page, 'enrolment-revoke-dialog');
  await confirm.click();
  await expect(page.getByText('Enrolment revoked')).toBeVisible();
  await expect(page.getByText('Revoked', { exact: true })).toBeVisible();
  expect(problems).toEqual([]);
});
