import { createHmac } from 'node:crypto';
import { createServer, type IncomingMessage } from 'node:http';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { BrowserContext, CDPSession, Page } from '@playwright/test';
import { expect } from './fixtures';

const repo = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const PYTHON = process.env.PYTHON ?? join(repo, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
export const REPO = repo;

/** Chromium virtual authenticator (CTAP2, platform, resident keys, user verification succeeds). */
export async function addVirtualAuthenticator(page: Page): Promise<{ cdp: CDPSession; id: string }> {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('WebAuthn.enable');
  const r = await cdp.send('WebAuthn.addVirtualAuthenticator', {
    options: { protocol: 'ctap2', transport: 'internal', hasResidentKey: true, hasUserVerification: true,
      isUserVerified: true, automaticPresenceSimulation: true },
  });
  return { cdp, id: r.authenticatorId };
}

export async function credentialsOf(a: { cdp: CDPSession; id: string }) {
  return (await a.cdp.send('WebAuthn.getCredentials', { authenticatorId: a.id })).credentials;
}

/** RFC 6238 TOTP (SHA-1, 6 digits, 30 s) for a base32 secret. */
export function totp(secret: string, offsetSteps = 0, now = Date.now()): string {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
  let bits = '';
  for (const c of secret.replace(/=+$/, '').toUpperCase()) bits += alphabet.indexOf(c).toString(2).padStart(5, '0');
  const key = Buffer.from(bits.match(/.{8}/g)!.map((b) => parseInt(b, 2)));
  const counter = Math.floor(now / 1000 / 30) + offsetSteps;
  const msg = Buffer.alloc(8);
  msg.writeBigUInt64BE(BigInt(counter));
  const h = createHmac('sha1', key).update(msg).digest();
  const o = h[h.length - 1] & 0xf;
  const n = ((h[o] & 0x7f) << 24) | (h[o + 1] << 16) | (h[o + 2] << 8) | h[o + 3];
  return String(n % 1_000_000).padStart(6, '0');
}

/** Complete a standard step-up dialog with the given method if it appears. */
export async function confirmStepUp(page: Page, opts: { password?: string; code?: string; method?: 'Password' | 'Authenticator' | 'Passkey' | 'Recovery code' }) {
  const dialog = page.getByTestId('step-up-dialog');
  await expect(dialog).toBeVisible();
  if (opts.method) {
    const tab = dialog.getByRole('tab', { name: opts.method });
    if (await tab.count()) await tab.click();
  }
  if (opts.password) await dialog.getByLabel('Password').fill(opts.password);
  if (opts.code) await dialog.getByLabel(/6-digit code|Recovery code/).fill(opts.code);
  await dialog.getByTestId('step-up-confirm').click();
  await expect(dialog).toBeHidden();
}

function backendEnv(): Record<string, string> {
  return JSON.parse(readFileSync(join(repo, '.e2e', 'backend-env-fake.json'), 'utf-8'));
}

/** Test-only database operations (time travel); the helper refuses to run outside APP_ENV=test. */
export function e2eOp(...args: string[]) {
  const r = spawnSync(PYTHON, ['-m', 'backend.tests.e2e_ops', ...args], { cwd: repo, env: { ...process.env, ...backendEnv() },
    encoding: 'utf-8' });
  if (r.status !== 0) throw new Error(`e2e_ops ${args.join(' ')} failed: ${r.stderr}`);
}

export function backendPort(): string {
  return backendEnv().BACKEND_PORT ?? '8100';
}

export interface Received { headers: IncomingMessage['headers']; body: string }

/** A local webhook receiver (the E2E backend allows private webhook targets). */
export async function webhookReceiver(status = 200) {
  const received: Received[] = [];
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => { received.push({ headers: req.headers, body }); res.writeHead(status); res.end(); });
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', () => r()));
  const port = (server.address() as { port: number }).port;
  return { url: `http://127.0.0.1:${port}/hook`, received, close: () => new Promise<void>((r) => server.close(() => r())) };
}

export async function mobileContext(browser: { newContext: (o: object) => Promise<BrowserContext> }, devices: Record<string, object>) {
  return browser.newContext({ ...devices['Pixel 7'], permissions: ['camera'] });
}
