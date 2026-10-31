import { test as base, expect, type Page, type APIRequestContext } from '@playwright/test';

export const USERS = {
  admin: { email: 'admin@e2e.local', password: 'E2e-Admin-Pass-1', name: 'Ada Admin' },
  operator: { email: 'operator@e2e.local', password: 'E2e-User-Pass-1', name: 'Olivia Operator' },
  viewer: { email: 'viewer@e2e.local', password: 'E2e-User-Pass-1', name: 'Victor Viewer' },
} as const;
export const SUBJECT_PASSWORD = 'E2e-User-Pass-1';

export interface Scene {
  faces: number;
  yaw: number;
  blink: number;
  identity: number;
  real: boolean;
  quality: number;
  widthCode: number;
}

export interface CamOptions {
  mode: 'ok' | 'denied' | 'notfound';
  scene?: Partial<Scene>;
  autopilot?: boolean;
}

/**
 * Installs a synthetic camera. Frames carry the FakeProvider marker grid
 * (8 cells × 40 px, gray = 17·v) and per-frame noise; no real face is ever drawn.
 */
export async function installFakeCamera(page: Page, opts: CamOptions) {
  await page.addInitScript((o: CamOptions) => {
    const scene: Scene = { faces: 1, yaw: 0, blink: 0, identity: 1, real: true, quality: 0.9, widthCode: 7, ...(o.scene ?? {}) };
    const w = window as unknown as Record<string, unknown>;
    w.__fakeCam = { scene, autopilot: o.autopilot ?? false };
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d')!;
    let blinkPhase = 0;

    const currentStep = (): string => {
      const li = document.querySelector('[aria-label="Liveness challenge"] [aria-current="step"]');
      return li?.textContent ?? '';
    };
    const effective = (): Scene => {
      const cam = w.__fakeCam as { scene: Scene; autopilot: boolean };
      const s = { ...cam.scene };
      if (cam.autopilot) {
        const step = currentStep();
        s.yaw = step.includes('Turn left') ? 25 : step.includes('Turn right') ? -25 : 0;
        s.blink = step.includes('Blink') ? (blinkPhase % 2 ? 1 : 0) : 0;
      }
      return s;
    };
    setInterval(() => { blinkPhase++; }, 260);

    const draw = () => {
      const s = effective();
      ctx.fillStyle = 'rgb(90,90,90)';
      ctx.fillRect(0, 0, 640, 480);
      for (let i = 0; i < 160; i++) {                        // noise → no two frames identical
        const g = 60 + Math.floor(Math.random() * 80);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(Math.random() * 640, 50 + Math.random() * 430, 18, 18);
      }
      const vals = [10, s.faces, Math.round(s.yaw / 5) + 8, Math.round(s.blink * 15), s.identity, s.real ? 12 : 2,
        Math.round(s.quality * 15), s.widthCode];
      vals.forEach((v, i) => {
        const g = Math.max(0, Math.min(15, v)) * 17;
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(i * 40, 0, 40, 40);
      });
      // A neutral silhouette (not a face) so screenshots show where the subject would be.
      if (s.faces > 0) {
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.ellipse(320 + s.yaw * 2, 250, 110, 140, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
    };
    setInterval(draw, 33);
    draw();

    const md = navigator.mediaDevices;
    md.getUserMedia = async () => {
      if (o.mode === 'denied') throw new DOMException('Permission denied', 'NotAllowedError');
      if (o.mode === 'notfound') throw new DOMException('Requested device not found', 'NotFoundError');
      return canvas.captureStream(30);
    };
    md.enumerateDevices = async () => (o.mode === 'notfound' ? [] : [
      { deviceId: 'fake-cam-1', groupId: 'g', kind: 'videoinput', label: 'Synthetic test camera', toJSON() { return {}; } } as MediaDeviceInfo,
    ]);

    // Deterministic tracker matching the scene (only honoured by the VITE_E2E_HOOKS build).
    w.__FV_E2E__ = {
      detect: () => {
        const s = effective();
        const width = Math.min(0.9, 0.08 + 0.04 * s.widthCode);
        const faces = [];
        for (let k = 0; k < s.faces; k++) {
          const cx = 0.5 + (k ? k * 0.3 : 0) - (s.faces > 1 ? 0.15 : 0);
          const box = { x: cx - width / 2, y: 0.52 - width * 0.65, w: width, h: width * 1.3 };
          faces.push({ box, yaw: s.yaw, pitch: 0, closureBoth: s.blink, closureAny: s.blink,
            points: [[cx - width * 0.2, 0.42], [cx + width * 0.2, 0.42], [cx, 0.52], [cx - width * 0.15, 0.62], [cx + width * 0.15, 0.62]] });
        }
        return faces;
      },
    };
  }, opts);
}

export async function setScene(page: Page, scene: Partial<Scene>) {
  await page.evaluate((s) => {
    const cam = (window as unknown as { __fakeCam: { scene: Record<string, unknown> } }).__fakeCam;
    Object.assign(cam.scene, s);
  }, scene);
}

export async function setAutopilot(page: Page, on: boolean) {
  await page.evaluate((v) => { (window as unknown as { __fakeCam: { autopilot: boolean } }).__fakeCam.autopilot = v; }, on);
}

export async function login(page: Page, who: { email: string; password: string }) {
  await page.goto('/login');
  await page.getByLabel('Email').fill(who.email);
  await page.getByLabel('Password').fill(who.password);
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await expect(page.getByTestId('user-menu')).toBeVisible();
}

export async function apiLogin(request: APIRequestContext, who: { email: string; password: string }): Promise<Record<string, string>> {
  const r = await request.post('/api/v1/auth/login', { data: who });
  expect(r.ok()).toBeTruthy();
  const token = (await r.json()).access_token as string;
  // The request context now also holds the auth cookies, so state-changing calls must
  // pass the double-submit CSRF check like a browser would.
  const csrf = /fv_csrf=([^;]+)/.exec(r.headersArray().filter((h) => h.name.toLowerCase() === 'set-cookie')
    .map((h) => h.value).join('\n'))?.[1] ?? '';
  return { Authorization: `Bearer ${token}`, 'X-CSRF-Token': decodeURIComponent(csrf) };
}

/** Collects console errors and failed API calls; tests assert they stay empty. */
export const test = base.extend<{ problems: string[] }>({
  problems: async ({ page }, use) => {
    const problems: string[] = [];
    page.on('console', (m) => {
      if (m.type() === 'error' && !/Failed to load resource|status of 4\d\d|net::ERR_INTERNET_DISCONNECTED|ERR_NETWORK_CHANGED/.test(m.text())) {
        problems.push(`console: ${m.text()}`);
      }
    });
    page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
    page.on('response', (r) => {
      if (r.url().includes('/api/') && r.status() >= 500) problems.push(`HTTP ${r.status()} ${r.url()}`);
    });
    await use(problems);
  },
});

export { expect };

export async function shot(page: Page, name: string) {
  await page.waitForTimeout(150);
  await page.screenshot({ path: `e2e/.artifacts/screens/${name}.png`, fullPage: true });
}
