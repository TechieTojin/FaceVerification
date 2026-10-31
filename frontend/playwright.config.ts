import { defineConfig, devices } from '@playwright/test';

/**
 * Deterministic end-to-end suite.
 * Backend: isolated SQLite DB + test-only FakeProvider (APP_ENV=test).
 * Frontend: `build:e2e` (VITE_E2E_HOOKS) served by `vite preview`, proxying /api to the backend.
 * Camera: a synthetic marker stream injected per test (no real faces).
 */
export default defineConfig({
  testDir: './e2e/specs',
  outputDir: './test-results',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  use: {
    // localhost (not 127.0.0.1): WebAuthn requires a domain as relying-party id.
    baseURL: 'http://localhost:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    permissions: ['camera'],
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
  ],
  webServer: [
    {
      command: 'node e2e/start-backend.mjs',
      url: 'http://127.0.0.1:8100/api/v1/health/ready',
      reuseExistingServer: false,
      timeout: 120_000,
      stdout: 'pipe',
      env: { E2E_MODE: 'fake' },
    },
    {
      command: 'npm run build:e2e && npx vite preview --mode e2e --outDir dist-e2e --port 4173 --strictPort --host 127.0.0.1',
      url: 'http://127.0.0.1:4173',
      reuseExistingServer: false,
      timeout: 180_000,
      env: { VITE_BACKEND_URL: 'http://127.0.0.1:8100' },
    },
  ],
});
