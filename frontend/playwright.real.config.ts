import { defineConfig, devices } from '@playwright/test';

/**
 * Real-model browser tests (opt-in): production build, real MediaPipe in the browser,
 * real DeepFace/MediaPipe on the server, Chromium fake camera fed from Y4M clips of
 * public LFW test images (run `python scripts/make_test_videos.py` first).
 * Screenshots mask the video element so no face image is stored in artifacts.
 */
export default defineConfig({
  testDir: './e2e/real',
  outputDir: './test-results/real',
  workers: 1,
  timeout: 180_000,
  expect: { timeout: 20_000 },
  reporter: [['list']],
  use: { baseURL: 'http://localhost:4174', permissions: ['camera'], trace: 'retain-on-failure' },
  projects: [{ name: 'chromium-real', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } }],
  webServer: [
    {
      command: 'node e2e/start-backend.mjs',
      url: 'http://127.0.0.1:8101/api/v1/health/ready',
      timeout: 300_000,
      env: { E2E_MODE: 'real' },
    },
    {
      command: 'npm run build && npx vite preview --port 4174 --strictPort --host 127.0.0.1',
      url: 'http://127.0.0.1:4174',
      timeout: 240_000,
      env: { VITE_BACKEND_URL: 'http://127.0.0.1:8101' },
    },
  ],
});
