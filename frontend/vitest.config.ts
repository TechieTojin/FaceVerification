import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Vitest bundles its own Vite; use esbuild's automatic JSX runtime (no `import React` needed).
  esbuild: { jsx: 'automatic' },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
    restoreMocks: true,
  },
});
