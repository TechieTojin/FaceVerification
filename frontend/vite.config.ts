import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backend = process.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: backend, changeOrigin: false },
    },
  },
  preview: {
    port: 4173,
    proxy: { '/api': { target: backend, changeOrigin: false } },
  },
  build: {
    sourcemap: false,
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/recharts') || id.includes('node_modules/d3-')) return 'charts';
          if (id.includes('node_modules/react') || id.includes('node_modules/scheduler')) return 'react';
          return undefined;
        },
      },
    },
  },
});
