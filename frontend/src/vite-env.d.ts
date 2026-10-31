/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "true" only in the isolated e2e build: allows Playwright to inject a deterministic face tracker. */
  readonly VITE_E2E_HOOKS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
