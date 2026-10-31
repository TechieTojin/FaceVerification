// Copies the MediaPipe WASM runtime and the face-landmarker model into public/
// so the browser loads them from this app (no third-party CDN at runtime).
import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const wasmSrc = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmDst = join(root, 'public', 'mediapipe', 'wasm');
const modelSrc = join(root, '..', 'models', 'face_landmarker.task');
const modelDst = join(root, 'public', 'models', 'face_landmarker.task');

if (existsSync(wasmSrc)) {
  mkdirSync(wasmDst, { recursive: true });
  for (const f of readdirSync(wasmSrc)) cpSync(join(wasmSrc, f), join(wasmDst, f));
  console.log('[assets] MediaPipe WASM copied to public/mediapipe/wasm');
} else {
  console.warn('[assets] @mediapipe/tasks-vision not installed — run npm install');
}

if (existsSync(modelSrc)) {
  mkdirSync(dirname(modelDst), { recursive: true });
  cpSync(modelSrc, modelDst);
  console.log('[assets] face_landmarker.task copied to public/models');
} else if (!existsSync(modelDst)) {
  console.warn('[assets] models/face_landmarker.task missing — run `python scripts/fetch_models.py`. ' +
    'Live tracking will fall back to server-side guidance.');
}
