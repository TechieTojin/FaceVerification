/**
 * In-browser face tracking with MediaPipe FaceLandmarker (VIDEO mode).
 *
 * Runs entirely locally: the WASM runtime and the model are served from this
 * app (`/mediapipe/wasm`, `/models/face_landmarker.task`); no frames leave the
 * browser except the ones explicitly uploaded to the API.
 */
import type { TrackedFace } from './guidance';
import { yawFromLandmarks } from './guidance';

type Landmarker = {
  detectForVideo: (video: HTMLVideoElement, ts: number) => {
    faceLandmarks: { x: number; y: number }[][];
    faceBlendshapes?: { categories: { categoryName: string; score: number }[] }[];
  };
  close: () => void;
};

export interface FaceTracker {
  detect: (video: HTMLVideoElement, ts: number) => TrackedFace[];
  close: () => void;
  kind: 'mediapipe' | 'e2e';
}

// Landmark indices (MediaPipe face mesh topology)
const NOSE = 1;
const CHEEK_L = 234;
const CHEEK_R = 454;
const FOREHEAD = 10;
const CHIN = 152;
const OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
  149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const FEATURES = [33, 133, 159, 145, 362, 263, 386, 374, 1, 4, 61, 291, 13, 14];

declare global {
  interface Window {
    __FV_E2E__?: { detect?: (video: HTMLVideoElement) => TrackedFace[] };
  }
}

let shared: Promise<FaceTracker> | null = null;

export function loadFaceTracker(): Promise<FaceTracker> {
  if (import.meta.env.VITE_E2E_HOOKS === 'true' && typeof window !== 'undefined' && window.__FV_E2E__?.detect) {
    const fn = window.__FV_E2E__.detect;
    return Promise.resolve({ detect: (v) => fn(v), close: () => {}, kind: 'e2e' });
  }
  if (!shared) {
    shared = create().catch((err) => {
      shared = null;
      throw err;
    });
  }
  return shared;
}

async function create(): Promise<FaceTracker> {
  const vision = await import('@mediapipe/tasks-vision');
  const fileset = await vision.FilesetResolver.forVisionTasks('/mediapipe/wasm');
  const opts = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
    runningMode: 'VIDEO' as const,
    numFaces: 3,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  let lm: Landmarker;
  try {
    lm = (await vision.FaceLandmarker.createFromOptions(fileset, opts('GPU'))) as unknown as Landmarker;
  } catch {
    lm = (await vision.FaceLandmarker.createFromOptions(fileset, opts('CPU'))) as unknown as Landmarker;
  }
  let lastTs = 0;
  return {
    kind: 'mediapipe',
    close: () => lm.close(),
    detect(video, ts) {
      const t = ts <= lastTs ? lastTs + 1 : ts;   // MediaPipe requires strictly increasing timestamps
      lastTs = t;
      const res = lm.detectForVideo(video, t);
      return res.faceLandmarks.map((pts, i) => {
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        for (const p of pts) {
          if (p.x < minX) minX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.x > maxX) maxX = p.x;
          if (p.y > maxY) maxY = p.y;
        }
        const bs = new Map((res.faceBlendshapes?.[i]?.categories ?? []).map((c) => [c.categoryName, c.score]));
        const bl = bs.get('eyeBlinkLeft') ?? 0;
        const br = bs.get('eyeBlinkRight') ?? 0;
        const faceH = Math.max(1e-6, pts[CHIN].y - pts[FOREHEAD].y);
        const pitch = ((pts[NOSE].y - (pts[FOREHEAD].y + pts[CHIN].y) / 2) / faceH) * 90;
        return {
          box: { x: minX, y: minY, w: maxX - minX, h: maxY - minY },
          yaw: yawFromLandmarks(pts[NOSE].x, pts[CHEEK_L].x, pts[CHEEK_R].x),
          pitch,
          closureBoth: Math.min(bl, br),
          closureAny: Math.max(bl, br),
          points: [...OUTLINE, ...FEATURES].map((k) => [pts[k].x, pts[k].y] as [number, number]),
        };
      });
    },
  };
}

/** Mean luminance and a cheap sharpness score of the face region (both 0..1). */
export function measureRegion(video: HTMLVideoElement, box: { x: number; y: number; w: number; h: number } | null,
  scratch: HTMLCanvasElement): { brightness: number | null; sharpness: number | null } {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return { brightness: null, sharpness: null };
  const W = 64;
  const H = 64;
  scratch.width = W;
  scratch.height = H;
  const ctx = scratch.getContext('2d', { willReadFrequently: true });
  if (!ctx) return { brightness: null, sharpness: null };
  const b = box ?? { x: 0.25, y: 0.2, w: 0.5, h: 0.6 };
  ctx.drawImage(video, b.x * vw, b.y * vh, Math.max(1, b.w * vw), Math.max(1, b.h * vh), 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H);
  const gray = new Float32Array(W * H);
  let sum = 0;
  for (let i = 0; i < W * H; i++) {
    const g = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    gray[i] = g;
    sum += g;
  }
  let lapSum = 0;
  let lapSq = 0;
  let n = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const lap = gray[i - W] + gray[i + W] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      lapSum += lap;
      lapSq += lap * lap;
      n++;
    }
  }
  const mean = lapSum / n;
  const variance = lapSq / n - mean * mean;
  return { brightness: sum / (W * H) / 255, sharpness: Math.min(1, variance / 300) };
}
