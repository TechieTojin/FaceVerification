import { useEffect, useRef, useState } from 'react';
import { loadFaceTracker, measureRegion, type FaceTracker } from './faceTracker';
import type { FrameStats, TrackedFace } from './guidance';

export type TrackerStatus = 'off' | 'loading' | 'ready' | 'unavailable';

export interface TrackerSnapshot {
  faces: TrackedFace[];
  stats: FrameStats;
  fps: number;
  lostSince: number | null;
}

const DETECT_EVERY_MS = 66;
const UI_EVERY_MS = 120;

/**
 * Runs the local face tracker on the live video and draws the overlay.
 * Detection results are real MediaPipe output; nothing here is decorative.
 */
export function useFaceTracker(video: React.RefObject<HTMLVideoElement | null>,
  overlay: React.RefObject<HTMLCanvasElement | null>, active: boolean,
  colorFor: (faces: TrackedFace[], stats: FrameStats) => string) {
  const [status, setStatus] = useState<TrackerStatus>('off');
  const [snap, setSnap] = useState<TrackerSnapshot>({ faces: [], stats: { brightness: null, sharpness: null }, fps: 0, lostSince: null });
  const latest = useRef<TrackerSnapshot>(snap);
  const colorRef = useRef(colorFor);
  colorRef.current = colorFor;

  useEffect(() => {
    if (!active) {
      setStatus('off');
      return;
    }
    let tracker: FaceTracker | null = null;
    let raf = 0;
    let cancelled = false;
    let lastDetect = 0;
    let lastUi = 0;
    let frames = 0;
    let fpsWindow = performance.now();
    let fps = 0;
    let lostSince: number | null = null;
    const scratch = document.createElement('canvas');
    setStatus('loading');

    const tick = (now: number) => {
      if (cancelled) return;
      raf = requestAnimationFrame(tick);
      const v = video.current;
      if (!tracker || !v || v.readyState < 2 || now - lastDetect < DETECT_EVERY_MS) return;
      lastDetect = now;
      let faces: TrackedFace[] = [];
      try {
        faces = tracker.detect(v, now);
      } catch {
        return;
      }
      frames++;
      if (now - fpsWindow >= 1000) {
        fps = (frames * 1000) / (now - fpsWindow);
        frames = 0;
        fpsWindow = now;
      }
      if (faces.length === 0) lostSince = lostSince ?? now;
      else lostSince = null;
      const stats = measureRegion(v, faces.length === 1 ? faces[0].box : null, scratch);
      latest.current = { faces, stats, fps, lostSince };
      draw(overlay.current, v, faces, colorRef.current(faces, stats));
      if (now - lastUi >= UI_EVERY_MS) {
        lastUi = now;
        setSnap(latest.current);
      }
    };

    loadFaceTracker()
      .then((t) => {
        if (cancelled) return;
        tracker = t;
        setStatus('ready');
        raf = requestAnimationFrame(tick);
      })
      .catch(() => {
        if (!cancelled) setStatus('unavailable');
      });

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      const c = overlay.current;
      c?.getContext('2d')?.clearRect(0, 0, c.width, c.height);
    };
  }, [active, video, overlay]);

  return { status, snapshot: snap, latest };
}

function draw(canvas: HTMLCanvasElement | null, video: HTMLVideoElement, faces: TrackedFace[], color: string) {
  if (!canvas) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  // Map normalised video coordinates to the element box used by `object-cover`.
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const scale = Math.max(w / vw, h / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (w - dw) / 2;
  const oy = (h - dh) / 2;
  const X = (x: number) => ox + x * dw;
  const Y = (y: number) => oy + y * dh;
  for (const f of faces) {
    const bx = X(f.box.x);
    const by = Y(f.box.y);
    const bw = f.box.w * dw;
    const bh = f.box.h * dh;
    ctx.strokeStyle = faces.length > 1 ? '#d03b3b' : color;
    ctx.lineWidth = 2;
    const r = Math.min(16, bw / 6);
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, r);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    for (const [px, py] of f.points) {
      ctx.beginPath();
      ctx.arc(X(px), Y(py), 1.4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
