import { useCallback, useEffect, useMemo, useRef } from 'react';
import type { SessionView } from '../../lib/types';
import { captureFrame } from './capture';
import { computeGuidance, shouldSendImmediately, type GuidanceConfig } from './guidance';
import { useCamera } from './useCamera';
import { useCaptureSession } from './useCaptureSession';
import { useFaceTracker } from './useFaceTracker';
import type { CaptureTransport } from './transport';

const DEFAULT_CFG: GuidanceConfig = { minWidth: 0.2, maxWidth: 0.75, yawThreshold: 18, blinkThreshold: 0.5 };

/** Camera + local tracking + guidance + server session, wired together. */
export function useLiveCapture(start: () => Promise<SessionView>, onTerminal?: (s: SessionView) => void,
  transport?: CaptureTransport) {
  const camera = useCamera();
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const encodeCanvas = useRef<HTMLCanvasElement | null>(null);
  const stepRef = useRef<SessionView['challenge']>(null);
  const cfgRef = useRef<GuidanceConfig>(DEFAULT_CFG);

  const tracker = useFaceTracker(camera.videoRef, overlayRef, camera.status === 'active', (faces, stats) => {
    const g = computeGuidance(faces, stats, stepRef.current?.type ?? null, cfgRef.current);
    return g.sendable ? '#0ca30c' : '#fab219';
  });

  const session = useCaptureSession({
    start,
    transport,
    getFrame: (cfg) => {
      const v = camera.videoRef.current;
      if (!v) return Promise.resolve(null);
      encodeCanvas.current ??= document.createElement('canvas');
      return captureFrame(v, cfg.max_width, cfg.jpeg_quality, encodeCanvas.current);
    },
    gate: () => {
      const { faces, stats } = tracker.latest.current;
      const g = computeGuidance(faces, stats, stepRef.current?.type ?? null, cfgRef.current);
      return { sendable: g.sendable, trackerAvailable: tracker.status === 'ready' };
    },
    onTerminal: (s) => {
      camera.stop();          // stop the camera as soon as a decision is reached
      onTerminal?.(s);
    },
  });

  useEffect(() => {
    stepRef.current = session.session?.challenge ?? null;
    const c = session.session?.capture;
    if (c) {
      cfgRef.current = { minWidth: c.min_face_width_ratio, maxWidth: c.max_face_width_ratio,
        yawThreshold: c.yaw_threshold_degrees ?? 18, blinkThreshold: c.blink_threshold ?? 0.5 };
    }
  }, [session.session]);

  const snap = tracker.snapshot;
  const guidance = useMemo(
    () => computeGuidance(snap.faces, snap.stats, session.session?.challenge?.type ?? null, cfgRef.current),
    [snap, session.session],
  );

  useEffect(() => {
    if (session.phase !== 'running') return;
    const face = snap.faces.length === 1 ? snap.faces[0] : null;
    if (shouldSendImmediately(face, session.session?.challenge?.type ?? null, cfgRef.current)) session.requestImmediate();
  }, [snap, session]);

  const { begin: beginSession, cancel: cancelSession, reset: resetSession } = session;
  const begin = useCallback(async () => {
    const ok = camera.status === 'active' ? true : await camera.start();
    if (ok) await beginSession();
  }, [camera, beginSession]);

  const cancel = useCallback(async () => {
    await cancelSession();
    camera.stop();
  }, [cancelSession, camera]);

  const reset = useCallback(() => {
    resetSession();
    camera.stop();
  }, [resetSession, camera]);

  // Headline: client guidance while tracking is available, otherwise the server's latest guidance.
  const serverHint = session.frame?.guidance?.[0] ?? session.session?.challenge?.instruction;
  const headline = session.phase === 'running'
    ? (tracker.status === 'ready' ? guidance.headline : serverHint ?? 'Look at the camera')
    : camera.status === 'active' ? guidance.headline : undefined;
  const tone: 'ok' | 'warn' | 'bad' | 'idle' = camera.status !== 'active' ? 'idle'
    : guidance.issues.includes('MULTIPLE_FACES') ? 'bad' : guidance.sendable ? 'ok' : 'warn';

  return { camera, overlayRef, tracker, guidance, session, begin, cancel, reset, headline, tone };
}
