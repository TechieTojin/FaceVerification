/**
 * Client-side capture guidance (pure functions, unit tested).
 *
 * These measurements only steer the user and avoid uploading obviously
 * unusable frames. Every decision that matters (quality gate, liveness,
 * identity) is re-made on the server from the uploaded pixels.
 */
import type { StepType } from '../../lib/types';

export interface Box { x: number; y: number; w: number; h: number }   // normalised 0..1, unmirrored

export interface TrackedFace {
  box: Box;
  yaw: number;              // + = subject turned to *their* left (same convention as the server)
  pitch: number;
  closureBoth: number;      // eye closure both eyes reach (0 open … 1 closed)
  closureAny: number;
  points: [number, number][];
}

export interface FrameStats { brightness: number | null; sharpness: number | null }

export interface GuidanceConfig { minWidth: number; maxWidth: number; yawThreshold: number; blinkThreshold: number }

export type IssueCode =
  | 'NO_FACE' | 'MULTIPLE_FACES' | 'FACE_TOO_SMALL' | 'FACE_TOO_LARGE' | 'OUT_OF_FRAME'
  | 'TOO_DARK' | 'TOO_BRIGHT' | 'BLURRY' | 'POSE';

export const MESSAGES: Record<IssueCode, string> = {
  NO_FACE: 'No face detected — look at the camera',
  MULTIPLE_FACES: 'Only one person should be visible',
  FACE_TOO_SMALL: 'Move closer',
  FACE_TOO_LARGE: 'Move back slightly',
  OUT_OF_FRAME: 'Center your face in the frame',
  TOO_DARK: 'Improve lighting',
  TOO_BRIGHT: 'Reduce bright light or glare',
  BLURRY: 'Hold still',
  POSE: 'Look directly at the camera',
};

export interface Guidance {
  issues: IssueCode[];
  messages: string[];
  /** true when the frame is worth sending to the server */
  sendable: boolean;
  /** 0..1 — how close the user is to satisfying the current step (for the progress ring) */
  stepProgress: number;
  /** primary instruction to show */
  headline: string;
}

const STEP_TEXT: Record<StepType, string> = {
  LOOK_STRAIGHT: 'Look straight at the camera',
  TURN_LEFT: 'Slowly turn your head to your left',
  TURN_RIGHT: 'Slowly turn your head to your right',
  BLINK: 'Blink slowly',
};

const BLOCKING: IssueCode[] = ['NO_FACE', 'MULTIPLE_FACES', 'FACE_TOO_SMALL', 'FACE_TOO_LARGE', 'OUT_OF_FRAME', 'TOO_DARK'];

export function computeGuidance(faces: TrackedFace[], stats: FrameStats, step: StepType | null,
  cfg: GuidanceConfig): Guidance {
  const issues: IssueCode[] = [];
  if (faces.length === 0) issues.push('NO_FACE');
  else if (faces.length > 1) issues.push('MULTIPLE_FACES');
  const face = faces.length === 1 ? faces[0] : null;
  let stepProgress = 0;
  if (face) {
    const { x, y, w, h } = face.box;
    if (w < cfg.minWidth) issues.push('FACE_TOO_SMALL');
    else if (w > cfg.maxWidth) issues.push('FACE_TOO_LARGE');
    if (x < 0.01 || y < 0.01 || x + w > 0.99 || y + h > 0.99) issues.push('OUT_OF_FRAME');
    if (stats.brightness !== null) {
      if (stats.brightness < 0.22) issues.push('TOO_DARK');
      else if (stats.brightness > 0.85) issues.push('TOO_BRIGHT');
    }
    if (stats.sharpness !== null && stats.sharpness < 0.2) issues.push('BLURRY');
    const straight = step === 'LOOK_STRAIGHT' || step === null;
    // Client pitch is a coarse landmark estimate, so only flag clearly tilted heads; the server is strict.
    if (straight && (Math.abs(face.yaw) > 12 || Math.abs(face.pitch) > 28)) issues.push('POSE');
    if (step === 'TURN_LEFT') stepProgress = clamp01(face.yaw / cfg.yawThreshold);
    else if (step === 'TURN_RIGHT') stepProgress = clamp01(-face.yaw / cfg.yawThreshold);
    else if (step === 'BLINK') stepProgress = clamp01(face.closureBoth / cfg.blinkThreshold);
    else stepProgress = issues.length ? 0 : 1;
  }
  const blocking = issues.filter((i) => BLOCKING.includes(i) || (i === 'POSE' && step === 'LOOK_STRAIGHT'));
  const messages = issues.map((i) => MESSAGES[i]);
  const headline = blocking.length ? MESSAGES[blocking[0]] : step ? STEP_TEXT[step] : 'Position your face in the oval';
  return { issues, messages, sendable: blocking.length === 0, stepProgress, headline };
}

/**
 * During a BLINK step a closed-eyes frame is short-lived; send it immediately
 * (still subject to single-flight) rather than waiting for the next interval.
 */
export function shouldSendImmediately(face: TrackedFace | null, step: StepType | null, cfg: GuidanceConfig): boolean {
  if (!face || step !== 'BLINK') return false;
  return face.closureBoth >= cfg.blinkThreshold;
}

export function clamp01(v: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

/** Estimate yaw (deg) from 2-D landmarks: nose offset relative to the cheek line. */
export function yawFromLandmarks(noseX: number, leftCheekX: number, rightCheekX: number): number {
  const width = Math.abs(rightCheekX - leftCheekX);
  if (width < 1e-6) return 0;
  const offset = (noseX - (leftCheekX + rightCheekX) / 2) / width;
  return offset * 60;
}
