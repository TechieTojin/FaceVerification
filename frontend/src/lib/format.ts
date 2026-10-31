import type { FrameOutcome } from './types';

/** API datetimes are naive UTC ISO strings; treat them as UTC. */
export function parseUtc(value: string | null | undefined): Date | null {
  if (!value) return null;
  const s = /[zZ]|[+-]\d\d:?\d\d$/.test(value) ? value : `${value}Z`;
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function fmtDateTime(value: string | null | undefined): string {
  const d = parseUtc(value);
  return d ? d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—';
}

export function fmtTime(value: string | null | undefined): string {
  const d = parseUtc(value);
  return d ? d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
}

export function fmtRelative(value: string | null | undefined, now = Date.now()): string {
  const d = parseUtc(value);
  if (!d) return '—';
  const s = Math.round((now - d.getTime()) / 1000);
  if (Math.abs(s) < 45) return 'just now';
  const m = Math.round(s / 60);
  if (Math.abs(m) < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (Math.abs(h) < 24) return `${h} h ago`;
  return fmtDateTime(value);
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

export function fmtPct(rate: number | null | undefined, digits = 1): string {
  return rate === null || rate === undefined ? '—' : `${(rate * 100).toFixed(digits)}%`;
}

export function fmtNum(n: number | null | undefined, digits = 3): string {
  return n === null || n === undefined ? '—' : n.toFixed(digits);
}

export function fmtDuration(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  return `${m}m ${Math.floor(seconds % 60)}s`;
}

export function humanize(code: string | null | undefined): string {
  if (!code) return '—';
  return code.toLowerCase().replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

export type Tone = 'good' | 'bad' | 'warn' | 'info' | 'neutral';

export const OUTCOME_META: Record<FrameOutcome, { label: string; tone: Tone }> = {
  VERIFIED: { label: 'Verified', tone: 'good' },
  ENROLMENT_CAPTURED: { label: 'Enrolment captured', tone: 'good' },
  NOT_MATCHED: { label: 'Not matched', tone: 'bad' },
  LIVENESS_FAILED: { label: 'Liveness failed', tone: 'bad' },
  ATTEMPTS_EXHAUSTED: { label: 'Attempts exhausted', tone: 'bad' },
  ERROR: { label: 'Error', tone: 'bad' },
  SESSION_EXPIRED: { label: 'Expired', tone: 'warn' },
  SESSION_CANCELLED: { label: 'Cancelled', tone: 'neutral' },
  RETRY_REQUIRED: { label: 'Retry required', tone: 'warn' },
  NO_FACE: { label: 'No face', tone: 'warn' },
  MULTIPLE_FACES: { label: 'Multiple faces', tone: 'warn' },
  QUALITY_FAILED: { label: 'Quality too low', tone: 'warn' },
  IN_PROGRESS: { label: 'In progress', tone: 'info' },
};

export function outcomeMeta(o: string | null | undefined): { label: string; tone: Tone } {
  if (!o) return { label: 'In progress', tone: 'info' };
  return OUTCOME_META[o as FrameOutcome] ?? { label: humanize(o), tone: 'neutral' };
}

export const STEP_LABEL: Record<string, string> = {
  LOOK_STRAIGHT: 'Look straight',
  TURN_LEFT: 'Turn left',
  TURN_RIGHT: 'Turn right',
  BLINK: 'Blink',
};

export const REASON_TEXT: Record<string, string> = {
  EXISTING_TEMPLATE: 'Replaces an existing template',
  DUPLICATE_SUSPECTED: 'Matches another enrolled person',
  SELF_ENROLMENT: 'Operator enrolled themselves',
};
