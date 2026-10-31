import type { ReactNode } from 'react';
import { Camera, CameraOff, CheckCircle2, Circle, Eye, Loader2, MoveLeft, MoveRight, ScanFace, XCircle } from 'lucide-react';
import { cn } from '../../lib/cn';
import { STEP_LABEL, fmtMs, fmtNum, outcomeMeta } from '../../lib/format';
import type { ChallengeView, FrameView, SessionView, StepType } from '../../lib/types';
import { Badge, Callout } from '../../components/ui/primitives';
import { Button } from '../../components/ui/Button';
import { Select } from '../../components/ui/form';
import { CAMERA_MESSAGES, type CameraApi } from './useCamera';
import type { TrackerStatus } from './useFaceTracker';

const STEP_ICON: Record<StepType, ReactNode> = {
  LOOK_STRAIGHT: <ScanFace className="h-4 w-4" />,
  TURN_LEFT: <MoveLeft className="h-4 w-4" />,
  TURN_RIGHT: <MoveRight className="h-4 w-4" />,
  BLINK: <Eye className="h-4 w-4" />,
};

export function CameraStage({ camera, overlayRef, tracker, headline, tone, children, onRetry }: {
  camera: CameraApi;
  overlayRef: React.RefObject<HTMLCanvasElement | null>;
  tracker: TrackerStatus;
  headline?: string;
  tone?: 'ok' | 'warn' | 'bad' | 'idle';
  children?: ReactNode;
  onRetry?: () => void;
}) {
  const blocked = !['idle', 'requesting', 'active'].includes(camera.status);
  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl bg-black ring-1 ring-line sm:aspect-video" data-testid="camera-stage">
      <video ref={camera.videoRef} className="absolute inset-0 h-full w-full -scale-x-100 object-cover" muted playsInline
        autoPlay aria-label="Live camera preview" />
      <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" aria-hidden />
      {camera.status === 'active' && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center" aria-hidden>
          <div className={cn('h-[78%] w-[42%] rounded-[50%] border-2 border-dashed transition-colors',
            tone === 'ok' ? 'border-good/80' : tone === 'bad' ? 'border-critical/80' : tone === 'warn' ? 'border-warning/80' : 'border-white/40')} />
        </div>
      )}
      {camera.status === 'active' && headline && (
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 to-transparent px-4 pb-3 pt-10">
          <p className="text-center text-sm font-medium text-white" aria-live="polite" data-testid="guidance-headline">{headline}</p>
        </div>
      )}
      {camera.status === 'active' && (
        <div className="absolute left-3 top-3 flex flex-wrap gap-2">
          <span className="rounded-md bg-black/55 px-2 py-1 text-[11px] font-medium text-white">
            {tracker === 'ready' ? 'Live tracking' : tracker === 'loading' ? 'Loading tracker…' : tracker === 'unavailable' ? 'Server guidance only' : ''}
          </span>
        </div>
      )}
      {(camera.status === 'idle' || camera.status === 'requesting') && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-white/80">
          {camera.status === 'requesting' ? <Loader2 className="h-8 w-8 animate-spin" /> : <Camera className="h-10 w-10" />}
          <p className="text-sm">{camera.status === 'requesting' ? 'Waiting for camera permission…' : 'Camera is off'}</p>
        </div>
      )}
      {blocked && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/80 p-6 text-center" data-testid="camera-error">
          <CameraOff className="h-9 w-9 text-white/80" />
          <p className="max-w-sm text-sm text-white">{CAMERA_MESSAGES[camera.status as keyof typeof CAMERA_MESSAGES]}</p>
          {onRetry && <Button variant="secondary" size="sm" onClick={onRetry}>Try again</Button>}
        </div>
      )}
      {children}
    </div>
  );
}

export function DevicePicker({ camera }: { camera: CameraApi }) {
  if (camera.devices.length < 2) return null;
  return (
    <label className="flex items-center gap-2 text-xs text-muted">
      <span className="shrink-0">Camera</span>
      <Select value={camera.deviceId ?? ''} onChange={(e) => camera.setDeviceId(e.target.value)} className="h-8 text-xs" aria-label="Camera device">
        {camera.devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
      </Select>
    </label>
  );
}

export function ChallengeSteps({ challenge, stepProgress }: { challenge: ChallengeView | null; stepProgress: number }) {
  if (!challenge) return null;
  return (
    <ol className="space-y-2" aria-label="Liveness challenge">
      {challenge.steps.map((s, i) => {
        const done = i < challenge.index;
        const current = i === challenge.index;
        return (
          <li key={`${challenge.round}-${i}`} className={cn('flex items-center gap-3 rounded-lg px-3 py-2 text-sm',
            current ? 'bg-primary-soft/70 text-ink' : 'text-ink-2')} aria-current={current ? 'step' : undefined}>
            <span className={cn('flex h-7 w-7 shrink-0 items-center justify-center rounded-full',
              done ? 'bg-good/15 text-good-text' : current ? 'bg-primary text-white' : 'bg-surface-2 text-muted')}>
              {done ? <CheckCircle2 className="h-4 w-4" /> : STEP_ICON[s]}
            </span>
            <span className="flex-1">{STEP_LABEL[s]}</span>
            {done && <span className="text-xs text-good-text">Done</span>}
            {current && (
              <span className="h-1.5 w-16 overflow-hidden rounded-full bg-line" aria-hidden>
                <span className="block h-full bg-primary transition-all" style={{ width: `${Math.round(stepProgress * 100)}%` }} />
              </span>
            )}
            {!done && !current && <Circle className="h-3.5 w-3.5 text-line" aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}

export function FrameReadout({ frame, session }: { frame: FrameView | null; session: SessionView | null }) {
  if (!frame && !session) return null;
  const items: [string, string][] = [];
  if (session) {
    items.push(['Attempts left', String(session.attempts_remaining)]);
    items.push(['Frames', `${session.frames_processed} / ${session.frames_processed + session.frames_remaining}`]);
  }
  if (frame) {
    items.push(['Server time', fmtMs(frame.processing_ms)]);
    if (frame.quality) items.push(['Quality', fmtNum(frame.quality.score, 2)]);
    if (frame.pose) items.push(['Yaw / pitch', `${frame.pose.yaw.toFixed(0)}° / ${frame.pose.pitch.toFixed(0)}°`]);
    if (frame.passive) items.push(['Passive check', `${frame.passive.label} (${(frame.passive.confidence * 100).toFixed(0)}% conf.)`]);
  }
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt className="text-muted">{k}</dt>
          <dd className="tabular font-medium text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ResultPanel({ session, onAgain, extra }: { session: SessionView; onAgain?: () => void; extra?: ReactNode }) {
  const m = outcomeMeta(session.final_outcome);
  const good = m.tone === 'good';
  return (
    <div className={cn('rounded-xl border p-5', good ? 'border-good/40 bg-good/8' : m.tone === 'bad' ? 'border-critical/40 bg-critical/8' : 'border-line bg-surface-2')}
      data-testid="session-result" role="status">
      <div className="flex items-center gap-3">
        {good ? <CheckCircle2 className="h-8 w-8 text-good-text" /> : <XCircle className={cn('h-8 w-8', m.tone === 'bad' ? 'text-critical' : 'text-muted')} />}
        <div>
          <p className="text-lg font-semibold text-ink" data-testid="session-outcome">{m.label}</p>
          <p className="text-sm text-ink-2">{session.frame?.message || (session.reason_code ? session.reason_code.replace(/_/g, ' ').toLowerCase() : '')}</p>
        </div>
      </div>
      {(session.final_distance !== undefined && session.final_distance !== null) && (
        <p className="mt-3 text-xs text-muted">
          Cosine distance {fmtNum(session.final_distance)} vs. threshold {fmtNum(session.threshold ?? null)} (policy v{session.policy_version}).
          Distance is a dissimilarity score, not a probability.
        </p>
      )}
      {extra}
      {onAgain && <Button className="mt-4" variant="outline" onClick={onAgain}>Start another</Button>}
    </div>
  );
}

export function FrameBadge({ frame }: { frame: FrameView | null }) {
  if (!frame) return null;
  const m = outcomeMeta(frame.outcome);
  return (
    <span className="flex items-center gap-1.5 text-xs text-muted" title="Result of the most recent frame the server processed">
      Server:<Badge tone={m.tone}>{m.label}</Badge>
    </span>
  );
}

export function TrackerNotice({ status }: { status: TrackerStatus }) {
  if (status !== 'unavailable') return null;
  return (
    <Callout tone="warn" title="Live face tracking is unavailable">
      The in-browser tracker could not load, so guidance comes from the server after each frame. Verification still works.
    </Callout>
  );
}
