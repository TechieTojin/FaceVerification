import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Square, Wifi, WifiOff } from 'lucide-react';
import { Button } from '../../components/ui/Button';
import { Card, Callout } from '../../components/ui/primitives';
import { CameraStage, ChallengeSteps, DevicePicker, FrameBadge, FrameReadout, ResultPanel, TrackerNotice } from './components';
import type { useLiveCapture } from './useLiveCapture';

type Live = ReturnType<typeof useLiveCapture>;

/** Camera + challenge + controls, shared by verification, enrolment and document checks. */
export function LiveCapturePanel({ live, startLabel, canStart, idleHint, resultExtra, onAgain, side, historyLink = true,
  showDevicePicker = true }: {
  live: Live; startLabel: string; canStart: boolean; idleHint?: ReactNode; resultExtra?: ReactNode;
  onAgain?: () => void; side?: ReactNode; historyLink?: boolean; showDevicePicker?: boolean;
}) {
  const { camera, overlayRef, tracker, guidance, session: s } = live;
  const running = s.phase === 'running' || s.phase === 'starting';
  const done = s.phase === 'finished' && s.session && s.session.status === 'CLOSED';
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
      <div className="space-y-3">
        <CameraStage camera={camera} overlayRef={overlayRef} tracker={tracker.status} headline={live.headline} tone={live.tone}
          onRetry={() => void camera.start()} />
        <div className="flex flex-wrap items-center justify-between gap-3">
          {showDevicePicker ? <DevicePicker camera={camera} /> : <span />}
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
            {s.phase === 'running' && (s.link === 'ok'
              ? <span className="inline-flex items-center gap-1"><Wifi className="h-3.5 w-3.5" />Connected · {s.sent} frames sent</span>
              : <span className="inline-flex items-center gap-1 text-critical"><WifiOff className="h-3.5 w-3.5" />{s.notice}</span>)}
            {tracker.status === 'ready' && camera.status === 'active' && tracker.snapshot.fps > 0 && (
              <span>{tracker.snapshot.fps.toFixed(0)} fps tracking</span>)}
          </div>
        </div>
        <TrackerNotice status={tracker.status} />
      </div>

      <div className="flex flex-col gap-4">
        {/* While capturing, the challenge card comes first so it stays next to the camera on small screens. */}
        <div className={running ? 'order-2' : 'order-1'}>{side}</div>
        <Card className={running ? 'order-1 p-5' : 'order-2 p-5'}>
          {!running && !done && (
            <div className="space-y-3">
              {idleHint}
              {s.error && <Callout tone="bad" title="Could not continue">{s.error.message}</Callout>}
              <Button className="w-full" size="lg" disabled={!canStart} onClick={() => void live.begin()} data-testid="start-capture">
                {startLabel}
              </Button>
            </div>
          )}
          {running && (
            <div className="space-y-4" data-testid="capture-running">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-semibold text-ink">
                  {s.session?.challenge ? `Step ${s.session.challenge.index + 1} of ${s.session.challenge.total}` : 'Starting…'}
                </p>
                <FrameBadge frame={s.frame} />
              </div>
              <ChallengeSteps challenge={s.session?.challenge ?? null} stepProgress={guidance.stepProgress} />
              {s.frame?.new_round && <Callout tone="warn">{s.frame.message}</Callout>}
              {s.frame && s.frame.guidance.length > 0 && tracker.status !== 'ready' && (
                <ul className="list-disc pl-5 text-sm text-ink-2">{s.frame.guidance.map((g) => <li key={g}>{g}</li>)}</ul>
              )}
              {s.notice && s.link === 'ok' && <Callout tone="warn">{s.notice}</Callout>}
              <FrameReadout frame={s.frame} session={s.session} />
              <Button variant="outline" className="w-full" icon={<Square className="h-4 w-4" />} onClick={() => void live.cancel()}
                data-testid="cancel-capture">Cancel</Button>
            </div>
          )}
          {done && s.session && (
            <ResultPanel session={s.session} extra={<>
              {resultExtra}
              {historyLink && (
                <Link to={`/history/${s.session.session_id}`} className="mt-3 block text-xs font-medium text-primary hover:underline">
                  View session details
                </Link>
              )}
            </>} onAgain={onAgain ?? (() => live.reset())} />
          )}
        </Card>
      </div>
    </div>
  );
}
