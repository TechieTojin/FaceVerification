import type { ReactNode } from 'react';
import type { SessionView } from '../../lib/types';
import { LiveCapturePanel } from './LiveCapturePanel';
import type { CaptureTransport } from './transport';
import { useLiveCapture } from './useLiveCapture';

/**
 * A self-contained live face check against a session owned by a non-operator principal
 * (pending sign-in, step-up, paired phone, kiosk). The server (or kiosk agent) decides.
 */
export function FaceCheck({ start, transport, onTerminal, startLabel = 'Start face check', idleHint, side, onAgain, canStart = true }: {
  start: () => Promise<SessionView>;
  transport: CaptureTransport;
  onTerminal?: (s: SessionView) => void;
  startLabel?: string;
  idleHint?: ReactNode;
  side?: ReactNode;
  onAgain?: () => void;
  canStart?: boolean;
}) {
  const live = useLiveCapture(start, onTerminal, transport);
  return (
    <LiveCapturePanel live={live} startLabel={startLabel} canStart={canStart} idleHint={idleHint} side={side}
      historyLink={false} onAgain={onAgain} />
  );
}
