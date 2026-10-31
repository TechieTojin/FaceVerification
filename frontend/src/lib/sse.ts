import { useEffect, useRef, useState } from 'react';
import { refreshSession } from './api';

export type StreamState = 'connecting' | 'live' | 'stale' | 'offline';

export interface StreamMessage {
  event: string;
  channel: string;
  at: string;
  data: Record<string, unknown>;
}

const STALE_AFTER_MS = 35_000;   // server heartbeats every 15 s

export const STREAM_EVENTS = ['session_created', 'session_closed', 'alert', 'alert_updated', 'security_event',
  'enrolment_changed', 'policy_updated', 'notification', 'pairing_claimed', 'pairing_confirmed',
  'pairing_session_started', 'pairing_progress', 'pairing_result', 'pairing_session_ended', 'pairing_expired',
  'pairing_cancelled', 'review_created', 'review_updated', 'device_updated', 'offline_synced'];

/**
 * Subscribe to /api/v1/stream. Reconnects with capped exponential backoff and
 * reports 'stale' when neither data nor heartbeat has arrived recently.
 */
export function useEventStream(channels: string[], onMessage: (m: StreamMessage) => void, enabled = true): {
  state: StreamState;
  lastEventAt: number | null;
} {
  const [state, setState] = useState<StreamState>('connecting');
  const [lastEventAt, setLast] = useState<number | null>(null);
  const handler = useRef(onMessage);
  handler.current = onMessage;
  const key = channels.join(',');

  useEffect(() => {
    if (!enabled || typeof EventSource === 'undefined') {
      setState('offline');
      return;
    }
    let es: EventSource | null = null;
    let retry = 0;
    let timer: number | undefined;
    let staleTimer: number | undefined;
    let closed = false;

    const touch = () => {
      setLast(Date.now());
      setState('live');
      window.clearTimeout(staleTimer);
      staleTimer = window.setTimeout(() => setState('stale'), STALE_AFTER_MS);
    };

    const connect = () => {
      if (closed) return;
      setState('connecting');
      es = new EventSource(`/api/v1/stream?channels=${encodeURIComponent(key)}`, { withCredentials: true });
      es.addEventListener('hello', () => {
        retry = 0;
        touch();
      });
      es.addEventListener('heartbeat', touch);
      const relay = (ev: MessageEvent) => {
        touch();
        try {
          handler.current(JSON.parse(ev.data) as StreamMessage);
        } catch {
          /* ignore malformed */
        }
      };
      for (const name of STREAM_EVENTS) {
        es.addEventListener(name, relay as EventListener);
      }
      // End-to-end test hook (only in the VITE_E2E_HOOKS build): drop the stream as a network failure would.
      if (import.meta.env.VITE_E2E_HOOKS === 'true') {
        (window as unknown as { __FV_SSE_DROP__?: () => void }).__FV_SSE_DROP__ = () => { es?.close(); es?.onerror?.(new Event('error')); };
      }
      es.onerror = () => {
        es?.close();
        setState('offline');
        const delay = Math.min(30_000, 1000 * 2 ** retry++);
        // The access cookie may simply have expired; EventSource cannot refresh it itself.
        void refreshSession().finally(() => {
          if (!closed) timer = window.setTimeout(connect, delay);
        });
      };
    };
    connect();
    return () => {
      closed = true;
      es?.close();
      window.clearTimeout(timer);
      window.clearTimeout(staleTimer);
    };
  }, [key, enabled]);

  return { state, lastEventAt };
}
