/**
 * Capture-session controller.
 *
 * Concurrency model (all mutable loop state lives in refs, never in closures):
 *   - exactly one frame in flight per session; the loop awaits each upload
 *   - next frame is scheduled only after the previous response (backpressure),
 *     at the server-provided capture interval, or immediately on request
 *     (e.g. eyes closed during a BLINK step)
 *   - every request carries an AbortController; cancel/unmount aborts it
 *   - responses for a session that is no longer current are discarded
 *   - network failures retry with exponential backoff; a stale frame token
 *     (response lost in transit) is recovered via /resync
 *   - the server is authoritative: the loop stops only on a terminal state
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiError, type ApiErrorInfo } from '../../lib/api';
import type { FrameView, SessionView } from '../../lib/types';
import { operatorTransport, type CaptureTransport } from './transport';

export type Phase = 'idle' | 'starting' | 'running' | 'finished' | 'failed';
export type Link = 'ok' | 'retrying' | 'busy';

export interface CaptureGate {
  /** client guidance says the frame is worth uploading */
  sendable: boolean;
  /** client tracker is unavailable → fall back to server-only guidance */
  trackerAvailable: boolean;
}

export interface CaptureOptions {
  start: () => Promise<SessionView>;
  getFrame: (cfg: SessionView['capture']) => Promise<Blob | null>;
  gate: () => CaptureGate;
  onTerminal?: (s: SessionView) => void;
  /** where frames go (default: the signed-in operator's /sessions API) */
  transport?: CaptureTransport;
}

const MAX_NETWORK_RETRIES = 8;
/** Even when local guidance says "not ready", upload one frame this often so the
 *  server session stays alive and returns its own (authoritative) guidance. */
const KEEPALIVE_MS = 4000;

function sleep(ms: number, wake: { fn: (() => void) | null }): Promise<void> {
  return new Promise((resolve) => {
    const t = window.setTimeout(() => { wake.fn = null; resolve(); }, ms);
    wake.fn = () => { window.clearTimeout(t); wake.fn = null; resolve(); };
  });
}

export function useCaptureSession(opts: CaptureOptions) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [session, setSession] = useState<SessionView | null>(null);
  const [frame, setFrame] = useState<FrameView | null>(null);
  const [error, setError] = useState<ApiErrorInfo | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [link, setLink] = useState<Link>('ok');
  const [sent, setSent] = useState(0);

  const optsRef = useRef(opts);
  optsRef.current = opts;
  const sessionRef = useRef<SessionView | null>(null);
  const runId = useRef(0);                 // increments on every begin/cancel → invalidates old loops
  const abortRef = useRef<AbortController | null>(null);
  const wake = useRef<{ fn: (() => void) | null }>({ fn: null });
  const immediate = useRef(false);
  const startingRef = useRef(false);
  const lastSent = useRef(0);

  const applyView = (v: SessionView) => {
    const token = v.frame_token ?? (v.status === 'ACTIVE' ? sessionRef.current?.frame_token : undefined);
    const next = { ...v, frame_token: token };
    sessionRef.current = next;
    setSession(next);
    if (v.frame) setFrame(v.frame);
    return next;
  };

  const finish = (v: SessionView, id: number) => {
    if (id !== runId.current) return;
    applyView(v);
    setPhase('finished');
    optsRef.current.onTerminal?.(v);
  };

  const tx = () => optsRef.current.transport ?? operatorTransport;

  const loop = useCallback(async (id: number) => {
    let netErrors = 0;
    let delay = 0;
    const alive = () => id === runId.current;
    while (alive()) {
      if (delay > 0 && !immediate.current) await sleep(delay, wake.current);
      immediate.current = false;
      if (!alive()) return;
      const s = sessionRef.current;
      if (!s || s.status !== 'ACTIVE' || !s.frame_token) return;
      const interval = s.capture.interval_ms;
      if (Date.now() > Date.parse(s.expires_at.endsWith('Z') ? s.expires_at : `${s.expires_at}Z`) + 1000) {
        try {
          const r = await tx().http.get<SessionView>(`${tx().base}/${s.session_id}`);
          if (alive()) finish(r.data, id);
        } catch (err) {
          if (alive()) { setError(apiError(err)); setPhase('failed'); }
        }
        return;
      }
      const gate = optsRef.current.gate();
      if (gate.trackerAvailable && !gate.sendable && Date.now() - lastSent.current < KEEPALIVE_MS) {
        delay = Math.min(150, interval);         // keep polling the local tracker; nothing uploaded
        continue;
      }
      const blob = await optsRef.current.getFrame(s.capture);
      if (!alive()) return;
      if (!blob) { delay = 100; continue; }

      lastSent.current = Date.now();
      const form = new FormData();
      form.append('seq', String(s.frame_seq + 1));
      form.append('frame_token', s.frame_token);
      form.append('file', blob, 'frame.jpg');
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      try {
        const res = await tx().http.post<SessionView>(`${tx().base}/${s.session_id}/frames`, form, { signal: ctrl.signal });
        if (!alive() || sessionRef.current?.session_id !== s.session_id) return;   // stale response
        netErrors = 0;
        setLink('ok');
        setSent((n) => n + 1);
        setNotice(null);
        const v = applyView(res.data);
        if (v.status !== 'ACTIVE') { finish(v, id); return; }
        delay = interval;
      } catch (err) {
        if (!alive()) return;
        const e = apiError(err);
        const data = (err as { response?: { data?: { session?: SessionView } } }).response?.data;
        if (e.network || (e.status !== null && e.status >= 500 && e.status !== 503)) {
          netErrors++;
          setLink('retrying');
          setNotice(e.network ? 'Connection lost — retrying…' : 'Server error — retrying…');
          if (netErrors > MAX_NETWORK_RETRIES) { setError(e); setPhase('failed'); return; }
          delay = Math.min(8000, 400 * 2 ** netErrors);
          continue;
        }
        if (e.status === 503) {
          setLink('busy');
          setNotice('Server is busy — slowing down');
          delay = (e.retryAfter ?? 1) * 1000;
          continue;
        }
        if (e.status === 409 && e.code === 'SESSION_CLOSED' && data?.session) { finish(data.session, id); return; }
        if (e.status === 409 && e.code === 'FRAME_IN_FLIGHT') { delay = 400; continue; }
        if (e.status === 409 && e.code === 'STALE_FRAME' && tx().resync) {
          try {
            const r = await tx().http.post<SessionView>(`${tx().base}/${s.session_id}/resync`);
            if (!alive()) return;
            const v = applyView(r.data);
            if (v.status !== 'ACTIVE') { finish(v, id); return; }
            delay = 0;
            continue;
          } catch (err2) {
            setError(apiError(err2));
            setPhase('failed');
            return;
          }
        }
        if (e.status === 409 && e.code === 'FRAME_REPLAYED') {
          // Identical bytes: the feed is frozen or replayed. Tell the operator; the server ends the session if it persists.
          setNotice('The camera image is not changing — check the camera or remove any photo or screen');
          delay = interval;
          continue;
        }
        if (e.status === 413 || e.status === 422) {
          setNotice(e.message);
          delay = interval;
          continue;
        }
        setError(e);
        setPhase('failed');
        return;
      } finally {
        if (abortRef.current === ctrl) abortRef.current = null;
      }
    }
  }, []);

  const begin = useCallback(async () => {
    if (startingRef.current) return;              // double-click guard
    startingRef.current = true;
    const id = ++runId.current;
    abortRef.current?.abort();
    setPhase('starting');
    setError(null);
    setNotice(null);
    setFrame(null);
    setSent(0);
    setLink('ok');
    try {
      const v = await optsRef.current.start();
      if (id !== runId.current) {               // cancelled while starting
        void tx().http.post(`${tx().base}/${v.session_id}/cancel`).catch(() => undefined);
        return;
      }
      applyView(v);
      setPhase('running');
      void loop(id);
    } catch (err) {
      if (id === runId.current) {
        setError(apiError(err));
        setPhase('failed');
      }
    } finally {
      startingRef.current = false;
    }
  }, [loop]);

  const cancel = useCallback(async () => {
    runId.current++;
    abortRef.current?.abort();
    wake.current.fn?.();
    const s = sessionRef.current;
    if (s && s.status === 'ACTIVE') {
      try {
        const r = await tx().http.post<SessionView>(`${tx().base}/${s.session_id}/cancel`);
        sessionRef.current = r.data;
        setSession(r.data);
      } catch {
        /* server-side inactivity timeout will close it */
      }
    }
    setPhase((p) => (p === 'idle' ? p : 'finished'));
  }, []);

  const reset = useCallback(() => {
    runId.current++;
    abortRef.current?.abort();
    wake.current.fn?.();
    sessionRef.current = null;
    setSession(null);
    setFrame(null);
    setError(null);
    setNotice(null);
    setPhase('idle');
  }, []);

  const requestImmediate = useCallback(() => {
    immediate.current = true;
    wake.current.fn?.();
  }, []);

  useEffect(() => () => {
    // Unmount / navigation: stop the loop and close the server session.
    runId.current++;
    abortRef.current?.abort();
    wake.current.fn?.();
    const s = sessionRef.current;
    if (s && s.status === 'ACTIVE') void tx().http.post(`${tx().base}/${s.session_id}/cancel`).catch(() => undefined);
  }, []);

  return { phase, session, frame, error, notice, link, sent, begin, cancel, reset, requestImmediate };
}
