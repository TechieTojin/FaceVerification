import { act, renderHook, waitFor } from '@testing-library/react';
import { AxiosError, AxiosHeaders, type InternalAxiosRequestConfig } from 'axios';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SessionView } from '../../lib/types';

const post = vi.fn();
const get = vi.fn();
vi.mock('../../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../../lib/api')>('../../lib/api');
  return { ...actual, api: { post: (...a: unknown[]) => post(...a), get: (...a: unknown[]) => get(...a) } };
});

const { useCaptureSession } = await import('./useCaptureSession');

function view(over: Partial<SessionView> = {}): SessionView {
  return {
    session_id: 's1', purpose: 'VERIFY', target_user_id: 'u', document_check_id: null, status: 'ACTIVE', final_outcome: null,
    reason_code: null, created_at: new Date().toISOString(), expires_at: new Date(Date.now() + 60_000).toISOString(),
    completed_at: null, frame_seq: 0, frames_processed: 0, frames_remaining: 100, attempts_used: 0, attempts_remaining: 3,
    policy_version: 1, frame: null, frame_token: 't0',
    capture: { interval_ms: 5, max_width: 640, jpeg_quality: 0.8, max_frame_bytes: 600000, min_face_width_ratio: 0.2,
      max_face_width_ratio: 0.75, yaw_threshold_degrees: 18, blink_threshold: 0.5 },
    challenge: { round: 0, index: 0, total: 1, type: 'LOOK_STRAIGHT', instruction: 'Look', steps: ['LOOK_STRAIGHT'], progress: 0 },
    ...over,
  };
}

function httpError(status: number | null, data?: unknown, code?: string) {
  const config = { headers: new AxiosHeaders() } as InternalAxiosRequestConfig;
  return new AxiosError('x', code, config, undefined,
    status === null ? undefined : { status, data, headers: {}, config, statusText: '' });
}

const blob = () => Promise.resolve(new Blob(['x']));
const sendable = () => ({ sendable: true, trackerAvailable: true });

function setup(opts: { gate?: () => { sendable: boolean; trackerAvailable: boolean } } = {}) {
  const start = vi.fn().mockResolvedValue(view());
  const hook = renderHook(() => useCaptureSession({ start, getFrame: blob, gate: opts.gate ?? sendable }));
  return { start, hook };
}

const frameCalls = () => post.mock.calls.filter((c) => String(c[0]).endsWith('/frames'));

beforeEach(() => {
  post.mockReset();
  get.mockReset();
});

describe('useCaptureSession', () => {
  it('never has more than one frame in flight and stops on a terminal state', async () => {
    let inflight = 0;
    let maxInflight = 0;
    let n = 0;
    post.mockImplementation(async (url: string) => {
      if (!url.endsWith('/frames')) return { data: view({ status: 'CLOSED' }) };
      inflight++;
      maxInflight = Math.max(maxInflight, inflight);
      await new Promise((r) => setTimeout(r, 20));
      inflight--;
      n++;
      return { data: n >= 5 ? view({ status: 'CLOSED', final_outcome: 'VERIFIED', frame_token: undefined, frame_seq: n })
        : view({ frame_seq: n, frame_token: `t${n}` }) };
    });
    const { hook } = setup();
    await act(() => hook.result.current.begin());
    await waitFor(() => expect(hook.result.current.phase).toBe('finished'));
    expect(maxInflight).toBe(1);
    expect(frameCalls()).toHaveLength(5);
    // each frame carries the token and sequence issued by the previous response
    const forms = frameCalls().map((c) => c[1] as FormData);
    expect(forms.map((f) => f.get('seq'))).toEqual(['1', '2', '3', '4', '5']);
    expect(forms.map((f) => f.get('frame_token'))).toEqual(['t0', 't1', 't2', 't3', 't4']);
    await new Promise((r) => setTimeout(r, 50));
    expect(frameCalls()).toHaveLength(5);
    expect(hook.result.current.session?.final_outcome).toBe('VERIFIED');
  });

  it('does not upload while local guidance says the frame is unusable', async () => {
    const { hook } = setup({ gate: () => ({ sendable: false, trackerAvailable: true }) });
    post.mockResolvedValue({ data: view({ frame_seq: 1, frame_token: 't1' }) });
    await act(() => hook.result.current.begin());
    await new Promise((r) => setTimeout(r, 120));
    expect(frameCalls()).toHaveLength(1);              // only the initial keep-alive frame
  });

  it('ignores a response that arrives after cancel', async () => {
    let release!: () => void;
    post.mockImplementation((url: string) => {
      if (url.endsWith('/cancel')) return Promise.resolve({ data: view({ status: 'CLOSED', final_outcome: 'SESSION_CANCELLED' }) });
      return new Promise((r) => { release = () => r({ data: view({ frame_seq: 1, frame_token: 't1' }) }); });
    });
    const { hook } = setup();
    await act(() => hook.result.current.begin());
    await waitFor(() => expect(frameCalls()).toHaveLength(1));
    await act(() => hook.result.current.cancel());
    act(() => release());
    await new Promise((r) => setTimeout(r, 30));
    expect(hook.result.current.session?.final_outcome).toBe('SESSION_CANCELLED');
    expect(frameCalls()).toHaveLength(1);
  });

  it('retries after a network failure and recovers', async () => {
    let n = 0;
    post.mockImplementation(async (url: string) => {
      if (!url.endsWith('/frames')) return { data: view() };
      n++;
      if (n === 1) throw httpError(null, undefined, 'ERR_NETWORK');
      return { data: view({ status: 'CLOSED', final_outcome: 'VERIFIED', frame_seq: 1 }) };
    });
    const { hook } = setup();
    await act(() => hook.result.current.begin());
    await waitFor(() => expect(hook.result.current.link).toBe('retrying'));
    expect(hook.result.current.notice).toMatch(/Connection lost/);
    await waitFor(() => expect(hook.result.current.phase).toBe('finished'), { timeout: 3000 });
    expect(hook.result.current.link).toBe('ok');
  });

  it('resyncs the frame token after a lost response', async () => {
    let n = 0;
    post.mockImplementation(async (url: string, form?: FormData) => {
      if (url.endsWith('/resync')) return { data: view({ frame_seq: 1, frame_token: 'fresh' }) };
      n++;
      if (n === 1) throw httpError(409, { detail: 'stale', code: 'STALE_FRAME' });
      expect(form?.get('frame_token')).toBe('fresh');
      return { data: view({ status: 'CLOSED', final_outcome: 'VERIFIED' }) };
    });
    const { hook } = setup();
    await act(() => hook.result.current.begin());
    await waitFor(() => expect(hook.result.current.phase).toBe('finished'));
    expect(post.mock.calls.some((c) => String(c[0]).endsWith('/resync'))).toBe(true);
  });

  it('backs off when the server is busy', async () => {
    let n = 0;
    post.mockImplementation(async () => {
      n++;
      if (n === 1) throw httpError(503, { detail: 'busy', code: 'INFERENCE_BUSY' });
      return { data: view({ status: 'CLOSED', final_outcome: 'VERIFIED' }) };
    });
    const { hook } = setup();
    await act(() => hook.result.current.begin());
    await waitFor(() => expect(hook.result.current.link).toBe('busy'));
    await waitFor(() => expect(hook.result.current.phase).toBe('finished'), { timeout: 3000 });
  });

  it('a double start creates one session', async () => {
    post.mockResolvedValue({ data: view({ status: 'CLOSED', final_outcome: 'VERIFIED' }) });
    const { hook, start } = setup();
    await act(async () => {
      void hook.result.current.begin();
      void hook.result.current.begin();
    });
    await waitFor(() => expect(hook.result.current.phase).toBe('finished'));
    expect(start).toHaveBeenCalledTimes(1);
  });

  it('cancels the server session on unmount', async () => {
    post.mockImplementation((url: string) => (url.endsWith('/cancel')
      ? Promise.resolve({ data: view({ status: 'CLOSED' }) })
      : new Promise(() => {})));
    const { hook } = setup();
    await act(() => hook.result.current.begin());
    hook.unmount();
    expect(post.mock.calls.some((c) => String(c[0]).endsWith('/s1/cancel'))).toBe(true);
  });

  it('surfaces a start failure instead of swallowing it', async () => {
    const start = vi.fn().mockRejectedValue(httpError(409, { detail: 'This user is not enrolled', code: 'NOT_ENROLLED' }));
    const hook = renderHook(() => useCaptureSession({ start, getFrame: blob, gate: sendable }));
    await act(() => hook.result.current.begin());
    expect(hook.result.current.phase).toBe('failed');
    expect(hook.result.current.error?.message).toBe('This user is not enrolled');
  });
});
