import { AxiosError, AxiosHeaders, type AxiosAdapter, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, setStepUpHandler } from './api';
import { useStepUp } from '../store/stepUp';
import { deferredHeaderTransport, headerTransport } from '../features/camera/transport';

function respond(config: InternalAxiosRequestConfig, status: number, data: unknown): Promise<AxiosResponse> {
  const res = { status, data, headers: {}, config, statusText: '' } as AxiosResponse;
  if (status >= 400) return Promise.reject(new AxiosError('x', undefined, config, undefined, res));
  return Promise.resolve(res);
}

describe('step-up interception', () => {
  afterEach(() => { setStepUpHandler(null); vi.restoreAllMocks(); });

  it('asks for step-up once, then retries the original request', async () => {
    let calls = 0;
    const adapter: AxiosAdapter = (config) => {
      calls++;
      return calls === 1
        ? respond(config, 403, { code: 'STEP_UP_REQUIRED', detail: 'Confirm', step_up_level: 'face_and_factor' })
        : respond(config, 201, { ok: true });
    };
    const handler = vi.fn().mockResolvedValue(true);
    setStepUpHandler(handler);
    const r = await api.post('/integrations/clients', { name: 'x' }, { adapter });
    expect(r.data).toEqual({ ok: true });
    expect(handler).toHaveBeenCalledWith('face_and_factor');
    expect(calls).toBe(2);
  });

  it('does not retry when the person cancels, and never loops', async () => {
    const adapter: AxiosAdapter = (config) => respond(config, 403, { code: 'STEP_UP_REQUIRED', detail: 'Confirm' });
    setStepUpHandler(vi.fn().mockResolvedValue(false));
    await expect(api.post('/devices', {}, { adapter })).rejects.toBeInstanceOf(AxiosError);
    const always = vi.fn().mockResolvedValue(true);
    setStepUpHandler(always);
    await expect(api.post('/devices', {}, { adapter })).rejects.toBeInstanceOf(AxiosError);
    expect(always).toHaveBeenCalledTimes(1);           // a second STEP_UP_REQUIRED is not retried again
  });

  it('ignores other 403 responses', async () => {
    const handler = vi.fn();
    setStepUpHandler(handler);
    const adapter: AxiosAdapter = (config) => respond(config, 403, { detail: 'You do not have permission' });
    await expect(api.get('/users', { adapter })).rejects.toBeInstanceOf(AxiosError);
    expect(handler).not.toHaveBeenCalled();
  });

  it('concurrent step-up requests share one dialog', async () => {
    const a = useStepUp.getState().request('standard');
    const b = useStepUp.getState().request('standard');
    expect(useStepUp.getState().open).toBe(true);
    useStepUp.getState().finish(true);
    await expect(Promise.all([a, b])).resolves.toEqual([true, true]);
    expect(useStepUp.getState().open).toBe(false);
  });
});

describe('header transports', () => {
  it('send the header credential and the CSRF token, never a stale value', async () => {
    document.cookie = 'fv_csrf=csrf-123';
    const seen: Record<string, string | undefined>[] = [];
    const adapter: AxiosAdapter = (config) => {
      const h = config.headers as AxiosHeaders;
      seen.push({ secret: h.get('X-Pairing-Secret') as string | undefined, login: h.get('X-Login-Token') as string | undefined,
        csrf: h.get('X-CSRF-Token') as string | undefined });
      return respond(config, 200, {});
    };
    await headerTransport('X-Pairing-Secret', 'pair-secret').http.post('/remote/session', {}, { adapter });
    const deferred = deferredHeaderTransport('X-Login-Token');
    await deferred.transport.http.get('/sessions/x', { adapter });
    deferred.set('login-token');
    await deferred.transport.http.post('/sessions/x/frames', {}, { adapter });
    expect(seen).toEqual([
      { secret: 'pair-secret', login: undefined, csrf: 'csrf-123' },
      { secret: undefined, login: undefined, csrf: undefined },          // GET: no CSRF; token not set yet
      { secret: undefined, login: 'login-token', csrf: 'csrf-123' },
    ]);
    document.cookie = 'fv_csrf=; expires=Thu, 01 Jan 1970 00:00:00 GMT';
  });
});
