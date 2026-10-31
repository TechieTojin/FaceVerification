import { AxiosError, AxiosHeaders, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios';
import { afterEach, describe, expect, it, vi } from 'vitest';
import axios from 'axios';
import { api, apiError, refreshSession, setUnauthorizedHandler } from './api';

function err(status: number | null, data?: unknown, headers: Record<string, string> = {}, code?: string,
  cfg?: InternalAxiosRequestConfig): AxiosError {
  const config = cfg ?? ({ headers: new AxiosHeaders() } as InternalAxiosRequestConfig);
  const response = status === null ? undefined
    : ({ status, data, headers, config, statusText: '' } as AxiosResponse);
  return new AxiosError('x', code, config, undefined, response);
}

describe('apiError', () => {
  it('handles string, structured and validation details', () => {
    expect(apiError(err(409, { detail: 'Closed', code: 'SESSION_CLOSED' }))).toMatchObject({ status: 409, code: 'SESSION_CLOSED', message: 'Closed' });
    expect(apiError(err(409, { detail: { message: 'Confirm', code: 'CONFIRMATION_REQUIRED' } }))).toMatchObject({ code: 'CONFIRMATION_REQUIRED', message: 'Confirm' });
    const v = apiError(err(422, { detail: [{ loc: ['body', 'email'], msg: 'Value error, Enter a valid email address' }] }));
    expect(v.fieldErrors).toEqual({ email: 'Enter a valid email address' });
    expect(apiError(err(422, { detail: { message: 'Invalid settings', errors: { max_attempts: 'must be at most 10' } } })).fieldErrors)
      .toEqual({ max_attempts: 'must be at most 10' });
  });

  it('distinguishes network failures, timeouts and server errors', () => {
    expect(apiError(err(null, undefined, {}, 'ERR_NETWORK'))).toMatchObject({ network: true, status: null });
    expect(apiError(err(null, undefined, {}, 'ECONNABORTED')).message).toMatch(/too long/);
    expect(apiError(err(500, '<html>'))).toMatchObject({ status: 500, message: 'The server encountered an error. Try again.' });
    expect(apiError(err(503, { detail: 'busy' }, { 'retry-after': '2' })).retryAfter).toBe(2);
  });
});

describe('refresh on 401', () => {
  afterEach(() => vi.restoreAllMocks());

  it('shares one refresh between concurrent callers', async () => {
    let resolve!: () => void;
    const post = vi.spyOn(axios, 'post').mockImplementation(() => new Promise((r) => { resolve = () => r({ data: {} }); }));
    const a = refreshSession();
    const b = refreshSession();
    resolve();
    expect(await a).toBe(true);
    expect(await b).toBe(true);
    expect(post).toHaveBeenCalledTimes(1);
  });

  it('calls the unauthorized handler (no page reload) when refresh fails', async () => {
    vi.spyOn(axios, 'post').mockRejectedValue(err(401));
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    const adapter = vi.fn((config: InternalAxiosRequestConfig) => Promise.reject(err(401, { detail: 'Not authenticated' }, {}, undefined, config)));
    await expect(api.get('/auth/me', { adapter })).rejects.toBeTruthy();
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('retries the original request after a successful refresh', async () => {
    vi.spyOn(axios, 'post').mockResolvedValue({ data: {} });
    let calls = 0;
    const adapter = vi.fn(async (config: InternalAxiosRequestConfig) => {
      calls++;
      if (calls === 1) throw err(401, undefined, {}, undefined, config);
      return { data: { ok: true }, status: 200, statusText: 'OK', headers: {}, config } as AxiosResponse;
    });
    const r = await api.get('/dashboard/summary', { adapter });
    expect(r.data).toEqual({ ok: true });
    expect(calls).toBe(2);
  });

  it('never tries to refresh for the login endpoint itself', async () => {
    const post = vi.spyOn(axios, 'post');
    const adapter = vi.fn((config: InternalAxiosRequestConfig) => Promise.reject(err(401, undefined, {}, undefined, config)));
    await expect(api.post('/auth/login', {}, { adapter })).rejects.toBeTruthy();
    expect(post).not.toHaveBeenCalled();
  });
});
