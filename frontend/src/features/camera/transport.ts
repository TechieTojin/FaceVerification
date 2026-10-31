/**
 * How a capture loop talks to the service that owns its session.
 *
 * - `operatorTransport`: signed-in operators (HttpOnly cookies + CSRF) → /api/v1/sessions
 * - header transports: principals without a sign-in — a paired phone (`X-Pairing-Secret`) or a
 *   pending passkey sign-in / step-up doing its face check (`X-Login-Token`). Credentials stay in memory.
 *   Same-origin requests still carry any cookies the browser has, so the CSRF header is attached too.
 * - `kioskTransport`: the offline kiosk agent's local API (/kiosk/api), no central server involved.
 */
import axios, { type AxiosInstance } from 'axios';
import { api, readCookie } from '../../lib/api';

export interface CaptureTransport {
  http: AxiosInstance;
  /** path prefix for session resources, e.g. '/sessions' */
  base: string;
  /** whether the service supports POST {base}/{id}/resync */
  resync: boolean;
}

export const operatorTransport: CaptureTransport = { http: api, base: '/sessions', resync: true };

type HeaderName = 'X-Pairing-Secret' | 'X-Login-Token';

function headerClient(header: HeaderName, value: () => string): AxiosInstance {
  const http = axios.create({ baseURL: '/api/v1', timeout: 30_000 });
  http.interceptors.request.use((config) => {
    const v = value();
    if (v) config.headers.set(header, v);
    if (!['get', 'head', 'options'].includes((config.method ?? 'get').toLowerCase())) {
      const csrf = readCookie('fv_csrf');
      if (csrf) config.headers.set('X-CSRF-Token', csrf);
    }
    return config;
  });
  return http;
}

export function headerTransport(header: HeaderName, value: string): CaptureTransport {
  return { http: headerClient(header, () => value), base: '/sessions', resync: true };
}

/** A header transport whose credential arrives later (e.g. in the response that creates the session). */
export function deferredHeaderTransport(header: HeaderName) {
  const holder = { value: '' };
  return {
    transport: { http: headerClient(header, () => holder.value), base: '/sessions', resync: true } as CaptureTransport,
    set: (v: string) => { holder.value = v; },
  };
}

export const kioskHttp = axios.create({ baseURL: '/kiosk/api', timeout: 60_000 });
export const kioskTransport: CaptureTransport = { http: kioskHttp, base: '/sessions', resync: false };
