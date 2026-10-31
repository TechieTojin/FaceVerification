/**
 * HTTP client.
 *
 * - Cookies carry auth (HttpOnly fv_access / fv_refresh). No token ever touches JS storage.
 * - State-changing requests send X-CSRF-Token copied from the fv_csrf cookie.
 * - On 401 a single shared refresh is attempted; concurrent 401s wait for it.
 *   If refresh fails the auth store is cleared and routing (not a full page
 *   reload) sends the user to /login — no redirect loop on the login page.
 */
import axios, { AxiosError, type AxiosRequestConfig } from 'axios';

export const api = axios.create({ baseURL: '/api/v1', withCredentials: true, timeout: 30_000 });

export function readCookie(name: string): string {
  const m = document.cookie.match(new RegExp(`(?:^|;\\s*)${name}=([^;]+)`));
  return m ? decodeURIComponent(m[1]) : '';
}

api.interceptors.request.use((config) => {
  const method = (config.method ?? 'get').toLowerCase();
  if (!['get', 'head', 'options'].includes(method)) {
    const csrf = readCookie('fv_csrf');
    if (csrf) config.headers.set('X-CSRF-Token', csrf);
  }
  return config;
});

type Unauthorized = () => void;
let onUnauthorized: Unauthorized = () => {};
export function setUnauthorizedHandler(fn: Unauthorized) {
  onUnauthorized = fn;
}

let refreshing: Promise<boolean> | null = null;

export function refreshSession(): Promise<boolean> {
  if (!refreshing) {
    refreshing = axios
      .post('/api/v1/auth/refresh', {}, { withCredentials: true, timeout: 15_000 })
      .then(() => true)
      .catch(() => false)
      .finally(() => {
        refreshing = null;
      });
  }
  return refreshing;
}

type StepUpHandler = (level: 'standard' | 'face_and_factor') => Promise<boolean>;
let onStepUp: StepUpHandler | null = null;
/** Sensitive actions answer 403 STEP_UP_REQUIRED; the handler re-authenticates and the request is retried once. */
export function setStepUpHandler(fn: StepUpHandler | null) {
  onStepUp = fn;
}

const NO_REFRESH = ['/auth/login', '/auth/refresh', '/auth/logout'];

api.interceptors.response.use(
  (r) => r,
  async (error: AxiosError) => {
    const original = error.config as (AxiosRequestConfig & { _retried?: boolean; _steppedUp?: boolean }) | undefined;
    const url = original?.url ?? '';
    const body = error.response?.data as { code?: string; step_up_level?: 'standard' | 'face_and_factor' } | undefined;
    if (error.response?.status === 403 && body?.code === 'STEP_UP_REQUIRED' && original && !original._steppedUp && onStepUp) {
      original._steppedUp = true;
      if (await onStepUp(body.step_up_level ?? 'standard')) return api.request(original);
      return Promise.reject(error);
    }
    if (error.response?.status !== 401 || !original || original._retried || NO_REFRESH.some((p) => url.includes(p))) {
      return Promise.reject(error);
    }
    original._retried = true;
    const ok = await refreshSession();
    if (!ok) {
      onUnauthorized();
      return Promise.reject(error);
    }
    return api.request(original);
  },
);

export interface ApiErrorInfo {
  status: number | null;
  code: string | null;
  message: string;
  fieldErrors: Record<string, string>;
  network: boolean;
  retryAfter: number | null;
}

/** Normalise any thrown value into something the UI can display. */
export function apiError(err: unknown, fallback = 'Something went wrong'): ApiErrorInfo {
  const info: ApiErrorInfo = { status: null, code: null, message: fallback, fieldErrors: {}, network: false, retryAfter: null };
  if (!axios.isAxiosError(err)) {
    if (err instanceof Error && err.name !== 'CanceledError') info.message = err.message || fallback;
    return info;
  }
  if (!err.response) {
    info.network = err.code !== 'ERR_CANCELED';
    info.message = err.code === 'ECONNABORTED' ? 'The server took too long to respond' : 'Cannot reach the server — check your connection';
    if (err.code === 'ERR_CANCELED') info.message = 'Request cancelled';
    return info;
  }
  info.status = err.response.status;
  const ra = err.response.headers?.['retry-after'];
  info.retryAfter = ra ? Number(ra) || null : null;
  const data = err.response.data as { detail?: unknown; code?: string } | undefined;
  const detail = data?.detail;
  if (typeof data?.code === 'string') info.code = data.code;
  if (typeof detail === 'string') {
    info.message = detail;
  } else if (Array.isArray(detail)) {
    info.message = 'Some fields are invalid';
    for (const d of detail as { loc?: unknown[]; msg?: string }[]) {
      const key = String(d.loc?.[d.loc.length - 1] ?? 'form');
      info.fieldErrors[key] = (d.msg ?? 'Invalid').replace(/^Value error, /, '');
    }
  } else if (detail && typeof detail === 'object') {
    const d = detail as { message?: string; code?: string; errors?: Record<string, string> };
    if (d.message) info.message = d.message;
    if (d.code) info.code = d.code;
    if (d.errors) info.fieldErrors = d.errors;
  } else if (err.response.status >= 500) {
    info.message = 'The server encountered an error. Try again.';
  }
  return info;
}
