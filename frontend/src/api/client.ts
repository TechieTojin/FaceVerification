/**
 * Axios API client — v2.1
 *
 * Authentication:
 *   - withCredentials: true  → browser sends fv_access cookie automatically
 *   - No token stored in memory or localStorage
 *
 * CSRF (double-submit cookie):
 *   - Backend sets fv_csrf cookie (non-HttpOnly, JS-readable)
 *   - Client reads it and sends as X-CSRF-Token on state-changing requests
 *   - Backend middleware validates header === cookie
 *
 * Expired access token:
 *   - On 401, attempt one silent refresh via POST /auth/refresh (uses fv_refresh cookie)
 *   - If refresh succeeds, retry the original request
 *   - If refresh fails, clear local state and redirect to /login
 */
import axios from 'axios';
import { useAuthStore } from '../store/authStore';

const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
  withCredentials: true,   // always send auth cookies
});

function getCsrfToken(): string {
  const match = document.cookie.match(/(?:^|;\s*)fv_csrf=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : '';
}

// Attach CSRF token to state-changing requests
api.interceptors.request.use((config) => {
  const method = (config.method ?? '').toLowerCase();
  if (!['get', 'head', 'options'].includes(method)) {
    const csrf = getCsrfToken();
    if (csrf) {
      config.headers['X-CSRF-Token'] = csrf;
    }
  }
  return config;
});

let _refreshing = false;
let _refreshQueue: Array<(ok: boolean) => void> = [];

function _drainQueue(ok: boolean) {
  _refreshQueue.forEach((resolve) => resolve(ok));
  _refreshQueue = [];
}

// Silent refresh on 401
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;

    if (error.response?.status !== 401 || original._retried) {
      return Promise.reject(error);
    }

    // Don't try to refresh if the failing request IS the refresh or login endpoint
    if (
      original.url?.includes('/auth/refresh') ||
      original.url?.includes('/auth/login')
    ) {
      useAuthStore.getState().logout();
      window.location.href = '/login';
      return Promise.reject(error);
    }

    if (_refreshing) {
      // Queue the retry until the in-flight refresh resolves
      return new Promise((resolve, reject) => {
        _refreshQueue.push((ok) => {
          if (ok) {
            original._retried = true;
            resolve(api.request(original));
          } else {
            reject(error);
          }
        });
      });
    }

    _refreshing = true;
    original._retried = true;

    try {
      await axios.post('/api/v1/auth/refresh', {}, { withCredentials: true });
      _drainQueue(true);
      return api.request(original);
    } catch {
      _drainQueue(false);
      useAuthStore.getState().logout();
      window.location.href = '/login';
      return Promise.reject(error);
    } finally {
      _refreshing = false;
    }
  }
);

export default api;
