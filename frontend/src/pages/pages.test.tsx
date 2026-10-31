import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { AxiosError, AxiosHeaders, type InternalAxiosRequestConfig } from 'axios';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ReactNode } from 'react';
import { useAuth } from '../store/auth';

const post = vi.fn();
const get = vi.fn();
const put = vi.fn();
vi.mock('../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../lib/api')>('../lib/api');
  return { ...actual, api: { post: (...a: unknown[]) => post(...a), get: (...a: unknown[]) => get(...a), put: (...a: unknown[]) => put(...a) } };
});

const LoginPage = (await import('./Login')).default;
const SettingsPage = (await import('./Settings')).default;

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}><MemoryRouter>{node}</MemoryRouter></QueryClientProvider>);
}

function httpError(status: number, data: unknown) {
  const config = { headers: new AxiosHeaders() } as InternalAxiosRequestConfig;
  return new AxiosError('x', undefined, config, undefined, { status, data, headers: {}, config, statusText: '' });
}

beforeEach(() => {
  post.mockReset();
  get.mockReset();
  put.mockReset();
  useAuth.getState().clear();
});

describe('Login', () => {
  it('validates empty input client-side', async () => {
    wrap(<LoginPage />);
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(screen.getByText('Enter your email and password.')).toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });

  it('shows a generic message for wrong credentials and stores the user on success', async () => {
    post.mockRejectedValueOnce(httpError(401, { detail: 'Invalid credentials' }));
    wrap(<LoginPage />);
    await userEvent.type(screen.getByLabelText('Email'), 'a@b.co');
    await userEvent.type(screen.getByLabelText('Password'), 'pw');
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(await screen.findByText('Incorrect email or password.')).toBeInTheDocument();
    post.mockResolvedValueOnce({ data: { status: 'ok', user: { id: '1', email: 'a@b.co', full_name: 'A', role: 'ADMIN', permissions: [] } } });
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    await waitFor(() => expect(useAuth.getState().status).toBe('authenticated'));
  });

  it('asks for a second factor, refuses a wrong code and signs in with a recovery code', async () => {
    post.mockResolvedValueOnce({ data: { status: 'mfa_required', pending_token: 'pending-123', methods: ['totp', 'recovery_code'], enrol_required: false } });
    wrap(<LoginPage />);
    await userEvent.type(screen.getByLabelText('Email'), 'a@b.co');
    await userEvent.type(screen.getByLabelText('Password'), 'pw');
    await userEvent.click(screen.getByRole('button', { name: 'Sign in', exact: true }));
    expect(await screen.findByRole('heading', { name: 'Two-step verification' })).toBeInTheDocument();
    expect(useAuth.getState().status).toBe('anonymous');                 // no session yet
    post.mockRejectedValueOnce(httpError(401, { detail: 'Verification failed. Check the code and try again.', code: 'INVALID_CODE' }));
    await userEvent.type(screen.getByLabelText('6-digit code'), '000000');
    await userEvent.click(screen.getByRole('button', { name: 'Verify' }));
    expect(await screen.findByText('Verification failed. Check the code and try again.')).toBeInTheDocument();
    expect(post).toHaveBeenLastCalledWith('/auth/mfa/verify', { pending_token: 'pending-123', method: 'totp', code: '000000' });
    await userEvent.click(screen.getByRole('tab', { name: 'Recovery code' }));
    post.mockResolvedValueOnce({ data: { status: 'ok', user: { id: '1', email: 'a@b.co', full_name: 'A', role: 'ADMIN', permissions: [] } } });
    await userEvent.type(screen.getByLabelText('Recovery code'), 'abcd-efgh-jk');
    await userEvent.click(screen.getByRole('button', { name: 'Verify' }));
    await waitFor(() => expect(useAuth.getState().status).toBe('authenticated'));
    expect(post).toHaveBeenLastCalledWith('/auth/mfa/verify', { pending_token: 'pending-123', method: 'recovery_code', code: 'abcd-efgh-jk' });
  });

  it('shows the lockout message from the server', async () => {
    post.mockRejectedValueOnce(httpError(429, { detail: 'Account temporarily locked. Try again in 120s.' }));
    wrap(<LoginPage />);
    await userEvent.type(screen.getByLabelText('Email'), 'a@b.co');
    await userEvent.type(screen.getByLabelText('Password'), 'pw');
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(await screen.findByText(/temporarily locked/)).toBeInTheDocument();
  });
});

describe('Settings', () => {
  const policy = {
    version: 3, updated_at: null, active_sessions_note: 'Sessions keep their snapshot.',
    values: { max_attempts: 3, passive_liveness_required: true },
    specs: [
      { key: 'max_attempts', kind: 'int', default: 3, group: 'matching', label: 'Maximum attempts', description: '', min: 1, max: 10,
        choices: [], unit: '', sensitive: true, weakening: 'increase' },
      { key: 'passive_liveness_required', kind: 'bool', default: true, group: 'liveness', label: 'Passive anti-spoofing',
        description: '', min: null, max: null, choices: [], unit: '', sensitive: true, weakening: 'disable' },
    ],
  };

  beforeEach(() => {
    useAuth.getState().setUser({ id: '1', email: 'a@b.co', full_name: 'A', role: 'ADMIN', is_active: true, created_at: '',
      last_login_at: null, password_changed_at: null, permissions: ['policy:read', 'policy:manage'] });
    get.mockImplementation(async (url: string) => (url === '/policy' ? { data: policy } : { data: { versions: [] } }));
  });

  it('rejects out-of-range values before saving', async () => {
    wrap(<SettingsPage />);
    const input = await screen.findByLabelText(/Maximum attempts/);
    await userEvent.clear(input);
    await userEvent.type(input, '42');
    expect(screen.getByText('Maximum 10')).toBeInTheDocument();
    expect(screen.getByTestId('save-policy')).toBeDisabled();
  });

  it('asks for confirmation and a reason for a weakening change', async () => {
    put.mockResolvedValue({ data: { version: 4 } });
    wrap(<SettingsPage />);
    await userEvent.click(await screen.findByRole('switch', { name: /Passive anti-spoofing/ }));
    await userEvent.click(screen.getByTestId('save-policy'));
    expect(await screen.findByText('Confirm security-weakening changes')).toBeInTheDocument();
    const confirm = screen.getByRole('button', { name: 'Confirm and save' });
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/Reason/), 'Testing');
    await userEvent.click(confirm);
    await waitFor(() => expect(put).toHaveBeenCalledWith('/policy', {
      changes: { passive_liveness_required: false }, confirm: true, reason: 'Testing' }));
  });
});
