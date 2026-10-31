import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Me, Permission } from '../lib/types';
import { useAuth } from '../store/auth';
import { ChallengeSteps } from '../features/camera/components';
import { ConfirmDialog } from './ui/Modal';
import { Badge } from './ui/primitives';

vi.mock('../lib/sse', () => ({ useEventStream: () => ({ state: 'live', lastEventAt: null }) }));
const { AppShell } = await import('./layout/AppShell');
const shell = () => <QueryClientProvider client={new QueryClient()}><MemoryRouter><AppShell /></MemoryRouter></QueryClientProvider>;

function me(role: Me['role'], permissions: Permission[]): Me {
  return { id: '1', email: 'x@y.z', full_name: 'Test Person', role, is_active: true, created_at: '', last_login_at: null,
    password_changed_at: null, permissions, is_platform_admin: false, amr: ['pwd'],
    organization: { id: 'o1', name: 'Acme', slug: 'acme' }, organizations: [{ id: 'o1', name: 'Acme', slug: 'acme', role: role ?? undefined }] };
}

describe('navigation is filtered by server-provided permissions', () => {
  beforeEach(() => useAuth.getState().clear());

  it('viewer sees read-only modules only', () => {
    useAuth.getState().setUser(me('VIEWER', ['dashboard:view', 'history:read_all', 'analytics:read', 'org:read', 'review:read']));
    render(shell());
    const nav = screen.getAllByRole('navigation', { name: 'Main' })[0];
    expect(nav).toHaveTextContent('Dashboard');
    expect(nav).toHaveTextContent('Analytics');
    expect(nav).toHaveTextContent('Review queue');
    expect(nav).not.toHaveTextContent('QR verification');
    expect(nav).not.toHaveTextContent('Developers');
    expect(screen.getByTestId('org-switcher')).toHaveTextContent('Acme');
    expect(nav).not.toHaveTextContent('Live verification');
    expect(nav).not.toHaveTextContent('Users');
    expect(nav).not.toHaveTextContent('Policy settings');
  });

  it('operator sees capture modules but not administration', () => {
    useAuth.getState().setUser(me('OPERATOR', ['dashboard:view', 'verification:run', 'enrolment:capture', 'enrolment:read', 'history:read_own']));
    render(shell());
    const nav = screen.getAllByRole('navigation', { name: 'Main' })[0];
    expect(nav).toHaveTextContent('Live verification');
    expect(nav).toHaveTextContent('QR verification');
    expect(nav).toHaveTextContent('Enrolment approvals');
    expect(nav).not.toHaveTextContent('Security log');
    expect(nav).not.toHaveTextContent('System health');
    expect(nav).not.toHaveTextContent('Review queue');
    expect(nav).not.toHaveTextContent('Devices & kiosks');
  });
});

describe('ChallengeSteps', () => {
  it('marks done, current and pending steps', () => {
    render(<ChallengeSteps stepProgress={0.5} challenge={{ round: 0, index: 1, total: 4, type: 'TURN_LEFT', instruction: '',
      steps: ['LOOK_STRAIGHT', 'TURN_LEFT', 'BLINK', 'LOOK_STRAIGHT'], progress: 0.25 }} />);
    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(4);
    expect(items[0]).toHaveTextContent('Done');
    expect(items[1]).toHaveAttribute('aria-current', 'step');
    expect(items[1]).toHaveTextContent('Turn left');
  });
});

describe('ConfirmDialog', () => {
  it('requires the typed confirmation and a note before confirming', async () => {
    const onConfirm = vi.fn();
    render(<ConfirmDialog open onClose={() => {}} onConfirm={onConfirm} title="Revoke?" confirmLabel="Revoke"
      noteLabel="Reason" noteRequired typeToConfirm="DELETE" />);
    const btn = screen.getByRole('button', { name: 'Revoke' });
    expect(btn).toBeDisabled();
    await userEvent.type(screen.getByLabelText('Reason'), 'left');
    expect(btn).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/to confirm/), 'DELETE');
    expect(btn).toBeEnabled();
    await userEvent.click(btn);
    expect(onConfirm).toHaveBeenCalledWith('left');
  });

  it('closes on Escape', async () => {
    const onClose = vi.fn();
    render(<ConfirmDialog open onClose={onClose} onConfirm={() => {}} title="X" />);
    await userEvent.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalled();
  });
});

describe('Badge', () => {
  it('never relies on colour alone (icon + text)', () => {
    const { container } = render(<Badge tone="bad">Not matched</Badge>);
    expect(container.querySelector('svg')).not.toBeNull();
    expect(container).toHaveTextContent('Not matched');
  });
});
