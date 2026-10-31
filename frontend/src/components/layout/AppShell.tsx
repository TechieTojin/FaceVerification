import { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Fingerprint, LogOut, Menu, Monitor, Moon, ShieldCheck, Sun, UserCircle2, X } from 'lucide-react';
import { api, setStepUpHandler } from '../../lib/api';
import { cn } from '../../lib/cn';
import { useEventStream, type StreamState } from '../../lib/sse';
import { useTheme, type ThemeChoice } from '../../lib/theme';
import { useAuth } from '../../store/auth';
import { useStepUp } from '../../store/stepUp';
import { toast } from '../../store/toast';
import { NotificationBell } from '../NotificationBell';
import { OrgSwitcher } from '../OrgSwitcher';
import { StepUpDialog } from '../StepUpDialog';
import { NAV_SECTIONS } from './nav';

function Brand() {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-white"><ShieldCheck className="h-4.5 w-4.5" /></span>
      <div className="leading-tight">
        <p className="text-sm font-semibold text-ink">FaceVerification</p>
        <p className="text-[11px] text-muted">Identity platform 4.0</p>
      </div>
    </div>
  );
}

function SideNav({ onNavigate }: { onNavigate?: () => void }) {
  const user = useAuth((s) => s.user);
  const perms = new Set(user?.permissions ?? []);
  return (
    <nav className="flex-1 space-y-5 overflow-y-auto px-3 py-4" aria-label="Main">
      {NAV_SECTIONS.map((sec) => {
        const items = sec.items.filter((i) => i.any.some((p) => perms.has(p)));
        if (!items.length) return null;
        return (
          <div key={sec.title}>
            <p className="px-2 pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted">{sec.title}</p>
            <ul className="space-y-0.5">
              {items.map(({ to, label, icon: Icon }) => (
                <li key={to}>
                  <NavLink to={to} end={to === '/' || to === '/verify'} onClick={onNavigate}
                    className={({ isActive }) => cn('flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors',
                      isActive ? 'bg-primary-soft font-medium text-primary-strong' : 'text-ink-2 hover:bg-surface-2 hover:text-ink')}>
                    <Icon className="h-4 w-4 shrink-0" aria-hidden />
                    {label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </nav>
  );
}

const LINK_LABEL: Record<StreamState, string> = { connecting: 'Connecting', live: 'Live', stale: 'Stale', offline: 'Offline' };

function LiveIndicator({ state }: { state: StreamState }) {
  const color = state === 'live' ? 'bg-good' : state === 'stale' ? 'bg-warning' : state === 'offline' ? 'bg-critical' : 'bg-muted';
  return (
    <span className="hidden items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-ink-2 sm:inline-flex"
      title="Real-time update stream" data-testid="live-indicator">
      <span className={cn('h-2 w-2 rounded-full', color, state === 'live' && 'animate-pulse2')} aria-hidden />
      {LINK_LABEL[state]}
    </span>
  );
}

const THEMES: { value: ThemeChoice; icon: typeof Sun; label: string }[] = [
  { value: 'light', icon: Sun, label: 'Light' },
  { value: 'dark', icon: Moon, label: 'Dark' },
  { value: 'system', icon: Monitor, label: 'System' },
];

export function AppShell() {
  const [open, setOpen] = useState(false);
  const [menu, setMenu] = useState(false);
  const [theme, setTheme] = useTheme();
  const user = useAuth((s) => s.user);
  const clear = useAuth((s) => s.clear);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const location = useLocation();

  useEffect(() => { setOpen(false); setMenu(false); }, [location.pathname]);
  useEffect(() => {
    setStepUpHandler((level) => useStepUp.getState().request(level));
    return () => setStepUpHandler(null);
  }, []);

  const perms = user?.permissions ?? [];
  const channels = ['sessions', 'notifications', 'pairing',
    ...(perms.includes('security:read') ? ['alerts', 'enrolment'] : []),
    ...(perms.includes('review:read') ? ['reviews'] : []),
    ...(perms.includes('devices:read') ? ['devices'] : [])];
  const { state } = useEventStream(channels, (m) => {
    if (m.event === 'notification') {
      void qc.invalidateQueries({ queryKey: ['/notifications/unread-count'] });
      void qc.invalidateQueries({ queryKey: ['/notifications'] });
      const d = m.data as { title?: string; severity?: string };
      if (d.severity === 'WARNING' || d.severity === 'CRITICAL') toast.info(String(d.title ?? 'New notification'));
    }
    if (m.event.startsWith('review_')) void qc.invalidateQueries({ queryKey: ['/reviews'] });
    if (m.event === 'device_updated' || m.event === 'offline_synced') void qc.invalidateQueries({ queryKey: ['/devices'] });
    if (m.event === 'session_closed' || m.event === 'session_created') {
      void qc.invalidateQueries({ queryKey: ['/dashboard/summary'] });
    }
    if (m.event === 'alert' && (m.data as { created?: boolean }).created) {
      toast.error('Security alert', String((m.data as { title?: string }).title ?? ''));
      void qc.invalidateQueries({ queryKey: ['/security/alerts'] });
    }
    if (m.event === 'enrolment_changed') void qc.invalidateQueries({ queryKey: ['/enrolments/requests'] });
  }, !!user);

  const logout = async () => {
    try { await api.post('/auth/logout'); } catch { /* cookies are cleared server-side when reachable */ }
    clear();
    qc.clear();
    navigate('/login', { replace: true });
  };

  const ThemeIcon = THEMES.find((t) => t.value === theme)?.icon ?? Monitor;

  return (
    <div className="flex h-full">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-line bg-surface lg:flex">
        <div className="flex h-14 items-center border-b border-line px-3"><Brand /></div>
        <SideNav />
      </aside>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} aria-hidden />
          <aside className="absolute inset-y-0 left-0 flex w-72 animate-fade-in flex-col bg-surface shadow-pop">
            <div className="flex h-14 items-center justify-between border-b border-line px-3">
              <Brand />
              <button className="rounded-md p-1.5 text-muted hover:bg-surface-2" onClick={() => setOpen(false)} aria-label="Close navigation">
                <X className="h-5 w-5" />
              </button>
            </div>
            <SideNav onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-line bg-surface/90 px-4 backdrop-blur sm:px-6">
          <div className="flex min-w-0 items-center gap-2">
            <button className="rounded-md p-1.5 text-ink-2 hover:bg-surface-2 lg:hidden" onClick={() => setOpen(true)} aria-label="Open navigation">
              <Menu className="h-5 w-5" />
            </button>
            <div className="hidden sm:block lg:hidden"><Brand /></div>
            <OrgSwitcher />
          </div>
          <div className="flex shrink-0 items-center gap-1 sm:gap-2">
            <LiveIndicator state={state} />
            <NotificationBell />
            <button className="rounded-md p-2 text-ink-2 hover:bg-surface-2" aria-label={`Theme: ${theme}. Change theme`}
              onClick={() => setTheme(THEMES[(THEMES.findIndex((t) => t.value === theme) + 1) % THEMES.length].value)}>
              <ThemeIcon className="h-4 w-4" />
            </button>
            <div className="relative">
              <button className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-surface-2" onClick={() => setMenu((m) => !m)}
                aria-haspopup="menu" aria-expanded={menu} data-testid="user-menu">
                <UserCircle2 className="h-5 w-5 text-muted" />
                <span className="hidden max-w-[180px] truncate text-ink sm:inline">{user?.full_name}</span>
                <span className="rounded-md bg-surface-2 px-1.5 py-0.5 text-[11px] font-medium text-ink-2">{user?.role}</span>
              </button>
              {menu && (
                <div role="menu" className="absolute right-0 z-30 mt-1 w-56 animate-fade-in rounded-xl bg-surface p-1 shadow-pop">
                  <p className="truncate px-3 py-2 text-xs text-muted">{user?.email}</p>
                  <button role="menuitem" className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-ink hover:bg-surface-2"
                    onClick={() => navigate('/profile')}>
                    <UserCircle2 className="h-4 w-4" /> Profile & password
                  </button>
                  <button role="menuitem" className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-ink hover:bg-surface-2"
                    onClick={() => navigate('/account/security')}>
                    <Fingerprint className="h-4 w-4" /> Passkeys & two-step
                  </button>
                  <button role="menuitem" className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-critical hover:bg-surface-2"
                    onClick={logout}>
                    <LogOut className="h-4 w-4" /> Sign out
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>
        <main className="min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
            <Outlet />
          </div>
        </main>
        <StepUpDialog />
      </div>
    </div>
  );
}
