import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Bell, CheckCheck } from 'lucide-react';
import { api } from '../lib/api';
import { cn } from '../lib/cn';
import { fmtRelative } from '../lib/format';
import { useApi } from '../lib/query';
import type { NotificationList, NotificationRow } from '../lib/types';
import { Button } from './ui/Button';
import { EmptyState, Spinner } from './ui/primitives';

const SEVERITY_DOT: Record<string, string> = { INFO: 'bg-primary', WARNING: 'bg-warning', CRITICAL: 'bg-critical' };

export function NotificationItem({ n, onOpen }: { n: NotificationRow; onOpen: (n: NotificationRow) => void }) {
  return (
    <button type="button" onClick={() => onOpen(n)} data-testid="notification-item"
      className={cn('flex w-full gap-3 px-4 py-3 text-left transition-colors hover:bg-surface-2', !n.read && 'bg-primary-soft/30')}>
      <span className={cn('mt-1.5 h-2 w-2 shrink-0 rounded-full', n.read ? 'bg-line' : SEVERITY_DOT[n.severity] ?? 'bg-primary')} aria-hidden />
      <span className="min-w-0 flex-1">
        <span className={cn('block truncate text-sm', n.read ? 'text-ink-2' : 'font-medium text-ink')}>{n.title}</span>
        {n.body && <span className="line-clamp-2 block text-xs text-muted">{n.body}</span>}
        <span className="mt-0.5 block text-[11px] text-muted">{fmtRelative(n.created_at)}{!n.read && <span className="sr-only"> · unread</span>}</span>
      </span>
    </button>
  );
}

/** Header bell: unread counter (pushed live over SSE) and a drawer with the latest notifications. */
export function NotificationBell() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const count = useApi<{ unread: number }>('/notifications/unread-count', undefined, { refetchInterval: 60_000 });
  const list = useApi<NotificationList>(open ? '/notifications' : null, { page_size: 10 });
  const unread = count.data?.unread ?? 0;

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ['/notifications/unread-count'] });
    void qc.invalidateQueries({ queryKey: ['/notifications'] });
  };
  const openItem = async (n: NotificationRow) => {
    if (!n.read) await api.post(`/notifications/${n.id}/read`).catch(() => undefined);
    refresh();
    setOpen(false);
    if (n.link) navigate(n.link);
  };
  const markAll = async () => {
    await api.post('/notifications/read-all').catch(() => undefined);
    refresh();
  };

  return (
    <div className="relative" ref={ref}>
      <button className="relative rounded-md p-2 text-ink-2 hover:bg-surface-2" aria-haspopup="dialog" aria-expanded={open}
        aria-label={unread ? `Notifications, ${unread} unread` : 'Notifications'} onClick={() => setOpen((o) => !o)}
        data-testid="notification-bell">
        <Bell className="h-4 w-4" />
        {unread > 0 && (
          <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-critical px-1 text-[10px] font-semibold text-white"
            data-testid="unread-count">{unread > 99 ? '99+' : unread}</span>
        )}
      </button>
      {open && (
        <div role="dialog" aria-label="Notifications"
          className="absolute right-0 z-30 mt-1 w-[min(22rem,calc(100vw-2rem))] animate-fade-in overflow-hidden rounded-xl bg-surface shadow-pop">
          <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
            <p className="text-sm font-semibold text-ink">Notifications</p>
            <Button size="sm" variant="ghost" icon={<CheckCheck className="h-3.5 w-3.5" />} disabled={!unread} onClick={() => void markAll()}>
              Mark all read
            </Button>
          </div>
          <div className="max-h-[60vh] divide-y divide-line overflow-y-auto">
            {list.isLoading ? <div className="flex justify-center p-6"><Spinner /></div>
              : !list.data?.notifications.length ? <EmptyState icon={<Bell className="h-5 w-5" />} title="You're all caught up" />
                : list.data.notifications.map((n) => <NotificationItem key={n.id} n={n} onOpen={(x) => void openItem(x)} />)}
          </div>
          <Link to="/notifications" onClick={() => setOpen(false)}
            className="block border-t border-line px-4 py-2.5 text-center text-xs font-medium text-primary hover:bg-surface-2">
            All notifications & preferences
          </Link>
        </div>
      )}
    </div>
  );
}
