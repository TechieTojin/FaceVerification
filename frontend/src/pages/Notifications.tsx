import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Bell, CheckCheck } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { humanize } from '../lib/format';
import type { NotificationList, NotificationPref, NotificationRow } from '../lib/types';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Callout, Card, CardHeader, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Pagination } from '../components/ui/Table';
import { Tabs } from '../components/ui/Tabs';
import { NotificationItem } from '../components/NotificationBell';

export default function Notifications() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<'all' | 'unread'>('all');
  const [page, setPage] = useState(1);
  const list = useApi<NotificationList>('/notifications', { unread_only: filter === 'unread', page, page_size: 20 });
  const prefsQ = useApi<{ preferences: Record<string, NotificationPref>; email_available: boolean }>('/notifications/preferences');
  const [prefs, setPrefs] = useState<Record<string, NotificationPref> | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => { if (prefsQ.data) setPrefs(prefsQ.data.preferences); }, [prefsQ.data]);

  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ['/notifications'] });
    void qc.invalidateQueries({ queryKey: ['/notifications/unread-count'] });
  };
  const open = async (n: NotificationRow) => {
    if (!n.read) await api.post(`/notifications/${n.id}/read`).catch(() => undefined);
    refresh();
    if (n.link) navigate(n.link);
  };
  const save = async () => {
    if (!prefs) return;
    setSaving(true);
    try {
      await api.put('/notifications/preferences', {
        preferences: Object.fromEntries(Object.entries(prefs).map(([k, v]) => [k, { in_app: v.in_app, email: v.email }])),
      });
      toast.success('Preferences saved');
      void qc.invalidateQueries({ queryKey: ['/notifications/preferences'] });
    } catch (err) {
      toast.error('Could not save', apiError(err).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <PageHeader title="Notifications" description="Events that concern you in this organisation. Notifications never contain biometric data."
        actions={<Button variant="outline" icon={<CheckCheck className="h-4 w-4" />}
          onClick={() => void api.post('/notifications/read-all').then(refresh)}>Mark all read</Button>} />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Card>
          <div className="flex items-center justify-between gap-3 border-b border-line p-4">
            <Tabs label="Filter" value={filter} onChange={(v) => { setFilter(v); setPage(1); }}
              items={[{ value: 'all', label: 'All' }, { value: 'unread', label: 'Unread', count: list.data?.unread }]} />
          </div>
          {list.isLoading ? <LoadingBlock rows={5} /> : !list.data?.notifications.length
            ? <EmptyState icon={<Bell className="h-5 w-5" />} title={filter === 'unread' ? 'No unread notifications' : 'No notifications yet'} />
            : (
              <>
                <div className="divide-y divide-line">{list.data.notifications.map((n) => <NotificationItem key={n.id} n={n} onOpen={(x) => void open(x)} />)}</div>
                <Pagination page={page} pageSize={20} total={list.data.total} onPage={setPage} />
              </>
            )}
        </Card>
        <Card>
          <CardHeader title="Preferences" subtitle="Choose where each kind of notification reaches you." />
          {!prefs ? <LoadingBlock rows={4} /> : (
            <div className="p-4">
              {!prefsQ.data?.email_available && (
                <Callout tone="neutral" className="mb-3">Email delivery is unavailable: no mail server is configured. Email choices are saved
                  and take effect once one is configured.</Callout>
              )}
              <table className="w-full text-sm" aria-label="Notification preferences">
                <thead><tr className="text-left text-xs text-muted"><th className="py-2 font-medium">Event</th><th className="w-16 text-center font-medium">In app</th><th className="w-16 text-center font-medium">Email</th></tr></thead>
                <tbody className="divide-y divide-line">
                  {Object.entries(prefs).map(([type, p]) => (
                    <tr key={type}>
                      <td className="py-2 pr-2 text-ink-2">{p.label}{p.mandatory && <span className="block text-[11px] text-muted">Always shown</span>}</td>
                      <td className="text-center"><input type="checkbox" aria-label={`${p.label} in app`} checked={p.in_app} disabled={p.mandatory}
                        onChange={(e) => setPrefs({ ...prefs, [type]: { ...p, in_app: e.target.checked } })} /></td>
                      <td className="text-center"><input type="checkbox" aria-label={`${p.label} by email`} checked={p.email}
                        onChange={(e) => setPrefs({ ...prefs, [type]: { ...p, email: e.target.checked } })} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-4 flex justify-end"><Button loading={saving} onClick={() => void save()} data-testid="save-preferences">Save preferences</Button></div>
              <p className="mt-2 text-[11px] text-muted">{humanize('ACCOUNT_SECURITY')} notices cannot be turned off in the app.</p>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
