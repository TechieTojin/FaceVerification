import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { BellRing } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtRelative, humanize } from '../lib/format';
import type { Alert } from '../lib/types';
import { useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Select } from '../components/ui/form';
import { ConfirmDialog } from '../components/ui/Modal';
import { Pagination } from '../components/ui/Table';
import { Tabs } from '../components/ui/Tabs';

const SEV_TONE = { LOW: 'neutral', MEDIUM: 'warn', HIGH: 'bad', CRITICAL: 'bad' } as const;
type StatusTab = 'UNRESOLVED' | 'OPEN' | 'ACKNOWLEDGED' | 'RESOLVED';

export default function Alerts() {
  const can = useCan();
  const qc = useQueryClient();
  const [status, setStatus] = useState<StatusTab>('UNRESOLVED');
  const [severity, setSeverity] = useState('');
  const [page, setPage] = useState(1);
  const [act, setAct] = useState<{ alert: Alert; kind: 'acknowledge' | 'resolve' } | null>(null);
  const [busy, setBusy] = useState(false);
  const q = useApi<{ total: number; counts: Record<string, number>; alerts: Alert[] }>('/security/alerts',
    { status, severity, page, page_size: 25 }, { refetchInterval: 30_000 });
  const manage = can('alerts:manage');

  const run = async (note: string) => {
    if (!act) return;
    setBusy(true);
    try {
      await api.post(`/security/alerts/${act.alert.id}/${act.kind}`, { note });
      toast.success(act.kind === 'resolve' ? 'Alert resolved' : 'Alert acknowledged');
      setAct(null);
      void qc.invalidateQueries({ queryKey: ['/security/alerts'] });
    } catch (err) {
      toast.error('Update failed', apiError(err).message);
    } finally {
      setBusy(false);
    }
  };
  const c = q.data?.counts ?? {};

  return (
    <>
      <PageHeader title="Security alerts" description="Raised automatically by server-side rules. Thresholds are configured in Policy settings." />
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <Tabs label="Alert status" value={status} onChange={(v) => { setStatus(v); setPage(1); }} items={[
          { value: 'UNRESOLVED', label: 'Unresolved', count: (c.OPEN ?? 0) + (c.ACKNOWLEDGED ?? 0) },
          { value: 'OPEN', label: 'Open', count: c.OPEN ?? 0 },
          { value: 'ACKNOWLEDGED', label: 'Acknowledged', count: c.ACKNOWLEDGED ?? 0 },
          { value: 'RESOLVED', label: 'Resolved', count: c.RESOLVED ?? 0 },
        ]} />
        <Select aria-label="Severity" value={severity} onChange={(e) => { setSeverity(e.target.value); setPage(1); }} className="w-40">
          <option value="">All severities</option><option value="CRITICAL">Critical</option><option value="HIGH">High</option><option value="MEDIUM">Medium</option><option value="LOW">Low</option>
        </Select>
      </div>
      <Card>
        {q.isLoading ? <LoadingBlock rows={5} /> : q.isError ? <Callout tone="bad" className="m-4">{apiError(q.error).message}</Callout> :
          !q.data?.alerts.length ? <EmptyState icon={<BellRing className="h-5 w-5" />} title="No alerts" body="Nothing matches this filter." /> : (
            <>
              <ul className="divide-y divide-line" aria-label="Alerts">
                {q.data.alerts.map((a) => (
                  <li key={a.id} className="flex flex-wrap items-start justify-between gap-4 px-5 py-4" data-testid="alert-row">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone={SEV_TONE[a.severity]}>{humanize(a.severity)}</Badge>
                        <p className="text-sm font-medium text-ink">{a.title}</p>
                        {a.occurrence_count > 1 && <span className="rounded bg-surface-2 px-1.5 text-xs text-ink-2">×{a.occurrence_count}</span>}
                      </div>
                      <p className="mt-1 text-sm text-ink-2">{a.description}</p>
                      <p className="mt-1 text-xs text-muted">
                        {a.subject_label ? `Subject: ${a.subject_label} · ` : ''}First {fmtDateTime(a.first_seen_at)} · last {fmtRelative(a.last_seen_at)}
                        {a.acknowledged_by ? ` · acknowledged by ${a.acknowledged_by}` : ''}
                      </p>
                      {a.resolution_note && <p className="mt-1 text-xs text-ink-2">Resolution ({a.resolved_by}): {a.resolution_note}</p>}
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge tone={a.status === 'RESOLVED' ? 'good' : a.status === 'ACKNOWLEDGED' ? 'info' : 'warn'}>{humanize(a.status)}</Badge>
                      {manage && a.status === 'OPEN' && <Button size="sm" variant="outline" onClick={() => setAct({ alert: a, kind: 'acknowledge' })}>Acknowledge</Button>}
                      {manage && a.status !== 'RESOLVED' && <Button size="sm" onClick={() => setAct({ alert: a, kind: 'resolve' })}>Resolve</Button>}
                    </div>
                  </li>
                ))}
              </ul>
              <Pagination page={page} pageSize={25} total={q.data.total} onPage={setPage} />
            </>
          )}
      </Card>
      <ConfirmDialog open={!!act} onClose={() => setAct(null)} loading={busy} onConfirm={run}
        title={act?.kind === 'resolve' ? 'Resolve alert' : 'Acknowledge alert'} confirmLabel={act?.kind === 'resolve' ? 'Resolve' : 'Acknowledge'}
        noteLabel={act?.kind === 'resolve' ? 'Resolution note' : 'Note'} noteRequired={act?.kind === 'resolve'}
        body={<p>{act?.alert.title}</p>} />
    </>
  );
}
