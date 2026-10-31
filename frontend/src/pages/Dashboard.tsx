import { Link, useNavigate } from 'react-router-dom';
import { Activity, AlertTriangle, BellRing, CheckCircle2, Clock, Database, Cpu, ScanFace, XCircle } from 'lucide-react';
import { useApi } from '../lib/query';
import { fmtMs, fmtRelative, humanize } from '../lib/format';
import type { DashboardSummary } from '../lib/types';
import { useCan } from '../store/auth';
import { Card, CardHeader, Callout, EmptyState, LoadingBlock, PageHeader, StatCard, StatusDot } from '../components/ui/primitives';
import { Button } from '../components/ui/Button';
import { OutcomeBadge } from '../components/OutcomeBadge';

export default function Dashboard() {
  const can = useCan();
  const navigate = useNavigate();
  const q = useApi<DashboardSummary>('/dashboard/summary', undefined, { refetchInterval: 30_000 });
  const d = q.data;

  return (
    <>
      <PageHeader title="Dashboard" description="Live operational view — last 24 hours unless noted."
        actions={<>
          {d && <span className="text-xs text-muted" data-testid="dashboard-updated">Updated {fmtRelative(d.generated_at)}</span>}
          {can('verification:run') && <Button icon={<ScanFace className="h-4 w-4" />} onClick={() => navigate('/verify')}>Start verification</Button>}
        </>} />
      {q.isError && <Callout tone="bad" title="Dashboard data could not be loaded" className="mb-4"
        action={<Button size="sm" variant="outline" onClick={() => q.refetch()}>Retry</Button>}>The last values shown may be out of date.</Callout>}
      {!d ? <Card><LoadingBlock rows={6} /></Card> : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
            <StatCard label="Active sessions" value={d.active_sessions_total} icon={<Activity className="h-4 w-4" />}
              hint={Object.entries(d.active_sessions).map(([k, v]) => `${v} ${k.toLowerCase()}`).join(' · ') || 'None running'} />
            <StatCard label="Completed verifications" value={d.completed_verifications} icon={<Clock className="h-4 w-4" />} />
            <StatCard label="Verified" value={d.verified} icon={<CheckCircle2 className="h-4 w-4" />} tone="good" />
            <StatCard label="Not matched" value={d.not_matched} icon={<XCircle className="h-4 w-4" />} tone="bad" />
            <StatCard label="Liveness failures" value={d.liveness_failures} icon={<AlertTriangle className="h-4 w-4" />} tone="bad" />
            <StatCard label="Open alerts" value={d.open_alerts_total} icon={<BellRing className="h-4 w-4" />}
              tone={d.open_alerts_total ? 'bad' : undefined} hint={`${d.suspicious_events_24h} alert updates in 24 h`} />
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader title="Recent sessions" subtitle="Most recently completed capture sessions"
                actions={<Link to="/history" className="text-xs font-medium text-primary hover:underline">View history</Link>} />
              {d.recent_sessions.length === 0 ? (
                <EmptyState title="No completed sessions yet" body="Results appear here as soon as a verification finishes." />
              ) : (
                <ul className="divide-y divide-line">
                  {d.recent_sessions.map((s) => (
                    <li key={s.session_id}>
                      <Link to={`/history/${s.session_id}`} className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 hover:bg-surface-2/60">
                        <div className="min-w-0">
                          <p className="truncate text-sm text-ink">{s.target_name ?? humanize(s.purpose)}</p>
                          <p className="text-xs text-muted">{humanize(s.purpose)}{s.operator_name ? ` · by ${s.operator_name}` : ''} · {fmtRelative(s.completed_at)}</p>
                        </div>
                        <OutcomeBadge outcome={s.outcome} />
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </Card>

            <div className="space-y-6">
              <Card>
                <CardHeader title="System health" subtitle="Checked on each refresh"
                  actions={can('system:health') ? <Link to="/system" className="text-xs font-medium text-primary hover:underline">Details</Link> : undefined} />
                <ul className="space-y-3 px-5 py-4 text-sm" data-testid="health-panel">
                  <li className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-ink-2"><Database className="h-4 w-4" />Database</span>
                    <StatusDot tone={d.health.database.ok ? 'good' : 'bad'}
                      label={d.health.database.ok ? `Reachable · ${fmtMs(d.health.database.latency_ms)}` : `Unreachable${d.health.database.error ? ` (${d.health.database.error})` : ''}`} />
                  </li>
                  <li className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-ink-2"><Cpu className="h-4 w-4" />AI models</span>
                    <StatusDot tone={d.health.models.ok ? 'good' : d.health.models.state === 'WARMING' ? 'warn' : 'bad'}
                      label={humanize(d.health.models.state)} pulse={d.health.models.state === 'WARMING'} />
                  </li>
                  <li className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-ink-2"><Activity className="h-4 w-4" />Schema</span>
                    <StatusDot tone={d.health.migrations_up_to_date ? 'good' : 'bad'} label={d.health.migrations_up_to_date ? 'Up to date' : 'Migration pending'} />
                  </li>
                </ul>
              </Card>
              <Card>
                <CardHeader title="Last hour" />
                <dl className="grid grid-cols-2 gap-4 px-5 py-4 text-sm">
                  <div><dt className="text-xs text-muted">Sessions started</dt><dd className="tabular text-lg font-semibold">{d.throughput_last_hour.sessions}</dd></div>
                  <div><dt className="text-xs text-muted">Frames / min</dt><dd className="tabular text-lg font-semibold">{d.throughput_last_hour.frames_per_minute}</dd></div>
                  <div><dt className="text-xs text-muted">Latency p50</dt><dd className="tabular text-lg font-semibold">{fmtMs(d.latency_last_hour_ms.p50)}</dd></div>
                  <div><dt className="text-xs text-muted">Latency p95</dt><dd className="tabular text-lg font-semibold">{fmtMs(d.latency_last_hour_ms.p95)}</dd></div>
                </dl>
                <p className="px-5 pb-4 text-xs text-muted">Server processing time per frame ({d.latency_last_hour_ms.samples} frames).</p>
              </Card>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
