import { RefreshCw } from 'lucide-react';
import { apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtDuration, fmtMs, fmtRelative, humanize } from '../lib/format';
import type { SystemHealth as Health } from '../lib/types';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, KeyValue, LoadingBlock, PageHeader, StatusDot } from '../components/ui/primitives';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';

export default function SystemHealth() {
  const q = useApi<Health>('/system/health', undefined, { refetchInterval: 15_000 });
  const h = q.data;
  return (
    <>
      <PageHeader title="System health" description="Live readiness checks; refreshed every 15 seconds."
        actions={<>
          {h && <span className="text-xs text-muted">Checked {fmtRelative(h.checked_at)}</span>}
          <Button variant="outline" icon={<RefreshCw className="h-4 w-4" />} loading={q.isFetching} onClick={() => q.refetch()}>Refresh</Button>
        </>} />
      {q.isLoading ? <Card><LoadingBlock rows={8} /></Card> : q.isError || !h ? <Callout tone="bad" title="Health endpoint unreachable">{apiError(q.error).message}</Callout> : (
        <div className="space-y-6">
          <Callout tone={h.ready ? 'good' : 'bad'} title={h.ready ? 'All readiness checks pass' : 'Service is not ready'}>
            Readiness requires a reachable database, an up-to-date schema and loaded AI models.
          </Callout>
          <div className="grid gap-6 lg:grid-cols-3">
            {Object.entries(h.checks).map(([name, c]) => (
              <Card key={name} className="p-5">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold capitalize text-ink">{name}</p>
                  <Badge tone={c.ok ? 'good' : 'bad'}>{c.ok ? 'OK' : 'Failing'}</Badge>
                </div>
                <dl className="mt-3 space-y-1 text-xs">
                  {Object.entries(c).filter(([k]) => k !== 'ok').map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-3"><dt className="text-muted">{humanize(k)}</dt><dd className="tabular text-ink-2">{v === null || v === undefined ? '—' : String(v)}</dd></div>
                  ))}
                </dl>
              </Card>
            ))}
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader title="Inference runtime" subtitle={`Provider: ${h.inference.provider ?? '—'}`} />
              <div className="px-5 py-4">
                <KeyValue items={[
                  ['Warm-up', <StatusDot key="w" tone={h.inference.ready ? 'good' : 'warn'} label={humanize(h.inference.warmup_state)} />],
                  ['Workers / capacity', `${h.inference.workers} / ${h.inference.capacity}`],
                  ['In flight', h.inference.in_flight], ['Completed', h.inference.completed],
                  ['Rejected (busy)', h.inference.rejected], ['Failed / timeouts', `${h.inference.failed} / ${h.inference.timeouts}`],
                  ['Latency p50 / p95', `${fmtMs(h.inference.latency_ms.p50)} / ${fmtMs(h.inference.latency_ms.p95)}`],
                  ['Live stream subscribers', h.realtime.sse_subscribers],
                ]} />
              </div>
              <Table label="Models" className="border-t border-line">
                <THead><tr><Th>Model</Th><Th>Status</Th><Th className="text-right">Load</Th></tr></THead>
                <tbody>{h.inference.models.map((m) => (
                  <Tr key={m.name}><Td>{m.name}</Td><Td><StatusDot tone={m.status === 'READY' ? 'good' : 'bad'} label={humanize(m.status)} /></Td>
                    <Td className="tabular text-right text-xs">{fmtMs(m.load_ms ?? null)}</Td></Tr>
                ))}</tbody>
              </Table>
            </Card>
            <Card>
              <CardHeader title="Environment" />
              <div className="px-5 py-4">
                <KeyValue items={[
                  ['Version', h.version], ['Environment', h.environment], ['Uptime', fmtDuration(h.uptime_seconds)],
                  ['Database', h.database_backend], ['Python', h.python], ['Platform', <span key="p" className="break-all text-xs">{h.platform}</span>],
                  ['Maintenance last run', h.maintenance.last_run_at ? fmtDateTime(h.maintenance.last_run_at) : 'Not yet'],
                  ['Maintenance status', h.maintenance.ok === null ? '—' : h.maintenance.ok ? 'OK' : `Failed (${h.maintenance.error})`],
                ]} />
                {h.maintenance.result && (
                  <p className="mt-4 text-xs text-muted">Last run: {Object.entries(h.maintenance.result).map(([k, v]) => `${humanize(k)} ${v}`).join(' · ')}</p>
                )}
              </div>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
