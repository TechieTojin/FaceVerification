import { Link, useParams } from 'react-router-dom';
import { CheckCircle2, Circle, XCircle } from 'lucide-react';
import { apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { STEP_LABEL, fmtDateTime, fmtMs, fmtNum, fmtTime, humanize } from '../lib/format';
import type { HistoryDetail } from '../lib/types';
import { useCan } from '../store/auth';
import { Callout, Card, CardHeader, EmptyState, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { OutcomeBadge } from '../components/OutcomeBadge';

export default function SessionDetail() {
  const { sessionId = '' } = useParams();
  const can = useCan();
  const q = useApi<HistoryDetail>(`/history/sessions/${sessionId}`);
  const d = q.data;
  if (q.isLoading) return <Card><LoadingBlock rows={8} /></Card>;
  if (q.isError || !d) return <Callout tone="bad" title="Session unavailable">{apiError(q.error).message}</Callout>;
  const scores = can('history:scores');
  const rounds = [...new Set(d.challenges.map((c) => c.round))];

  return (
    <>
      <PageHeader title={`${humanize(d.purpose)} session`} description={<span className="font-mono text-xs">{d.session_id}</span>}
        actions={<OutcomeBadge outcome={d.outcome} active={d.status === 'ACTIVE'} />} />
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Summary" />
          <div className="px-5 py-4">
            <KeyValue items={[
              ['Person', d.target_name ? (d.target_user_id && can('users:read')
                ? <Link key="p" to={`/users/${d.target_user_id}`} className="text-primary hover:underline">{d.target_name}</Link> : d.target_name) : '—'],
              ['Operator', d.operator_name ?? '—'],
              ['Started', fmtDateTime(d.created_at)],
              ['Completed', fmtDateTime(d.completed_at)],
              ['Outcome', humanize(d.outcome)],
              ['Reason', humanize(d.reason_code)],
              ['Frames processed', d.frames],
              ['Attempts used', d.attempts_used],
              ['Server processing', fmtMs(d.processing_ms)],
              ['Policy version', d.policy_version ?? '—'],
              ...(scores ? [['Final distance / threshold', `${fmtNum(d.final_distance ?? null)} / ${fmtNum(d.threshold ?? null, 2)}`] as [string, string]] : []),
              ['Liveness', d.liveness_outcome ? humanize(d.liveness_outcome) : '—'],
            ]} />
            {scores && d.final_distance !== null && d.final_distance !== undefined && (
              <p className="mt-4 text-xs text-muted">Cosine distance measures dissimilarity between embeddings; it is not a probability that two faces belong to the same person.</p>
            )}
          </div>
        </Card>
        <Card>
          <CardHeader title="Liveness challenge" subtitle={`${rounds.length} round(s), server-generated`} />
          {d.challenges.length === 0 ? <EmptyState title="No challenge steps recorded" /> : (
            <div className="space-y-4 px-5 py-4">
              {rounds.map((r) => (
                <div key={r}>
                  <p className="mb-2 text-xs font-medium text-muted">Round {r + 1}</p>
                  <ol className="space-y-1.5">
                    {d.challenges.filter((c) => c.round === r).map((c) => (
                      <li key={`${r}-${c.index}`} className="flex items-center gap-2 text-sm">
                        {c.status === 'PASSED' ? <CheckCircle2 className="h-4 w-4 text-good-text" />
                          : c.status === 'FAILED' ? <XCircle className="h-4 w-4 text-critical" /> : <Circle className="h-4 w-4 text-muted" />}
                        <span className="flex-1 text-ink-2">{STEP_LABEL[c.type] ?? c.type}</span>
                        <span className="text-xs text-muted">{humanize(c.status)}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
      <Card className="mt-6">
        <CardHeader title="Frame timeline" subtitle="One row per processed frame. No images are stored." />
        <Table label="Frame events">
          <THead><tr><Th>Time</Th><Th>#</Th><Th>Step</Th><Th>Result</Th><Th>Detail</Th><Th className="text-right">Quality</Th>
            {scores && <><Th className="text-right">Yaw</Th><Th>Passive</Th><Th className="text-right">Distance</Th></>}<Th className="text-right">Time</Th></tr></THead>
          <tbody>{d.events.map((e, i) => (
            <Tr key={i}>
              <Td className="whitespace-nowrap text-xs">{fmtTime(e.timestamp)}</Td>
              <Td className="tabular text-xs">{e.seq ?? '—'}</Td>
              <Td className="text-xs">{e.step_type ? STEP_LABEL[e.step_type] : '—'}</Td>
              <Td><OutcomeBadge outcome={e.outcome} /></Td>
              <Td className="max-w-[260px] truncate text-xs" title={e.explanation ?? ''}>{e.explanation ?? humanize(e.reason_code)}</Td>
              <Td className="tabular text-right text-xs">{fmtNum(e.quality_score, 2)}</Td>
              {scores && <>
                <Td className="tabular text-right text-xs">{e.yaw !== null && e.yaw !== undefined ? `${e.yaw.toFixed(0)}°` : '—'}</Td>
                <Td className="text-xs">{e.passive_label ? `${e.passive_label} ${fmtNum(e.passive_confidence ?? null, 2)}` : '—'}</Td>
                <Td className="tabular text-right text-xs">{fmtNum(e.distance ?? null)}</Td>
              </>}
              <Td className="tabular text-right text-xs">{fmtMs(e.processing_ms)}</Td>
            </Tr>
          ))}</tbody>
        </Table>
      </Card>
      {d.policy_snapshot && (
        <Card className="mt-6">
          <CardHeader title={`Policy snapshot (v${d.policy_version})`} subtitle="The rules this session was evaluated under" />
          <dl className="grid grid-cols-1 gap-x-6 gap-y-2 px-5 py-4 text-xs sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(d.policy_snapshot).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3 border-b border-line py-1"><dt className="text-muted">{humanize(k)}</dt><dd className="tabular text-ink">{String(v)}</dd></div>
            ))}
          </dl>
        </Card>
      )}
    </>
  );
}
