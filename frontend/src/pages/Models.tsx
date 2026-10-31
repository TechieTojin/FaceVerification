import { useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { FlaskConical, Upload } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtMs, fmtNum, fmtPct, humanize } from '../lib/format';
import type { AnalyticsOverview, EvaluationRun } from '../lib/types';
import { useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, EmptyState, KeyValue, LoadingBlock, PageHeader, StatusDot } from '../components/ui/primitives';
import { Field, Input } from '../components/ui/form';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { AXIS, ChartCard, GRID, Legend, MiniTable, SERIES, tooltipStyle } from '../components/charts';

export default function Models() {
  const can = useCan();
  const qc = useQueryClient();
  const overview = useApi<AnalyticsOverview>('/analytics/overview', { days: 7 });
  const runs = useApi<{ runs: EvaluationRun[]; approved_models: string[] }>('/evaluation/runs', undefined, {
    refetchInterval: (query) => ((query.state.data as { runs?: EvaluationRun[] } | undefined)?.runs?.some((r) => r.status === 'RUNNING') ? 2000 : false),
  });
  const [selected, setSelected] = useState<string | null>(null);
  const [label, setLabel] = useState('');
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const run = runs.data?.runs.find((r) => r.id === selected) ?? runs.data?.runs.find((r) => r.status === 'COMPLETED') ?? null;

  const upload = async (file: File) => {
    setBusy(true);
    const form = new FormData();
    form.append('dataset', file);
    form.append('label', label);
    try {
      const r = await api.post<EvaluationRun>('/evaluation/runs', form, { timeout: 120_000 });
      toast.success('Evaluation started', `${r.data.image_count} images, ${r.data.identity_count} identities`);
      setSelected(r.data.id);
      void qc.invalidateQueries({ queryKey: ['/evaluation/runs'] });
    } catch (err) {
      toast.error('Evaluation not started', apiError(err).message);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const models = overview.data?.models ?? [];
  const res = run?.results;

  return (
    <>
      <PageHeader title="Model performance"
        description="Model readiness, live inference latency and offline accuracy evaluation on labelled data you supply." />
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Loaded models" subtitle="Reported by the inference runtime" />
          {overview.isLoading ? <LoadingBlock /> : models.length === 0 ? <EmptyState title="No model status reported yet" /> : (
            <Table label="Models">
              <THead><tr><Th>Model</Th><Th>Task</Th><Th>Status</Th><Th className="text-right">Load time</Th></tr></THead>
              <tbody>{models.map((m) => (
                <Tr key={m.name}>
                  <Td className="font-medium text-ink">{m.name}{m.embedding_dim ? <span className="ml-2 text-xs text-muted">{m.embedding_dim}-d</span> : null}</Td>
                  <Td>{humanize(m.task)}</Td>
                  <Td><StatusDot tone={m.status === 'READY' ? 'good' : 'bad'} label={humanize(m.status)} />{m.error && <p className="text-xs text-critical">{m.error}</p>}</Td>
                  <Td className="tabular text-right text-xs">{fmtMs(m.load_ms ?? null)}</Td>
                </Tr>
              ))}</tbody>
            </Table>
          )}
        </Card>
        <Card>
          <CardHeader title="Live inference" subtitle="Since server start" />
          <div className="px-5 py-4">
            <KeyValue className="sm:grid-cols-2" items={[
              ['p50', fmtMs(overview.data?.runtime_latency_ms.p50)], ['p95', fmtMs(overview.data?.runtime_latency_ms.p95)],
              ['p99', fmtMs(overview.data?.runtime_latency_ms.p99)], ['Samples', overview.data?.runtime_latency_ms.samples ?? 0],
            ]} />
          </div>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader title="Accuracy evaluation" subtitle="FAR, FRR and EER computed only from a labelled dataset — never from production traffic." />
        <div className="grid gap-6 px-5 py-5 lg:grid-cols-[1fr_320px]">
          <div className="space-y-3 text-sm text-ink-2">
            <p>Upload a ZIP whose top-level folders are identities, each containing face images (<code className="rounded bg-surface-2 px-1">alice/1.jpg</code>, <code className="rounded bg-surface-2 px-1">bob/1.jpg</code>…). Every same-folder pair is a genuine pair; every cross-folder pair is an impostor pair. Images are processed in memory and discarded.</p>
            <p className="text-xs text-muted">Use images you are authorised to process. Results only describe this dataset; small datasets give unreliable estimates.</p>
          </div>
          {can('evaluation:run') ? (
            <div className="space-y-3">
              <Field label="Dataset label" htmlFor="ev-label"><Input id="ev-label" value={label} onChange={(e) => setLabel(e.target.value)} maxLength={200} placeholder="e.g. Office pilot, Oct 2026" /></Field>
              <input ref={fileRef} id="ev-file" type="file" accept=".zip,application/zip" className="sr-only"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) void upload(f); }} />
              <Button className="w-full" icon={<Upload className="h-4 w-4" />} loading={busy} onClick={() => fileRef.current?.click()}>Upload labelled dataset</Button>
            </div>
          ) : <p className="text-xs text-muted">Only administrators can start evaluations.</p>}
        </div>
        {!runs.data?.runs.length ? <EmptyState icon={<FlaskConical className="h-5 w-5" />} title="No evaluation runs yet" /> : (
          <Table label="Evaluation runs">
            <THead><tr><Th>Started</Th><Th>Dataset</Th><Th>Model</Th><Th>Status</Th><Th className="text-right">Images</Th><Th className="text-right">Pairs (gen / imp)</Th><Th className="text-right">EER</Th><Th className="text-right">FAR / FRR @ policy</Th></tr></THead>
            <tbody>{runs.data.runs.map((r) => (
              <Tr key={r.id} onClick={() => setSelected(r.id)} className={r.id === run?.id ? 'bg-primary-soft/40' : ''}>
                <Td className="text-xs">{fmtDateTime(r.created_at)}</Td><Td>{r.dataset_label}</Td><Td>{r.model_name}</Td>
                <Td><Badge tone={r.status === 'COMPLETED' ? 'good' : r.status === 'FAILED' ? 'bad' : 'info'}>{humanize(r.status)}</Badge>
                  {r.error && <p className="mt-1 text-xs text-critical">{r.error}</p>}</Td>
                <Td className="tabular text-right text-xs">{r.image_count}{r.skipped_images ? ` (${r.skipped_images} skipped)` : ''}</Td>
                <Td className="tabular text-right text-xs">{r.genuine_pairs} / {r.impostor_pairs}</Td>
                <Td className="tabular text-right text-xs">{r.results ? fmtPct(r.results.eer, 2) : '—'}</Td>
                <Td className="tabular text-right text-xs">{r.results ? `${fmtPct(r.results.at_policy_threshold.far, 2)} / ${fmtPct(r.results.at_policy_threshold.frr, 2)}` : '—'}</Td>
              </Tr>
            ))}</tbody>
          </Table>
        )}
      </Card>

      {run && res && (
        <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <ChartCard title={`Error rates vs threshold — ${run.dataset_label}`}
            subtitle={`EER ${fmtPct(res.eer, 2)} at distance ${fmtNum(res.eer_threshold, 3)} · dashed line = policy threshold ${fmtNum(res.at_policy_threshold.threshold, 2)}`}
            legend={<Legend items={[{ label: 'FAR (impostors accepted)', color: SERIES[1] }, { label: 'FRR (genuine rejected)', color: SERIES[0] }]} />}
            chart={<ResponsiveContainer><LineChart data={res.curve.map((p) => ({ ...p, far: p.far * 100, frr: p.frr * 100 }))} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="threshold" type="number" domain={[0, 1.2]} {...AXIS} tickCount={7} />
              <YAxis {...AXIS} unit="%" domain={[0, 100]} axisLine={false} />
              <Tooltip {...tooltipStyle} formatter={(v: number) => `${v.toFixed(2)}%`} labelFormatter={(t: number) => `Threshold ${Number(t).toFixed(3)}`} />
              <ReferenceLine x={res.at_policy_threshold.threshold} stroke="var(--chart-axis)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="far" name="FAR" stroke={SERIES[1]} strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="frr" name="FRR" stroke={SERIES[0]} strokeWidth={2} dot={false} />
            </LineChart></ResponsiveContainer>}
            table={<MiniTable head={['Threshold', 'FAR', 'FRR']} rows={res.curve.map((p) => [p.threshold.toFixed(2), fmtPct(p.far, 2), fmtPct(p.frr, 2)])} />} />
          <Card>
            <CardHeader title="Run detail" />
            <div className="space-y-4 px-5 py-4">
              {res.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}
              <KeyValue className="sm:grid-cols-2" items={[
                ['Identities', run.identity_count], ['Images used', run.image_count - run.skipped_images],
                ['Genuine pairs', res.genuine_pairs], ['Impostor pairs', res.impostor_pairs],
                ['Genuine mean / max', `${fmtNum(res.genuine_distance.mean)} / ${fmtNum(res.genuine_distance.max)}`],
                ['Impostor mean / min', `${fmtNum(res.impostor_distance.mean)} / ${fmtNum(res.impostor_distance.min)}`],
              ]} />
              <MiniTable head={['Threshold', 'FAR', 'FRR']} rows={res.operating_points.map((p) => [p.threshold.toFixed(2), fmtPct(p.far, 2), fmtPct(p.frr, 2)])} />
            </div>
          </Card>
        </div>
      )}
    </>
  );
}
