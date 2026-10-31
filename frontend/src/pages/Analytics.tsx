import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtMs, fmtPct, humanize } from '../lib/format';
import type { AnalyticsOverview } from '../lib/types';
import { Callout, Card, LoadingBlock, PageHeader, StatCard } from '../components/ui/primitives';
import { Tabs } from '../components/ui/Tabs';
import { AXIS, ChartCard, GRID, Legend, MiniTable, SERIES, tooltipStyle } from '../components/charts';

const RANGES = [{ value: '7', label: '7 days' }, { value: '30', label: '30 days' }, { value: '90', label: '90 days' }] as const;
type Range = (typeof RANGES)[number]['value'];

export default function Analytics() {
  const [days, setDays] = useState<Range>('30');
  const q = useApi<AnalyticsOverview>('/analytics/overview', { days });
  const a = q.data;
  const daily = (a?.daily ?? []).map((d) => ({ ...d, label: d.date.slice(5) }));
  const hourly = (a?.throughput_24h ?? []).map((h) => ({ ...h, label: new Date(`${h.hour}Z`).toLocaleTimeString([], { hour: '2-digit' }) }));
  const outcomes = Object.entries(a?.outcome_distribution ?? {}).sort((x, y) => y[1] - x[1]).map(([k, v]) => ({ name: humanize(k), value: v }));
  const hasSessions = (a?.totals.sessions ?? 0) > 0;
  const hasFrames = daily.some((d) => d.latency_p50 !== null);
  // A line needs ≥2 points to be visible; with few data days draw point markers too.
  const sparse = daily.filter((d) => d.latency_p50 !== null).length <= 3;

  return (
    <>
      <PageHeader title="Analytics" description="Operational performance from recorded sessions. These are outcome rates, not accuracy measurements."
        actions={<Tabs label="Time range" value={days} onChange={setDays} items={[...RANGES]} />} />
      {q.isLoading ? <Card><LoadingBlock rows={8} /></Card> : q.isError || !a ? <Callout tone="bad">{apiError(q.error).message}</Callout> : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4 xl:grid-cols-7">
            <StatCard label="Verification sessions" value={a.totals.sessions} hint={`${a.totals.frames} frames`} />
            <StatCard label="Completion rate" value={fmtPct(a.session_completion_rate)} hint="Reached a decision" />
            <StatCard label="Verified rate" value={fmtPct(a.verified_rate)} hint="Of decided sessions" />
            <StatCard label="Liveness rejections" value={fmtPct(a.liveness_rejection_rate)} hint="Of decided sessions" />
            <StatCard label="Quality failures" value={fmtPct(a.quality_failure_rate)} hint="Of frames" />
            <StatCard label="Latency p95" value={fmtMs(a.latency_ms.p95)} hint={`p50 ${fmtMs(a.latency_ms.p50)} · p99 ${fmtMs(a.latency_ms.p99)}`} />
            <StatCard label="Frame error rate" value={fmtPct(a.frame_error_rate)} />
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <ChartCard title="Sessions per day" subtitle="Decided verification outcomes" empty={!hasSessions}
              legend={<Legend items={[{ label: 'Verified', color: SERIES[0] }, { label: 'Not matched', color: SERIES[1] }, { label: 'Liveness failed', color: SERIES[2] }]} />}
              chart={<ResponsiveContainer><BarChart data={daily} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
                <CartesianGrid {...GRID} /><XAxis dataKey="label" {...AXIS} interval="preserveStartEnd" minTickGap={16} />
                <YAxis {...AXIS} allowDecimals={false} axisLine={false} /><Tooltip {...tooltipStyle} />
                <Bar dataKey="verified" name="Verified" stackId="a" fill={SERIES[0]} stroke="var(--chart-surface)" strokeWidth={2} />
                <Bar dataKey="not_matched" name="Not matched" stackId="a" fill={SERIES[1]} stroke="var(--chart-surface)" strokeWidth={2} />
                <Bar dataKey="liveness_failed" name="Liveness failed" stackId="a" fill={SERIES[2]} stroke="var(--chart-surface)" strokeWidth={2} radius={[4, 4, 0, 0]} />
              </BarChart></ResponsiveContainer>}
              table={<MiniTable head={['Date', 'Sessions', 'Verified', 'Not matched', 'Liveness failed']}
                rows={daily.filter((d) => d.sessions).map((d) => [d.date, d.sessions, d.verified, d.not_matched, d.liveness_failed])} />} />

            <ChartCard title="Server latency per frame" subtitle="Daily p50 and p95, milliseconds" empty={!hasFrames}
              legend={<Legend items={[{ label: 'p50', color: SERIES[0] }, { label: 'p95', color: SERIES[1] }]} />}
              chart={<ResponsiveContainer><LineChart data={daily} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid {...GRID} /><XAxis dataKey="label" {...AXIS} interval="preserveStartEnd" minTickGap={16} />
                <YAxis {...AXIS} axisLine={false} unit=" ms" width={64} /><Tooltip {...tooltipStyle} />
                <Line type="monotone" dataKey="latency_p50" name="p50" stroke={SERIES[0]} strokeWidth={2} connectNulls
                  dot={sparse ? { r: 4, strokeWidth: 2, fill: 'var(--chart-surface)' } : false} />
                <Line type="monotone" dataKey="latency_p95" name="p95" stroke={SERIES[1]} strokeWidth={2} connectNulls
                  dot={sparse ? { r: 4, strokeWidth: 2, fill: 'var(--chart-surface)' } : false} />
              </LineChart></ResponsiveContainer>}
              table={<MiniTable head={['Date', 'Frames', 'p50 ms', 'p95 ms', 'Frame errors']}
                rows={daily.filter((d) => d.frames).map((d) => [d.date, d.frames, d.latency_p50 ?? '—', d.latency_p95 ?? '—', d.frame_errors])} />} />

            <ChartCard title="Throughput — last 24 hours" subtitle="Capture sessions started per hour" empty={!hourly.some((h) => h.sessions)}
              chart={<ResponsiveContainer><BarChart data={hourly} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
                <CartesianGrid {...GRID} /><XAxis dataKey="label" {...AXIS} interval={3} />
                <YAxis {...AXIS} allowDecimals={false} axisLine={false} /><Tooltip {...tooltipStyle} />
                <Bar dataKey="sessions" name="Sessions" fill={SERIES[0]} radius={[4, 4, 0, 0]} maxBarSize={18} />
              </BarChart></ResponsiveContainer>}
              table={<MiniTable head={['Hour (local)', 'Sessions']} rows={hourly.map((h) => [h.label, h.sessions])} />} />

            <ChartCard title="Outcome mix" subtitle={`All closed verification sessions, last ${days} days`} empty={!outcomes.length}
              chart={<ResponsiveContainer><BarChart data={outcomes} layout="vertical" margin={{ top: 0, right: 16, left: 24, bottom: 0 }}>
                <CartesianGrid {...GRID} horizontal={false} vertical /><XAxis type="number" {...AXIS} allowDecimals={false} />
                <YAxis type="category" dataKey="name" {...AXIS} width={120} axisLine={false} /><Tooltip {...tooltipStyle} />
                <Bar dataKey="value" name="Sessions" fill={SERIES[0]} radius={[0, 4, 4, 0]} maxBarSize={18} />
              </BarChart></ResponsiveContainer>}
              table={<MiniTable head={['Outcome', 'Sessions']} rows={outcomes.map((o) => [o.name, o.value])} />} />
          </div>

          <Callout tone="info" title="About these numbers">
            {a.notes.map((n) => <p key={n}>{n}</p>)}
            <p className="mt-1"><Link to="/models" className="font-medium text-primary hover:underline">Model performance</Link> computes FAR/FRR/EER from a labelled evaluation dataset.</p>
          </Callout>
        </div>
      )}
    </>
  );
}
