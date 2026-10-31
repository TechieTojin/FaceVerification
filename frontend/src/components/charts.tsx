import { useState, type ReactNode } from 'react';
import { Table2 } from 'lucide-react';
import { Card, CardHeader } from './ui/primitives';
import { Button } from './ui/Button';

/** Fixed categorical order (validated reference palette). Colour follows the series, never its rank. */
export const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)'];
export const AXIS = { stroke: 'var(--chart-axis)', tick: { fill: 'var(--chart-axis)', fontSize: 11 }, tickLine: false };
export const GRID = { stroke: 'var(--chart-grid)', strokeDasharray: '0', vertical: false };

export const tooltipStyle = {
  contentStyle: {
    background: 'rgb(var(--surface))', border: '1px solid rgb(var(--line))', borderRadius: 8, fontSize: 12,
    color: 'rgb(var(--ink))', boxShadow: '0 8px 24px -8px rgb(0 0 0 / 0.25)',
  },
  labelStyle: { color: 'rgb(var(--ink-2))', marginBottom: 4 },
  itemStyle: { color: 'rgb(var(--ink))', padding: 0 },
  cursor: { fill: 'rgb(var(--surface-2))', opacity: 0.6 },
};

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-label="Legend">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: i.color }} aria-hidden />{i.label}
        </li>
      ))}
    </ul>
  );
}

/** Chart card with a table-view toggle (identity never rests on colour alone). */
export function ChartCard({ title, subtitle, legend, chart, table, empty }: {
  title: string; subtitle?: string; legend?: ReactNode; chart: ReactNode; table: ReactNode; empty?: boolean;
}) {
  const [asTable, setAsTable] = useState(false);
  return (
    <Card>
      <CardHeader title={title} subtitle={subtitle}
        actions={!empty && <Button size="sm" variant="ghost" icon={<Table2 className="h-4 w-4" />} aria-pressed={asTable}
          onClick={() => setAsTable((v) => !v)}>{asTable ? 'Chart' : 'Table'}</Button>} />
      <div className="px-5 py-4">
        {empty ? <p className="py-10 text-center text-sm text-muted">No data in this period yet.</p> : asTable ? (
          <div className="max-h-72 overflow-auto">{table}</div>
        ) : (
          <>
            {legend && <div className="mb-3">{legend}</div>}
            <div className="h-64 w-full">{chart}</div>
          </>
        )}
      </div>
    </Card>
  );
}

export function MiniTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <table className="w-full text-xs">
      <thead><tr className="border-b border-line text-left text-muted">{head.map((h) => <th key={h} className="py-1.5 pr-3 font-medium">{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => (
        <tr key={i} className="border-b border-line last:border-0">{r.map((c, j) => <td key={j} className="tabular py-1.5 pr-3 text-ink-2">{c}</td>)}</tr>
      ))}</tbody>
    </table>
  );
}
