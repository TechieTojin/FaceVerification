import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Download, History as HistoryIcon, Search } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi, type Params } from '../lib/query';
import { fmtDateTime, fmtMs, fmtNum, humanize } from '../lib/format';
import type { HistoryRow } from '../lib/types';
import { useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Input, Select } from '../components/ui/form';
import { Pagination, Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { OutcomeBadge } from '../components/OutcomeBadge';
import { useDebounced } from '../components/UserPicker';

const OUTCOMES = ['VERIFIED', 'NOT_MATCHED', 'LIVENESS_FAILED', 'ENROLMENT_CAPTURED', 'SESSION_EXPIRED', 'SESSION_CANCELLED',
  'ATTEMPTS_EXHAUSTED', 'ERROR'];

export default function History() {
  const can = useCan();
  const navigate = useNavigate();
  const scores = can('history:scores');
  const [search, setSearch] = useState('');
  const [purpose, setPurpose] = useState('');
  const [outcome, setOutcome] = useState('');
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');
  const [sort, setSort] = useState('newest');
  const [page, setPage] = useState(1);
  const [exporting, setExporting] = useState(false);
  const params: Params = { search: useDebounced(search), purpose, outcome, date_from: from || undefined, date_to: to || undefined };
  const q = useApi<{ total: number; sessions: HistoryRow[] }>('/history/sessions', { ...params, sort, page, page_size: 20 },
    { placeholderData: (prev) => prev });
  const reset = <T,>(set: (v: T) => void) => (v: T) => { set(v); setPage(1); };

  const exportCsv = async () => {
    setExporting(true);
    try {
      const r = await api.get<Blob>('/history/export.csv', { params: Object.fromEntries(Object.entries(params).filter(([, v]) => v)), responseType: 'blob' });
      const url = URL.createObjectURL(r.data);
      const a = document.createElement('a');
      a.href = url;
      a.download = `verification-history-${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success('Export downloaded');
    } catch (err) {
      toast.error('Export failed', apiError(err).message);
    } finally {
      setExporting(false);
    }
  };

  return (
    <>
      <PageHeader title="Verification history" description={can('history:read_all') ? 'All capture sessions.' : 'Sessions you ran.'}
        actions={(can('history:export') || can('history:read_all')) && (
          <Button variant="outline" icon={<Download className="h-4 w-4" />} loading={exporting} onClick={exportCsv}>Export CSV</Button>)} />
      <Card>
        <div className="flex flex-wrap items-end gap-3 border-b border-line p-4">
          <div className="relative min-w-[200px] flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden />
            <Input aria-label="Search history" className="pl-9" placeholder="Person, operator or session ID" value={search}
              onChange={(e) => reset(setSearch)(e.target.value)} />
          </div>
          <Select aria-label="Purpose" value={purpose} onChange={(e) => reset(setPurpose)(e.target.value)} className="w-36">
            <option value="">All types</option><option value="VERIFY">Verification</option><option value="ENROL">Enrolment</option><option value="DOCUMENT">Document</option>
          </Select>
          <Select aria-label="Outcome" value={outcome} onChange={(e) => reset(setOutcome)(e.target.value)} className="w-44">
            <option value="">All outcomes</option>{OUTCOMES.map((o) => <option key={o} value={o}>{humanize(o)}</option>)}
          </Select>
          <label className="text-xs text-muted">From<Input type="date" aria-label="From date" value={from} onChange={(e) => reset(setFrom)(e.target.value)} className="mt-1 w-40" /></label>
          <label className="text-xs text-muted">To<Input type="date" aria-label="To date" value={to} onChange={(e) => reset(setTo)(e.target.value)} className="mt-1 w-40" /></label>
          <Select aria-label="Sort" value={sort} onChange={(e) => setSort(e.target.value)} className="w-44">
            <option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="duration">Longest processing</option>
            {scores && <option value="distance">Closest distance</option>}
          </Select>
        </div>
        {q.isLoading ? <LoadingBlock rows={8} /> : q.isError ? <Callout tone="bad" className="m-4">{apiError(q.error).message}</Callout> :
          !q.data?.sessions.length ? <EmptyState icon={<HistoryIcon className="h-5 w-5" />} title="No sessions match" body="Adjust the filters or run a verification." /> : (
            <>
              <Table label="Verification sessions">
                <THead><tr><Th>Started</Th><Th>Type</Th><Th>Person</Th><Th>Operator</Th><Th>Outcome</Th><Th>Liveness</Th>
                  {scores && <Th className="text-right">Distance</Th>}<Th className="text-right">Processing</Th></tr></THead>
                <tbody>
                  {q.data.sessions.map((s) => (
                    <Tr key={s.session_id} onClick={() => navigate(`/history/${s.session_id}`)}>
                      <Td className="whitespace-nowrap text-xs">{fmtDateTime(s.created_at)}</Td>
                      <Td>{humanize(s.purpose)}</Td>
                      <Td><span className="text-ink">{s.target_name ?? '—'}</span></Td>
                      <Td className="text-xs">{s.operator_name ?? '—'}</Td>
                      <Td><OutcomeBadge outcome={s.outcome} active={s.status === 'ACTIVE'} /></Td>
                      <Td>{s.liveness_outcome ? <Badge tone={s.liveness_outcome === 'PASSED' ? 'good' : 'bad'}>{humanize(s.liveness_outcome)}</Badge> : <span className="text-xs text-muted">—</span>}</Td>
                      {scores && <Td className="tabular text-right text-xs">{fmtNum(s.final_distance ?? null)}{s.threshold ? ` / ${fmtNum(s.threshold, 2)}` : ''}</Td>}
                      <Td className="tabular text-right text-xs">{fmtMs(s.processing_ms)}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
              <Pagination page={page} pageSize={20} total={q.data.total} onPage={setPage} />
            </>
          )}
      </Card>
    </>
  );
}
