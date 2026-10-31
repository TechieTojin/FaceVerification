import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ClipboardList } from 'lucide-react';
import { useApi } from '../lib/query';
import { fmtRelative, humanize } from '../lib/format';
import type { Paged, Review } from '../lib/types';
import { Badge, Card, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Select } from '../components/ui/form';
import { Pagination, Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { Tabs } from '../components/ui/Tabs';
import { OutcomeBadge } from '../components/OutcomeBadge';

export const RISK_TONE = { LOW: 'neutral', MEDIUM: 'warn', HIGH: 'bad' } as const;
export const REVIEW_TONE: Record<string, 'good' | 'warn' | 'bad' | 'neutral' | 'info'> = {
  PENDING: 'warn', IN_REVIEW: 'info', APPROVED: 'good', REJECTED: 'bad', RETRY_REQUESTED: 'neutral', EXPIRED: 'neutral',
};

type Status = 'OPEN' | 'ALL' | 'APPROVED' | 'REJECTED' | 'RETRY_REQUESTED' | 'EXPIRED';

export default function Reviews() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<Status>('OPEN');
  const [risk, setRisk] = useState('');
  const [mine, setMine] = useState(false);
  const [page, setPage] = useState(1);
  const q = useApi<Paged & { counts: Record<string, number>; reviews: Review[] }>('/reviews', { status, risk, mine, page, page_size: 20 });
  const counts = q.data?.counts ?? {};
  const open = (counts.PENDING ?? 0) + (counts.IN_REVIEW ?? 0);

  return (
    <>
      <PageHeader title="Review queue" description="Uncertain or high-risk results routed to a person. A review is an administrative decision: it never changes the technical outcome, cannot approve a failed liveness check, and never signs anyone in." />
      <Card>
        <div className="flex flex-wrap items-center gap-3 border-b border-line p-4">
          <Tabs label="Status" value={status} onChange={(v) => { setStatus(v); setPage(1); }} items={[
            { value: 'OPEN', label: 'Open', count: open }, { value: 'APPROVED', label: 'Approved', count: counts.APPROVED ?? 0 },
            { value: 'REJECTED', label: 'Rejected', count: counts.REJECTED ?? 0 }, { value: 'RETRY_REQUESTED', label: 'Retry', count: counts.RETRY_REQUESTED ?? 0 },
            { value: 'EXPIRED', label: 'Expired', count: counts.EXPIRED ?? 0 }, { value: 'ALL', label: 'All' }]} />
          <Select aria-label="Risk" value={risk} onChange={(e) => { setRisk(e.target.value); setPage(1); }} className="w-36">
            <option value="">Any risk</option><option value="HIGH">High</option><option value="MEDIUM">Medium</option><option value="LOW">Low</option>
          </Select>
          <label className="flex items-center gap-2 text-sm text-ink-2"><input type="checkbox" checked={mine} onChange={(e) => { setMine(e.target.checked); setPage(1); }} />Assigned to me</label>
        </div>
        {q.isLoading ? <LoadingBlock rows={5} /> : !q.data?.reviews.length
          ? <EmptyState icon={<ClipboardList className="h-5 w-5" />} title="Nothing to review" body="New cases appear here as soon as a session finishes." /> : (
            <>
              <Table label="Reviews">
                <THead><tr><Th>Raised</Th><Th>Subject</Th><Th>Reason</Th><Th>Risk</Th><Th>Technical outcome</Th><Th>Status</Th><Th>Assignee</Th></tr></THead>
                <tbody>
                  {q.data.reviews.map((r) => (
                    <Tr key={r.id} onClick={() => navigate(`/reviews/${r.id}`)} className="cursor-pointer" data-testid="review-row">
                      <Td>{fmtRelative(r.created_at)}</Td>
                      <Td className="font-medium text-ink">{r.subject_name ?? '—'}<span className="block text-xs font-normal text-muted">{humanize(r.channel ?? '')}</span></Td>
                      <Td className="max-w-xs"><span className="line-clamp-2 text-sm">{r.reason_text[0]}</span></Td>
                      <Td><Badge tone={RISK_TONE[r.risk_level]}>{r.risk_level}</Badge></Td>
                      <Td><OutcomeBadge outcome={r.technical_outcome} /></Td>
                      <Td><Badge tone={REVIEW_TONE[r.status] ?? 'neutral'}>{humanize(r.status)}</Badge></Td>
                      <Td>{r.assigned_to ?? '—'}</Td>
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
