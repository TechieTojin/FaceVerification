import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { ClipboardCheck } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { REASON_TEXT, fmtDateTime, fmtNum, humanize } from '../lib/format';
import type { EnrolmentRequest } from '../lib/types';
import { useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Card, Callout, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { ConfirmDialog } from '../components/ui/Modal';
import { Pagination, Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { Tabs } from '../components/ui/Tabs';

type StatusFilter = 'PENDING' | 'APPROVED' | 'REJECTED' | 'AUTO_APPROVED' | '';

const STATUS_TONE: Record<EnrolmentRequest['status'], 'good' | 'bad' | 'warn' | 'info' | 'neutral'> = {
  PENDING: 'warn', APPROVED: 'good', AUTO_APPROVED: 'good', REJECTED: 'bad', CANCELLED: 'neutral',
};

export default function Approvals() {
  const can = useCan();
  const qc = useQueryClient();
  const [status, setStatus] = useState<StatusFilter>('PENDING');
  const [page, setPage] = useState(1);
  const [action, setAction] = useState<{ req: EnrolmentRequest; approve: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const q = useApi<{ total: number; page: number; page_size: number; requests: EnrolmentRequest[] }>('/enrolments/requests',
    { status, page, page_size: 20 });
  const reviewer = can('enrolment:approve');

  const review = async (note: string) => {
    if (!action) return;
    setBusy(true);
    try {
      await api.post(`/enrolments/requests/${action.req.id}/${action.approve ? 'approve' : 'reject'}`, { note });
      toast.success(action.approve ? 'Request approved — template is now active' : 'Request rejected');
      setAction(null);
      void qc.invalidateQueries({ queryKey: ['/enrolments/requests'] });
    } catch (err) {
      toast.error('Review failed', apiError(err).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader title="Enrolment approvals"
        description={reviewer ? 'Template replacements, suspected duplicates and self-enrolments wait here for review.' : 'Enrolment requests you submitted.'} />
      <div className="mb-4">
        <Tabs label="Request status" value={status} onChange={(v) => { setStatus(v); setPage(1); }} items={[
          { value: 'PENDING', label: 'Pending' }, { value: 'APPROVED', label: 'Approved' },
          { value: 'AUTO_APPROVED', label: 'Auto-approved' }, { value: 'REJECTED', label: 'Rejected' }, { value: '', label: 'All' },
        ]} />
      </div>
      <Card>
        {q.isLoading ? <LoadingBlock /> : q.isError ? <Callout tone="bad" className="m-4">{apiError(q.error).message}</Callout> :
          !q.data?.requests.length ? (
            <EmptyState icon={<ClipboardCheck className="h-5 w-5" />} title="Nothing to review" body="No requests match this filter." />
          ) : (
            <>
              <Table label="Enrolment requests">
                <THead><tr><Th>Subject</Th><Th>Kind</Th><Th>Why review</Th><Th>Quality</Th><Th>Requested</Th><Th>Status</Th>{reviewer && <Th className="text-right">Actions</Th>}</tr></THead>
                <tbody>
                  {q.data.requests.map((r) => (
                    <Tr key={r.id}>
                      <Td>
                        <Link to={`/enrolments/${r.user_id}`} className="font-medium text-ink hover:underline">{r.user_name ?? r.user_id.slice(0, 8)}</Link>
                        <p className="text-xs text-muted">{r.user_email}</p>
                      </Td>
                      <Td>{r.kind === 'REENROL' ? 'Replacement' : 'Initial'}</Td>
                      <Td>
                        {r.approval_reasons.length ? (
                          <ul className="space-y-0.5 text-xs">{r.approval_reasons.map((x) => <li key={x}>{REASON_TEXT[x] ?? humanize(x)}</li>)}</ul>
                        ) : <span className="text-xs text-muted">None</span>}
                        {r.duplicate_user_ids && r.duplicate_user_ids.length > 0 && (
                          <p className="mt-1 text-xs text-critical">Matches: {r.duplicate_user_ids.map((id) => (
                            <Link key={id} to={`/users/${id}`} className="mr-1 underline">{id.slice(0, 8)}</Link>))}</p>
                        )}
                      </Td>
                      <Td className="tabular text-xs">{fmtNum(r.quality_score, 2)} · {r.sample_count} samples
                        <br /><span className="text-muted">spread {fmtNum(r.consistency_score, 3)}</span></Td>
                      <Td className="text-xs">{fmtDateTime(r.created_at)}<br /><span className="text-muted">{r.requested_by}</span></Td>
                      <Td><Badge tone={STATUS_TONE[r.status]}>{humanize(r.status)}</Badge>
                        {r.review_note && <p className="mt-1 max-w-[220px] truncate text-xs text-muted" title={r.review_note}>{r.review_note}</p>}</Td>
                      {reviewer && (
                        <Td className="text-right">
                          {r.status === 'PENDING' && (
                            <div className="flex justify-end gap-2">
                              <Button size="sm" variant="outline" onClick={() => setAction({ req: r, approve: false })}>Reject</Button>
                              <Button size="sm" onClick={() => setAction({ req: r, approve: true })}>Approve</Button>
                            </div>
                          )}
                        </Td>
                      )}
                    </Tr>
                  ))}
                </tbody>
              </Table>
              <Pagination page={page} pageSize={20} total={q.data.total} onPage={setPage} />
            </>
          )}
      </Card>
      <ConfirmDialog open={!!action} onClose={() => setAction(null)} loading={busy} onConfirm={review}
        title={action?.approve ? 'Approve enrolment?' : 'Reject enrolment?'} danger={!action?.approve}
        confirmLabel={action?.approve ? 'Approve and activate' : 'Reject'} noteLabel="Review note" noteRequired={!action?.approve}
        body={action && (action.approve
          ? <p>The new template for <strong>{action.req.user_name}</strong> becomes active immediately{action.req.kind === 'REENROL' ? ' and replaces the current one' : ''}.</p>
          : <p>The candidate template for <strong>{action.req.user_name}</strong> is discarded.</p>)} />
    </>
  );
}
