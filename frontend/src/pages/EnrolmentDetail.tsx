import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { UserPlus } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtNum, humanize } from '../lib/format';
import type { EnrolmentDetail as Detail, EnrolmentRequest } from '../lib/types';
import { useAuth, useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, EmptyState, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { ConfirmDialog } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';

type Action = 'revoke' | 'withdraw' | 'reenrol' | null;

export default function EnrolmentDetail() {
  const { userId = '' } = useParams();
  const can = useCan();
  const me = useAuth((s) => s.user);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const q = useApi<Detail>(`/enrolments/${userId}`);
  const reqs = useApi<{ requests: EnrolmentRequest[] }>('/enrolments/requests', { user_id: userId, page_size: 20 });
  const [action, setAction] = useState<Action>(null);
  const [busy, setBusy] = useState(false);
  const d = q.data;

  const run = async (note: string) => {
    setBusy(true);
    try {
      if (action === 'revoke') await api.delete(`/enrolments/${userId}`, { data: { reason: note } });
      if (action === 'withdraw') await api.post(`/enrolments/${userId}/consent/withdraw`, { reason: note });
      if (action === 'reenrol') await api.post(`/enrolments/${userId}/request-reenrolment`, { reason: note });
      toast.success({ revoke: 'Enrolment revoked', withdraw: 'Consent withdrawn — template deleted', reenrol: 'Re-enrolment requested' }[action!]);
      setAction(null);
      void qc.invalidateQueries({ queryKey: [`/enrolments/${userId}`] });
    } catch (err) {
      toast.error('Action failed', apiError(err).message);
    } finally {
      setBusy(false);
    }
  };

  if (q.isLoading) return <Card><LoadingBlock rows={6} /></Card>;
  if (q.isError || !d) return <Callout tone="bad" title="Enrolment unavailable">{apiError(q.error).message}</Callout>;
  const self = me?.id === userId;

  return (
    <>
      <PageHeader title={d.user_name} description={`${d.user_email} · ${d.user_role}`}
        actions={<>
          {(can('enrolment:capture') && (d.user_role !== 'ADMIN' || me?.role === 'ADMIN')) ? (
            <Button icon={<UserPlus className="h-4 w-4" />} onClick={() => navigate(`/enrol/${userId}`)}>
              {d.is_enrolled ? 'Re-enrol' : 'Enrol'}
            </Button>) : null}
          {can('enrolment:approve') && d.is_enrolled && !d.reenrolment_requested && (
            <Button variant="outline" onClick={() => setAction('reenrol')}>Request re-enrolment</Button>)}
          {can('enrolment:revoke') && d.is_enrolled && <Button variant="danger" onClick={() => setAction('revoke')}>Revoke</Button>}
          {(self || can('enrolment:approve')) && d.consent_given && <Button variant="outline" onClick={() => setAction('withdraw')}>Withdraw consent</Button>}
        </>} />
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Template" subtitle="The encrypted reference used for verification" />
          <div className="px-5 py-4">
            <KeyValue items={[
              ['Status', <Badge key="s" tone={d.usable ? 'good' : d.status === 'NOT_ENROLLED' ? 'neutral' : 'warn'}>{humanize(d.status)}</Badge>],
              ['Usable for verification', d.usable ? 'Yes' : 'No'],
              ['Enrolled', fmtDateTime(d.enrolled_at)],
              ['Model', d.model_name ?? '—'],
              ['Samples', d.sample_count ?? '—'],
              ['Quality', fmtNum(d.quality_score ?? null, 2)],
              ['Sample spread (max distance)', fmtNum(d.consistency_score ?? null, 3)],
              ['Re-enrolment requested', d.reenrolment_requested ? 'Yes' : 'No'],
            ]} />
            {d.status === 'REENROLMENT_REQUIRED' && <Callout tone="warn" className="mt-4">This template was created by an earlier pipeline and cannot be used until the person is re-enrolled.</Callout>}
          </div>
        </Card>
        <Card>
          <CardHeader title="Consent history" subtitle="Append-only record" />
          {d.consent_history.length === 0 ? <EmptyState title="No consent recorded" /> : (
            <ul className="divide-y divide-line">
              {d.consent_history.map((c, i) => (
                <li key={i} className="px-5 py-3 text-sm">
                  <div className="flex items-center justify-between gap-2">
                    <Badge tone={c.granted ? 'good' : 'warn'}>{c.granted ? 'Granted' : 'Withdrawn'}</Badge>
                    <span className="text-xs text-muted">{fmtDateTime(c.recorded_at)}</span>
                  </div>
                  <p className="mt-1 text-xs text-muted">{humanize(c.purpose)} · {humanize(c.method)}{c.version ? ` · v${c.version}` : ''}</p>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
      <Card className="mt-6">
        <CardHeader title="Enrolment requests" actions={<Link to="/approvals" className="text-xs font-medium text-primary hover:underline">Approval queue</Link>} />
        {!reqs.data?.requests.length ? <EmptyState title="No requests" /> : (
          <Table label="Enrolment requests for this user">
            <THead><tr><Th>Created</Th><Th>Kind</Th><Th>Status</Th><Th>Requested by</Th><Th>Reviewed</Th></tr></THead>
            <tbody>{reqs.data.requests.map((r) => (
              <Tr key={r.id}><Td>{fmtDateTime(r.created_at)}</Td><Td>{humanize(r.kind)}</Td><Td>{humanize(r.status)}</Td>
                <Td>{r.requested_by}</Td><Td>{r.reviewed_by ? `${r.reviewed_by} · ${fmtDateTime(r.reviewed_at)}` : '—'}</Td></Tr>
            ))}</tbody>
          </Table>
        )}
      </Card>
      <ConfirmDialog open={action !== null} onClose={() => setAction(null)} loading={busy} onConfirm={run} danger={action !== 'reenrol'}
        title={action === 'revoke' ? 'Revoke enrolment?' : action === 'withdraw' ? 'Withdraw biometric consent?' : 'Request re-enrolment?'}
        confirmLabel={action === 'revoke' ? 'Revoke template' : action === 'withdraw' ? 'Withdraw and delete' : 'Request'}
        noteLabel="Reason" noteRequired={action === 'revoke'} typeToConfirm={action === 'reenrol' ? undefined : 'DELETE'}
        body={<p>{action === 'revoke' ? 'The encrypted template is deleted and verification is blocked until a new enrolment is approved.'
          : action === 'withdraw' ? 'Consent is withdrawn and the template is permanently deleted. The person cannot be verified until they consent and enrol again.'
            : 'The next enrolment capture will replace the current template without a separate approval.'}</p>} />
    </>
  );
}
