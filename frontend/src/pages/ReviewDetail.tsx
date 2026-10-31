import { useState, type FormEvent } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, Check, MessageSquare, RotateCcw, UserCheck, X } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtNum, fmtRelative, humanize } from '../lib/format';
import type { Review } from '../lib/types';
import { useAuth, useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Select, Textarea } from '../components/ui/form';
import { OutcomeBadge } from '../components/OutcomeBadge';
import { REVIEW_TONE, RISK_TONE } from './Reviews';

type Decision = 'APPROVE' | 'REJECT' | 'REQUEST_RETRY';

export default function ReviewDetail() {
  const { reviewId = '' } = useParams();
  const can = useCan();
  const me = useAuth((s) => s.user);
  const qc = useQueryClient();
  const q = useApi<Review>(`/reviews/${reviewId}`);
  const reviewers = useApi<{ reviewers: { id: string; full_name: string }[] }>(can('review:decide') ? '/reviews/reviewers' : null);
  const [note, setNote] = useState('');
  const [comment, setComment] = useState('');
  const [assignee, setAssignee] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const r = q.data;
  const open = r && (r.status === 'PENDING' || r.status === 'IN_REVIEW');
  const conflict = r && me && (r.operator_user_id === me.id || r.subject_user_id === me.id);
  const update = (data: Review) => qc.setQueryData([`/reviews/${reviewId}`, undefined], data);

  const run = async (key: string, fn: () => Promise<void>) => {
    setBusy(key); setError(null);
    try { await fn(); } catch (err) {
      const e = apiError(err);
      setError(e.code === 'CONCURRENT_UPDATE' || e.code === 'ALREADY_DECIDED' ? `${e.message} — reloaded.` : e.message);
      if (e.code === 'CONCURRENT_UPDATE' || e.code === 'ALREADY_DECIDED') void q.refetch();
    } finally { setBusy(null); }
  };

  const decide = (decision: Decision) => r && run(decision, async () => {
    update((await api.post<Review>(`/reviews/${r.id}/decide`, { decision, note, version: r.version })).data);
    setNote('');
    toast.success(`Review ${humanize(decision === 'REQUEST_RETRY' ? 'retry_requested' : decision === 'APPROVE' ? 'approved' : 'rejected').toLowerCase()}`);
    void qc.invalidateQueries({ queryKey: ['/reviews'] });
  });
  const assign = (id: string) => r && run('assign', async () => {
    update((await api.post<Review>(`/reviews/${r.id}/assign`, { assignee_user_id: id })).data);
  });
  const addComment = (e: FormEvent) => {
    e.preventDefault();
    if (r) void run('comment', async () => { update((await api.post<Review>(`/reviews/${r.id}/comments`, { note: comment })).data); setComment(''); });
  };

  if (q.isLoading || !r) return q.isError ? <Callout tone="bad">{apiError(q.error).message}</Callout> : <LoadingBlock rows={6} />;
  return (
    <>
      <Link to="/reviews" className="mb-3 inline-flex items-center gap-1 text-sm text-muted hover:text-ink"><ArrowLeft className="h-4 w-4" />Review queue</Link>
      <PageHeader title={`Review: ${r.subject_name ?? 'unknown subject'}`} description={r.reason_text.join(' · ')} />
      {error && <Callout tone="bad" className="mb-4">{error}</Callout>}
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-6">
          <Card>
            <CardHeader title="Case" actions={<><Badge tone={RISK_TONE[r.risk_level]}>{r.risk_level} risk</Badge>
              <Badge tone={REVIEW_TONE[r.status] ?? 'neutral'}>{humanize(r.status)}</Badge></>} />
            <div className="p-5">
              <KeyValue items={[
                ['Technical outcome', <OutcomeBadge key="o" outcome={r.technical_outcome} />],
                ['Channel', humanize(r.channel ?? '')],
                ['Operator', r.operator_name ?? '—'],
                ['Raised', fmtDateTime(r.created_at)],
                ['Expires', open ? fmtDateTime(r.expires_at) : '—'],
                ['Assigned to', r.assigned_to ?? 'Unassigned'],
                ['Decided by', r.decided_by ? `${r.decided_by}, ${fmtDateTime(r.decided_at)}` : '—'],
                ['Decision note', r.decision_note ?? '—'],
              ]} />
              {r.spoof_confirmed && <Callout tone="bad" className="mt-4" title="Presentation attack detected">Passive anti-spoofing classified this capture as a spoof. The case can be rejected or sent for a retry, never approved.</Callout>}
            </div>
          </Card>
          {r.session && (
            <Card>
              <CardHeader title="Session" actions={r.session_id && <Link to={`/history/${r.session_id}`} className="text-xs font-medium text-primary hover:underline">Full session timeline →</Link>} />
              <div className="p-5">
                <KeyValue items={[
                  ['Purpose', humanize(r.session.purpose)], ['Reason', humanize(r.session.reason_code ?? '')],
                  ['Distance / threshold', r.session.final_distance !== null ? `${fmtNum(r.session.final_distance)} / ${fmtNum(r.session.threshold)}` : '—'],
                  ['Frames / attempts', `${r.session.frames} / ${r.session.attempts_used}`],
                  ['Policy version', String(r.session.policy_version ?? '—')],
                ]} />
              </div>
            </Card>
          )}
          <Card>
            <CardHeader title="History & comments" />
            <ol className="space-y-3 p-5">
              {(r.history ?? []).map((h, i) => (
                <li key={i} className="text-sm"><span className="font-medium text-ink">{humanize(h.action)}</span>
                  <span className="text-muted"> · {h.actor ?? 'System'} · {fmtRelative(h.at)}</span>
                  {h.note && <p className="mt-0.5 whitespace-pre-wrap text-ink-2">
                    {h.action === 'CREATED' ? h.note.split(', ').map(humanize).join(', ') : h.note}</p>}</li>))}
            </ol>
            <form onSubmit={addComment} className="flex gap-2 border-t border-line p-4">
              <Textarea aria-label="Comment" rows={2} value={comment} onChange={(e) => setComment(e.target.value)} placeholder="Add a comment" />
              <Button type="submit" variant="outline" icon={<MessageSquare className="h-4 w-4" />} loading={busy === 'comment'} disabled={!comment.trim()}>Comment</Button>
            </form>
          </Card>
        </div>

        {can('review:decide') && open && (
          <div className="space-y-4">
            <Card className="space-y-3 p-5">
              <p className="text-sm font-semibold text-ink">Assignment</p>
              <div className="flex gap-2">
                <Select aria-label="Assignee" value={assignee} onChange={(e) => setAssignee(e.target.value)}>
                  <option value="">Choose reviewer…</option>
                  {reviewers.data?.reviewers.map((u) => <option key={u.id} value={u.id}>{u.full_name}</option>)}
                </Select>
                <Button variant="outline" disabled={!assignee} loading={busy === 'assign'} onClick={() => void assign(assignee)}>Assign</Button>
              </div>
              {me && !conflict && r.assigned_to_user_id !== me.id && (
                <Button variant="ghost" size="sm" icon={<UserCheck className="h-4 w-4" />} onClick={() => void assign(me.id)}>Assign to me</Button>
              )}
            </Card>
            <Card className="space-y-3 p-5" data-testid="review-decision">
              <p className="text-sm font-semibold text-ink">Decision</p>
              {conflict ? <Callout tone="warn">You ran this session or are its subject, so another reviewer must decide.</Callout> : (
                <>
                  <Field label="Reason for the decision (required)" htmlFor="review-note">
                    <Textarea id="review-note" rows={3} value={note} onChange={(e) => setNote(e.target.value)} />
                  </Field>
                  <div className="grid gap-2">
                    <Button icon={<Check className="h-4 w-4" />} disabled={!note.trim() || !r.can_approve || busy !== null} loading={busy === 'APPROVE'}
                      onClick={() => void decide('APPROVE')} data-testid="approve-review">Approve</Button>
                    {!r.can_approve && <p className="text-xs text-muted">Approval is not possible because liveness failed.</p>}
                    <Button variant="danger" icon={<X className="h-4 w-4" />} disabled={!note.trim() || busy !== null} loading={busy === 'REJECT'}
                      onClick={() => void decide('REJECT')} data-testid="reject-review">Reject</Button>
                    <Button variant="outline" icon={<RotateCcw className="h-4 w-4" />} disabled={!note.trim() || busy !== null} loading={busy === 'REQUEST_RETRY'}
                      onClick={() => void decide('REQUEST_RETRY')}>Request a new verification</Button>
                  </div>
                </>
              )}
            </Card>
          </div>
        )}
      </div>
    </>
  );
}
