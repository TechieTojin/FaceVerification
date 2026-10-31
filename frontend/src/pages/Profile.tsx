import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Laptop } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtRelative, humanize } from '../lib/format';
import type { EnrolmentDetail, SignInSession } from '../lib/types';
import { useAuth } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, EmptyState, KeyValue, PageHeader } from '../components/ui/primitives';
import { Field, Input } from '../components/ui/form';
import { ConfirmDialog } from '../components/ui/Modal';

export default function Profile() {
  const user = useAuth((s) => s.user)!;
  const clear = useAuth((s) => s.clear);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const sessions = useApi<{ sessions: SignInSession[] }>('/auth/sessions');
  const enrolment = useApi<EnrolmentDetail>(`/enrolments/${user.id}`);
  const [withdraw, setWithdraw] = useState(false);

  const revoke = async (family?: string) => {
    try {
      const r = await api.post<{ revoked: number }>(family ? `/auth/sessions/${family}/revoke` : '/auth/sessions/revoke-others');
      toast.success(`${r.data.revoked} session(s) signed out`);
      void qc.invalidateQueries({ queryKey: ['/auth/sessions'] });
    } catch (err) {
      toast.error('Could not sign out session', apiError(err).message);
    }
  };

  const doWithdraw = async (note: string) => {
    try {
      await api.post(`/enrolments/${user.id}/consent/withdraw`, { reason: note });
      toast.success('Consent withdrawn', 'Your biometric template was deleted.');
      setWithdraw(false);
      void enrolment.refetch();
    } catch (err) {
      toast.error('Withdrawal failed', apiError(err).message);
    }
  };

  return (
    <>
      <PageHeader title="Profile & security" description="Your account, sign-in sessions and biometric consent." />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Account" />
          <div className="px-5 py-4">
            <KeyValue items={[
              ['Name', user.full_name], ['Email', user.email], ['Role', user.role],
              ['Last sign-in', fmtDateTime(user.last_login_at)], ['Password changed', fmtDateTime(user.password_changed_at)],
              ['Permissions', <span key="p" className="text-xs text-ink-2">{user.permissions.length} granted</span>],
            ]} />
          </div>
        </Card>
        <PasswordCard onChanged={() => { clear(); qc.clear(); navigate('/login', { replace: true }); }} />
        <Card>
          <CardHeader title="Signed-in sessions" subtitle="Each browser or device you are signed in on"
            actions={(sessions.data?.sessions.length ?? 0) > 1 && <Button size="sm" variant="outline" onClick={() => void revoke()}>Sign out others</Button>} />
          {!sessions.data?.sessions.length ? <EmptyState title="No active sessions" /> : (
            <ul className="divide-y divide-line">
              {sessions.data.sessions.map((s) => (
                <li key={s.family_id} className="flex items-center justify-between gap-3 px-5 py-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <Laptop className="h-4 w-4 shrink-0 text-muted" />
                    <div className="min-w-0">
                      <p className="truncate text-sm text-ink">{(s.user_agent ?? 'Unknown client').slice(0, 80)}</p>
                      <p className="text-xs text-muted">{s.ip_address ?? '—'} · active {fmtRelative(s.last_rotated_at)}</p>
                    </div>
                  </div>
                  {s.current ? <Badge tone="info" icon={false}>This device</Badge> : <Button size="sm" variant="ghost" onClick={() => void revoke(s.family_id)}>Sign out</Button>}
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card>
          <CardHeader title="My biometric data" />
          <div className="space-y-4 px-5 py-4">
            {enrolment.data ? (
              <>
                <KeyValue items={[
                  ['Template', <Badge key="t" tone={enrolment.data.usable ? 'good' : 'neutral'}>{humanize(enrolment.data.status)}</Badge>],
                  ['Consent', enrolment.data.consent_given ? `Given ${fmtDateTime(enrolment.data.consent_timestamp)}` : 'Not given'],
                ]} />
                <p className="text-xs text-muted">Only an encrypted face template is stored — never your photos. Withdrawing consent deletes the template immediately.</p>
                {enrolment.data.consent_given && <Button variant="danger" onClick={() => setWithdraw(true)}>Withdraw consent</Button>}
              </>
            ) : enrolment.isError ? <Callout tone="bad">{apiError(enrolment.error).message}</Callout> : null}
          </div>
        </Card>
      </div>
      <ConfirmDialog open={withdraw} onClose={() => setWithdraw(false)} onConfirm={(n) => void doWithdraw(n)} danger
        title="Withdraw biometric consent?" confirmLabel="Withdraw and delete" noteLabel="Reason" typeToConfirm="DELETE"
        body={<p>Your template is permanently deleted and you cannot be verified until you consent and enrol again.</p>} />
    </>
  );
}

function PasswordCard({ onChanged }: { onChanged: () => void }) {
  const [cur, setCur] = useState('');
  const [next, setNext] = useState('');
  const [again, setAgain] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (next.length < 10) { setError('New password must be at least 10 characters.'); return; }
    if (next !== again) { setError('The new passwords do not match.'); return; }
    setBusy(true);
    try {
      await api.post('/auth/change-password', { current_password: cur, new_password: next });
      toast.success('Password changed', 'Please sign in again.');
      onChanged();
    } catch (err) {
      setError(apiError(err).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card>
      <CardHeader title="Change password" subtitle="Signs you out of every session" />
      <form onSubmit={submit} className="space-y-4 px-5 py-4" noValidate>
        {error && <Callout tone="bad">{error}</Callout>}
        <Field label="Current password" htmlFor="cp-cur"><Input id="cp-cur" type="password" autoComplete="current-password" value={cur} onChange={(e) => setCur(e.target.value)} /></Field>
        <Field label="New password" htmlFor="cp-new" hint="10+ characters with at least three of: lowercase, uppercase, digit, symbol.">
          <Input id="cp-new" type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />
        </Field>
        <Field label="Confirm new password" htmlFor="cp-again"><Input id="cp-again" type="password" autoComplete="new-password" value={again} onChange={(e) => setAgain(e.target.value)} /></Field>
        <Button type="submit" loading={busy} disabled={!cur || !next}>Change password</Button>
      </form>
    </Card>
  );
}
