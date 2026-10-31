import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { KeyRound, Pencil, ScanFace } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtRelative, humanize } from '../lib/format';
import type { AuditEntry, HistoryRow, Role, UserDetail as Detail } from '../lib/types';
import { useAuth, useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, EmptyState, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Input, Select } from '../components/ui/form';
import { ConfirmDialog, Modal } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { OutcomeBadge } from '../components/OutcomeBadge';

export default function UserDetail() {
  const { userId = '' } = useParams();
  const can = useCan();
  const me = useAuth((s) => s.user);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const q = useApi<Detail>(`/users/${userId}`);
  const history = useApi<{ sessions: HistoryRow[]; total: number }>('/history/sessions', { target_user_id: userId, page_size: 10 });
  const audit = useApi<{ entries: AuditEntry[] }>(`/users/${userId}/audit`, { limit: 30 });
  const [edit, setEdit] = useState(false);
  const [confirm, setConfirm] = useState<'disable' | 'enable' | 'delete' | 'reset-mfa' | null>(null);
  const [reset, setReset] = useState(false);
  const [busy, setBusy] = useState(false);
  const u = q.data;
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: [`/users/${userId}`] });
    void qc.invalidateQueries({ queryKey: [`/users/${userId}/audit`] });
    void qc.invalidateQueries({ queryKey: ['/users'] });
  };

  const doConfirm = async () => {
    setBusy(true);
    try {
      if (confirm === 'delete') {
        await api.delete(`/users/${userId}`);
        toast.success('User deleted');
        void qc.invalidateQueries({ queryKey: ['/users'] });
        navigate('/users');
        return;
      }
      await api.post(`/users/${userId}/${confirm}`);
      toast.success(confirm === 'disable' ? 'Account disabled — sessions revoked'
        : confirm === 'reset-mfa' ? 'Sign-in factors removed — the user was notified' : 'Account enabled');
      refresh();
    } catch (err) {
      toast.error('Action failed', apiError(err).message);
    } finally {
      setBusy(false);
      setConfirm(null);
    }
  };

  if (q.isLoading) return <Card><LoadingBlock rows={6} /></Card>;
  if (q.isError || !u) return <Callout tone="bad" title="User unavailable">{apiError(q.error).message}</Callout>;
  const self = me?.id === u.id;
  const manage = can('users:manage');

  return (
    <>
      <PageHeader title={u.full_name} description={u.email}
        actions={manage && <>
          <Button variant="outline" icon={<Pencil className="h-4 w-4" />} onClick={() => setEdit(true)}>Edit</Button>
          <Button variant="outline" icon={<KeyRound className="h-4 w-4" />} onClick={() => setReset(true)}>Reset password</Button>
          {u.managed_here && !self && <Button variant="outline" onClick={() => setConfirm('reset-mfa')}>Reset sign-in factors</Button>}
          {u.is_active
            ? <Button variant="outline" disabled={self} onClick={() => setConfirm('disable')}>Disable</Button>
            : <Button variant="outline" onClick={() => setConfirm('enable')}>Enable</Button>}
          <Button variant="danger" disabled={self} onClick={() => setConfirm('delete')}>Delete</Button>
        </>} />
      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader title="Account" />
          <div className="px-5 py-4">
            <KeyValue className="sm:grid-cols-1" items={[
              ['Role', <Badge key="r" tone={u.role === 'ADMIN' ? 'info' : 'neutral'} icon={false}>{u.role}</Badge>],
              ['Status', <Badge key="s" tone={u.is_active ? 'good' : 'bad'}>{u.is_active ? 'Active' : `Disabled ${fmtRelative(u.disabled_at)}`}</Badge>],
              ['Created', fmtDateTime(u.created_at)],
              ['Last sign-in', fmtDateTime(u.last_login_at)],
              ['Verifications as subject', u.verification_count],
            ]} />
          </div>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader title="Biometric enrolment" actions={<>
            <Link to={`/enrolments/${u.id}`} className="text-xs font-medium text-primary hover:underline">Enrolment record</Link>
          </>} />
          <div className="space-y-4 px-5 py-4">
            <KeyValue items={[
              ['Status', <Badge key="e" tone={u.enrolment.usable ? 'good' : 'neutral'}>{humanize(u.enrolment.status)}</Badge>],
              ['Consent', u.enrolment.consent_given ? `Given ${fmtDateTime(u.enrolment.consent_timestamp)}` : 'Not given'],
              ['Enrolled', fmtDateTime(u.enrolment.enrolled_at)],
              ['Pending requests', u.pending_requests],
            ]} />
            <div className="flex flex-wrap gap-2">
              {can('enrolment:capture') && <Button size="sm" variant="outline" onClick={() => navigate(`/enrol/${u.id}`)}>{u.enrolment.is_enrolled ? 'Re-enrol' : 'Enrol'}</Button>}
              {can('verification:run') && u.enrolment.usable && (
                <Button size="sm" variant="outline" icon={<ScanFace className="h-4 w-4" />} onClick={() => navigate('/verify')}>Verify</Button>)}
            </div>
          </div>
        </Card>
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Recent verifications" subtitle={`${history.data?.total ?? 0} sessions where this user was the subject`} />
          {!history.data?.sessions.length ? <EmptyState title="No sessions yet" /> : (
            <ul className="divide-y divide-line">{history.data.sessions.map((s) => (
              <li key={s.session_id}><Link to={`/history/${s.session_id}`} className="flex items-center justify-between gap-3 px-5 py-3 text-sm hover:bg-surface-2/60">
                <span className="text-ink-2">{fmtDateTime(s.created_at)}<span className="block text-xs text-muted">{humanize(s.purpose)} · {s.operator_name}</span></span>
                <OutcomeBadge outcome={s.outcome} active={s.status === 'ACTIVE'} />
              </Link></li>))}</ul>
          )}
        </Card>
        <Card>
          <CardHeader title="Audit trail" subtitle="Changes made to or by this account" />
          {!audit.data?.entries.length ? <EmptyState title="No audit entries" /> : (
            <Table label="Audit trail" className="[&_table]:min-w-0">
              <THead><tr><Th>When</Th><Th>Action</Th><Th>By</Th></tr></THead>
              <tbody>{audit.data.entries.map((e) => (
                <Tr key={e.id}><Td className="text-xs">{fmtDateTime(e.timestamp)}</Td><Td>{humanize(e.action)}</Td><Td className="text-xs">{e.actor ?? '—'}</Td></Tr>
              ))}</tbody>
            </Table>
          )}
        </Card>
      </div>
      {edit && <EditUserModal user={u} self={self} onClose={() => setEdit(false)} onSaved={refresh} />}
      {reset && <ResetPasswordModal userId={u.id} name={u.full_name} onClose={() => setReset(false)} onDone={refresh} />}
      <ConfirmDialog open={confirm !== null} onClose={() => setConfirm(null)} onConfirm={doConfirm} loading={busy}
        danger={confirm !== 'enable'}
        confirmLabel={confirm === 'delete' ? 'Delete user' : confirm === 'disable' ? 'Disable account' : confirm === 'reset-mfa' ? 'Reset factors' : 'Enable account'}
        title={confirm === 'delete' ? `Delete ${u.full_name}?` : confirm === 'disable' ? `Disable ${u.full_name}?`
          : confirm === 'reset-mfa' ? `Reset sign-in factors of ${u.full_name}?` : `Enable ${u.full_name}?`}
        typeToConfirm={confirm === 'delete' ? u.email : undefined}
        body={<p>{confirm === 'delete' ? 'Only accounts without verification or audit history can be deleted; otherwise disable the account to preserve the audit trail.'
          : confirm === 'disable' ? 'The user is signed out everywhere and cannot sign in until re-enabled.'
            : confirm === 'reset-mfa' ? 'Removes all passkeys, authenticator apps and recovery codes (for a member who lost them). They are signed out and can set new factors after signing in with their password. You will be asked to confirm your identity.'
              : 'The user will be able to sign in again.'}</p>} />
    </>
  );
}

function EditUserModal({ user, self, onClose, onSaved }: { user: Detail; self: boolean; onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState(user.full_name);
  const [role, setRole] = useState<Role>(user.role);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const save = async () => {
    if (!name.trim()) { setError('Name is required'); return; }
    setBusy(true);
    try {
      await api.patch(`/users/${user.id}`, { full_name: name.trim(), ...(role !== user.role ? { role } : {}) });
      toast.success('User updated');
      onSaved();
      onClose();
    } catch (err) {
      setError(apiError(err).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal open onClose={onClose} title="Edit user"
      footer={<><Button variant="outline" onClick={onClose}>Cancel</Button><Button onClick={save} loading={busy}>Save changes</Button></>}>
      <div className="space-y-4">
        {error && <Callout tone="bad">{error}</Callout>}
        <Field label="Full name" htmlFor="eu-name"><Input id="eu-name" value={name} onChange={(e) => setName(e.target.value)} /></Field>
        <Field label="Role" htmlFor="eu-role" hint={self ? 'You cannot change your own role.' : 'Changing the role signs the user out of all sessions.'}>
          <Select id="eu-role" value={role} disabled={self} onChange={(e) => setRole(e.target.value as Role)}>
            <option value="ADMIN">Administrator</option><option value="OPERATOR">Operator</option><option value="VIEWER">Viewer</option>
          </Select>
        </Field>
      </div>
    </Modal>
  );
}

function ResetPasswordModal({ userId, name, onClose, onDone }: { userId: string; name: string; onClose: () => void; onDone: () => void }) {
  const [pw, setPw] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.post(`/users/${userId}/reset-password`, { new_password: pw });
      toast.success('Password reset — all sessions signed out');
      onDone();
      onClose();
    } catch (err) {
      setError(apiError(err).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal open onClose={onClose} title={`Reset password for ${name}`} description="Communicate the new password securely."
      footer={<><Button variant="outline" onClick={onClose}>Cancel</Button><Button onClick={save} loading={busy} disabled={pw.length < 10}>Reset password</Button></>}>
      <div className="space-y-3">
        {error && <Callout tone="bad">{error}</Callout>}
        <Field label="New password" htmlFor="rp-pw" hint="10+ characters, at least three of lowercase, uppercase, digit, symbol.">
          <Input id="rp-pw" type="password" autoComplete="new-password" value={pw} onChange={(e) => setPw(e.target.value)} />
        </Field>
      </div>
    </Modal>
  );
}
