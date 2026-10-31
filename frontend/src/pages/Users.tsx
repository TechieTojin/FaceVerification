import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Plus, Search, Users as UsersIcon } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtRelative, humanize } from '../lib/format';
import type { Role, UserList } from '../lib/types';
import { useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Input, Select } from '../components/ui/form';
import { Modal } from '../components/ui/Modal';
import { Pagination, Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { useDebounced } from '../components/UserPicker';

export default function Users() {
  const can = useCan();
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [role, setRole] = useState('');
  const [status, setStatus] = useState('');
  const [enrolment, setEnrolment] = useState('');
  const [sort, setSort] = useState('name');
  const [page, setPage] = useState(1);
  const [creating, setCreating] = useState(false);
  const q = useApi<UserList>('/users', { search: useDebounced(search), role, status, enrolment, sort, page, page_size: 20 });

  const filter = (setter: (v: string) => void) => (v: string) => { setter(v); setPage(1); };

  return (
    <>
      <PageHeader title="Users" description="Accounts, roles and biometric enrolment status."
        actions={can('users:manage') && <Button icon={<Plus className="h-4 w-4" />} onClick={() => setCreating(true)}>New user</Button>} />
      <Card>
        <div className="flex flex-wrap items-end gap-3 border-b border-line p-4">
          <div className="relative min-w-[200px] flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden />
            <Input aria-label="Search users" className="pl-9" placeholder="Search name or email" value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }} />
          </div>
          <Select aria-label="Role" value={role} onChange={(e) => filter(setRole)(e.target.value)} className="w-36">
            <option value="">All roles</option><option value="ADMIN">Admin</option><option value="OPERATOR">Operator</option><option value="VIEWER">Viewer</option>
          </Select>
          <Select aria-label="Status" value={status} onChange={(e) => filter(setStatus)(e.target.value)} className="w-36">
            <option value="">Any status</option><option value="active">Active</option><option value="disabled">Disabled</option>
          </Select>
          <Select aria-label="Enrolment" value={enrolment} onChange={(e) => filter(setEnrolment)(e.target.value)} className="w-44">
            <option value="">Any enrolment</option><option value="ACTIVE">Enrolled</option><option value="NOT_ENROLLED">Not enrolled</option>
            <option value="REENROLMENT_REQUIRED">Re-enrolment required</option><option value="REVOKED">Revoked</option><option value="WITHDRAWN">Withdrawn</option>
          </Select>
          <Select aria-label="Sort" value={sort} onChange={(e) => setSort(e.target.value)} className="w-40">
            <option value="name">Sort: name</option><option value="email">Sort: email</option><option value="created">Sort: newest</option><option value="last_login">Sort: last login</option>
          </Select>
        </div>
        {q.isLoading ? <LoadingBlock rows={6} /> : q.isError ? <Callout tone="bad" className="m-4">{apiError(q.error).message}</Callout> :
          !q.data?.users.length ? <EmptyState icon={<UsersIcon className="h-5 w-5" />} title="No users match these filters" /> : (
            <>
              <Table label="Users">
                <THead><tr><Th>Name</Th><Th>Role</Th><Th>Status</Th><Th>Enrolment</Th><Th>Last sign-in</Th></tr></THead>
                <tbody>
                  {q.data.users.map((u) => (
                    <Tr key={u.id} onClick={() => navigate(`/users/${u.id}`)}>
                      <Td><p className="font-medium text-ink">{u.full_name}</p><p className="text-xs text-muted">{u.email}</p></Td>
                      <Td><Badge tone={u.role === 'ADMIN' ? 'info' : 'neutral'} icon={false}>{u.role}</Badge></Td>
                      <Td><Badge tone={u.is_active ? 'good' : 'bad'}>{u.is_active ? 'Active' : 'Disabled'}</Badge></Td>
                      <Td><Badge tone={u.enrolment_usable ? 'good' : u.enrolment_status === 'NOT_ENROLLED' ? 'neutral' : 'warn'}>
                        {u.enrolment_usable ? 'Enrolled' : humanize(u.enrolment_status)}</Badge></Td>
                      <Td className="text-xs">{fmtRelative(u.last_login_at)}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
              <Pagination page={page} pageSize={20} total={q.data.total} onPage={setPage} />
            </>
          )}
      </Card>
      <CreateUserModal open={creating} onClose={() => setCreating(false)} />
    </>
  );
}

function CreateUserModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: '', full_name: '', password: '', role: 'OPERATOR' as Role });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const set = (k: keyof typeof form) => (v: string) => setForm((f) => ({ ...f, [k]: v }));

  const validate = () => {
    const e: Record<string, string> = {};
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(form.email.trim())) e.email = 'Enter a valid email address';
    if (!form.full_name.trim()) e.full_name = 'Name is required';
    if (form.password.length < 10) e.password = 'At least 10 characters';
    return e;
  };

  const submit = async (ev: FormEvent) => {
    ev.preventDefault();
    const e = validate();
    setErrors(e);
    if (Object.keys(e).length) return;
    setBusy(true);
    try {
      const r = await api.post<{ id: string }>('/users', { ...form, email: form.email.trim(), full_name: form.full_name.trim() });
      toast.success('User created');
      void qc.invalidateQueries({ queryKey: ['/users'] });
      setForm({ email: '', full_name: '', password: '', role: 'OPERATOR' });
      onClose();
      navigate(`/users/${r.data.id}`);
    } catch (err) {
      const info = apiError(err);
      setErrors({ ...info.fieldErrors, form: info.message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="New user" description="Passwords need 10+ characters from at least three character classes."
      footer={<><Button variant="outline" onClick={onClose}>Cancel</Button><Button type="submit" form="create-user" loading={busy}>Create user</Button></>}>
      <form id="create-user" onSubmit={submit} className="space-y-4" noValidate>
        {errors.form && <Callout tone="bad">{errors.form}</Callout>}
        <Field label="Full name" htmlFor="nu-name" error={errors.full_name}>
          <Input id="nu-name" value={form.full_name} onChange={(e) => set('full_name')(e.target.value)} invalid={!!errors.full_name} />
        </Field>
        <Field label="Email" htmlFor="nu-email" error={errors.email}>
          <Input id="nu-email" type="email" value={form.email} onChange={(e) => set('email')(e.target.value)} invalid={!!errors.email} />
        </Field>
        <Field label="Initial password" htmlFor="nu-pw" error={errors.password} hint="Share it securely; the user should change it after first sign-in.">
          <Input id="nu-pw" type="password" autoComplete="new-password" value={form.password} onChange={(e) => set('password')(e.target.value)} invalid={!!errors.password} />
        </Field>
        <Field label="Role" htmlFor="nu-role">
          <Select id="nu-role" value={form.role} onChange={(e) => set('role')(e.target.value)}>
            <option value="OPERATOR">Operator — enrol and verify people</option>
            <option value="VIEWER">Viewer — read-only dashboards and history</option>
            <option value="ADMIN">Administrator — full control</option>
          </Select>
        </Field>
      </form>
    </Modal>
  );
}
