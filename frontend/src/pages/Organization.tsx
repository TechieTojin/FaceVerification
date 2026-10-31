import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Building2, MailPlus, Plus } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtRelative, humanize } from '../lib/format';
import type { Invitation, Me, Organization as Org, Role } from '../lib/types';
import { useAuth, useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, EmptyState, KeyValue, LoadingBlock, PageHeader, StatCard } from '../components/ui/primitives';
import { Field, Input, Select } from '../components/ui/form';
import { ConfirmDialog, Modal } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { CopyField } from '../components/CopyField';

const STATUS_TONE: Record<string, 'good' | 'warn' | 'bad' | 'neutral'> = {
  PENDING: 'warn', ACCEPTED: 'good', REVOKED: 'neutral', EXPIRED: 'neutral',
};


export default function OrganizationPage() {
  const can = useCan();
  const user = useAuth((s) => s.user);
  const setUser = useAuth((s) => s.setUser);
  const qc = useQueryClient();
  const current = useApi<Org>('/organizations/current');
  const orgs = useApi<{ organizations: Org[]; can_create: boolean }>('/organizations');
  const invites = useApi<{ invitations: Invitation[] }>(can('org:manage') ? '/organizations/current/invitations' : null);
  const [inviting, setInviting] = useState(false);
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<Role>('OPERATOR');
  const [created, setCreated] = useState<Invitation | null>(null);
  const [creatingOrg, setCreatingOrg] = useState(false);
  const [orgName, setOrgName] = useState('');
  const [orgSlug, setOrgSlug] = useState('');
  const [rename, setRename] = useState<string | null>(null);
  const [suspend, setSuspend] = useState<Org | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try { await fn(); } catch (err) { setError(apiError(err).message); } finally { setBusy(false); }
  };

  const invite = (e: FormEvent) => {
    e.preventDefault();
    void run(async () => {
      const r = await api.post<Invitation>('/organizations/current/invitations', { email, role });
      setCreated(r.data);
      setInviting(false);
      setEmail('');
      void qc.invalidateQueries({ queryKey: ['/organizations/current/invitations'] });
    });
  };

  const createOrg = (e: FormEvent) => {
    e.preventDefault();
    void run(async () => {
      await api.post('/organizations', { name: orgName, slug: orgSlug || undefined });
      toast.success('Organisation created', 'You are its first administrator. Switch to it from the header.');
      setCreatingOrg(false);
      setOrgName('');
      setOrgSlug('');
      void qc.invalidateQueries({ queryKey: ['/organizations'] });
      const me = await api.get<Me>('/auth/me');
      setUser(me.data);
    });
  };

  const saveName = (e: FormEvent) => {
    e.preventDefault();
    void run(async () => {
      await api.patch('/organizations/current', { name: rename });
      setRename(null);
      void qc.invalidateQueries({ queryKey: ['/organizations/current'] });
      const me = await api.get<Me>('/auth/me');
      setUser(me.data);
    });
  };

  const c = current.data;
  return (
    <>
      <PageHeader title="Organisation" description="Your active organisation, its members and invitations. Data of one organisation is never visible in another."
        actions={can('org:manage') && <Button icon={<MailPlus className="h-4 w-4" />} onClick={() => setInviting(true)} data-testid="invite-member">Invite member</Button>} />
      {error && <Callout tone="bad" className="mb-4">{error}</Callout>}
      {current.isLoading || !c ? <LoadingBlock rows={4} /> : (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard label="Members" value={c.members} hint={`${c.active_members} active`} />
            <StatCard label="Verification sessions" value={c.sessions} />
            <StatCard label="Registered devices" value={c.devices} />
            <StatCard label="Your role" value={c.role ?? '—'} />
          </div>
          <Card>
            <CardHeader title={c.name} subtitle={`Slug: ${c.slug}`}
              actions={can('org:manage') && <Button size="sm" variant="outline" onClick={() => setRename(c.name)}>Rename</Button>} />
            <div className="p-5">
              <KeyValue items={[
                ['Status', <Badge key="s" tone={c.status === 'ACTIVE' ? 'good' : 'bad'}>{c.status}</Badge>],
                ['Created', fmtDateTime(c.created_at)],
                ['Members', <Link key="m" to="/users" className="text-primary hover:underline">Manage members →</Link>],
                ['Policies', <Link key="p" to="/settings" className="text-primary hover:underline">Organisation policy →</Link>],
              ]} />
            </div>
          </Card>

          {can('org:manage') && (
            <Card>
              <CardHeader title="Invitations" subtitle="Invitation links expire after 7 days and work once." />
              {invites.isLoading ? <LoadingBlock rows={3} /> : !invites.data?.invitations.length
                ? <EmptyState icon={<MailPlus className="h-5 w-5" />} title="No invitations yet" /> : (
                  <Table label="Invitations">
                    <THead><tr><Th>Email</Th><Th>Role</Th><Th>Status</Th><Th>Expires</Th><Th className="text-right">Actions</Th></tr></THead>
                    <tbody>
                      {invites.data.invitations.map((i) => (
                        <Tr key={i.id}>
                          <Td className="font-medium text-ink">{i.email}</Td>
                          <Td>{i.role}</Td>
                          <Td><Badge tone={STATUS_TONE[i.status] ?? 'neutral'}>{humanize(i.status)}</Badge></Td>
                          <Td>{i.status === 'PENDING' ? fmtRelative(i.expires_at) : '—'}</Td>
                          <Td className="text-right">{i.status === 'PENDING' && (
                            <Button size="sm" variant="ghost" onClick={() => void run(async () => {
                              await api.post(`/organizations/current/invitations/${i.id}/revoke`);
                              void qc.invalidateQueries({ queryKey: ['/organizations/current/invitations'] });
                            })}>Revoke</Button>)}</Td>
                        </Tr>
                      ))}
                    </tbody>
                  </Table>
                )}
            </Card>
          )}

          <Card>
            <CardHeader title={orgs.data?.can_create ? 'All organisations (platform administrator)' : 'Your organisations'}
              actions={orgs.data?.can_create && <Button size="sm" icon={<Plus className="h-4 w-4" />} onClick={() => setCreatingOrg(true)}
                data-testid="create-org">New organisation</Button>} />
            {orgs.isLoading ? <LoadingBlock rows={3} /> : (
              <Table label="Organisations">
                <THead><tr><Th>Name</Th><Th>Status</Th><Th>Members</Th><Th>Sessions</Th><Th>Your role</Th>
                  {orgs.data?.can_create && <Th className="text-right">Platform</Th>}</tr></THead>
                <tbody>
                  {orgs.data?.organizations.map((o) => (
                    <Tr key={o.id}>
                      <Td><span className="inline-flex items-center gap-2 font-medium text-ink"><Building2 className="h-4 w-4 text-muted" />{o.name}</span>
                        {o.id === user?.organization?.id && <Badge tone="info" className="ml-2">Active</Badge>}</Td>
                      <Td><Badge tone={o.status === 'ACTIVE' ? 'good' : 'bad'}>{o.status}</Badge></Td>
                      <Td className="tabular">{o.members}</Td>
                      <Td className="tabular">{o.sessions}</Td>
                      <Td>{o.role ?? 'Not a member'}</Td>
                      {orgs.data?.can_create && (
                        <Td className="text-right">
                          {o.status === 'ACTIVE'
                            ? <Button size="sm" variant="ghost" disabled={o.id === user?.organization?.id} onClick={() => setSuspend(o)}>Suspend</Button>
                            : <Button size="sm" variant="ghost" onClick={() => void run(async () => {
                              await api.post(`/organizations/${o.id}/activate`);
                              void qc.invalidateQueries({ queryKey: ['/organizations'] });
                            })}>Reactivate</Button>}
                        </Td>
                      )}
                    </Tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Card>
        </div>
      )}

      <Modal open={inviting} onClose={() => setInviting(false)} title="Invite a member"
        description="Existing accounts sign in and accept; new people choose a password when they accept.">
        <form onSubmit={invite} className="space-y-4">
          <Field label="Email" htmlFor="invite-email"><Input id="invite-email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></Field>
          <Field label="Role" htmlFor="invite-role">
            <Select id="invite-role" value={role} onChange={(e) => setRole(e.target.value as Role)}>
              <option value="VIEWER">Viewer</option><option value="OPERATOR">Operator</option><option value="ADMIN">Administrator</option>
            </Select>
          </Field>
          <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setInviting(false)}>Cancel</Button>
            <Button type="submit" loading={busy} disabled={!email.includes('@')}>Create invitation</Button></div>
        </form>
      </Modal>
      <Modal open={!!created} onClose={() => setCreated(null)} title="Invitation created"
        description={created?.email_status === 'SENT' ? `An email was sent to ${created.email}.`
          : 'Email delivery is not available, so share this link with the person yourself. It is shown only now.'}
        footer={<Button onClick={() => setCreated(null)}>Done</Button>}>
        {created?.link && <CopyField value={created.link} label="Invitation link" />}
      </Modal>
      <Modal open={creatingOrg} onClose={() => setCreatingOrg(false)} title="New organisation"
        description="Creating an organisation is a high-risk action and asks you to confirm your identity.">
        <form onSubmit={createOrg} className="space-y-4">
          <Field label="Name" htmlFor="org-name"><Input id="org-name" value={orgName} onChange={(e) => setOrgName(e.target.value)} required /></Field>
          <Field label="Slug (optional)" htmlFor="org-slug" hint="Lowercase letters, digits and hyphens.">
            <Input id="org-slug" value={orgSlug} onChange={(e) => setOrgSlug(e.target.value.toLowerCase())} /></Field>
          <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setCreatingOrg(false)}>Cancel</Button>
            <Button type="submit" loading={busy} disabled={orgName.trim().length < 2}>Create</Button></div>
        </form>
      </Modal>
      <Modal open={rename !== null} onClose={() => setRename(null)} title="Rename organisation">
        <form onSubmit={saveName} className="space-y-4">
          <Field label="Name" htmlFor="rename-org"><Input id="rename-org" value={rename ?? ''} onChange={(e) => setRename(e.target.value)} /></Field>
          <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setRename(null)}>Cancel</Button>
            <Button type="submit" loading={busy} disabled={(rename ?? '').trim().length < 2}>Save</Button></div>
        </form>
      </Modal>
      <ConfirmDialog open={!!suspend} onClose={() => setSuspend(null)} danger confirmLabel="Suspend" loading={busy}
        title={`Suspend ${suspend?.name ?? ''}?`} typeToConfirm={suspend?.slug}
        body="Members lose access immediately and integration keys stop working until the organisation is reactivated. No data is deleted."
        onConfirm={() => void run(async () => {
          await api.post(`/organizations/${suspend!.id}/suspend`);
          setSuspend(null);
          void qc.invalidateQueries({ queryKey: ['/organizations'] });
        })} />
    </>
  );
}
