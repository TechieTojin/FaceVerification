import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, KeyRound, ShieldAlert, Users } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtRelative, humanize } from '../lib/format';
import type { Device, OfflineEventRow, UserRow } from '../lib/types';
import { useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, EmptyState, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { ConfirmDialog, Modal } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { CopyField } from '../components/CopyField';
import { UserPicker } from '../components/UserPicker';
import { OutcomeBadge } from '../components/OutcomeBadge';
import { CONNECTION_TONE } from './Devices';

export default function DeviceDetail() {
  const { deviceId = '' } = useParams();
  const can = useCan();
  const qc = useQueryClient();
  const q = useApi<Device>(`/devices/${deviceId}`);
  const events = useApi<{ events: OfflineEventRow[] }>(q.data?.offline_enabled ? `/devices/${deviceId}/offline-events` : null);
  const [revoking, setRevoking] = useState(false);
  const [code, setCode] = useState<string | null>(null);
  const [subjects, setSubjects] = useState<{ id: string; name: string }[] | null>(null);
  const [pick, setPick] = useState<UserRow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const refresh = () => { void qc.invalidateQueries({ queryKey: [`/devices/${deviceId}`] }); void qc.invalidateQueries({ queryKey: ['/devices'] }); };

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setError(null);
    try { await fn(); } catch (err) { setError(apiError(err).message); } finally { setBusy(false); }
  };

  const d = q.data;
  if (q.isLoading || !d) return q.isError ? <Callout tone="bad">{apiError(q.error).message}</Callout> : <LoadingBlock rows={6} />;

  const saveSubjects = () => subjects && run(async () => {
    await api.put(`/devices/${d.id}/offline-subjects`, { user_ids: subjects.map((s) => s.id) });
    toast.success('Offline subjects updated', 'The kiosk receives the new list at its next sync.');
    setSubjects(null);
    refresh();
  });

  return (
    <>
      <Link to="/devices" className="mb-3 inline-flex items-center gap-1 text-sm text-muted hover:text-ink"><ArrowLeft className="h-4 w-4" />Devices</Link>
      <PageHeader title={d.name} description={d.kind === 'KIOSK' ? 'Managed kiosk (kiosk agent)' : 'Browser terminal'}
        actions={can('devices:manage') && d.status !== 'REVOKED' && (
          <div className="flex gap-2">
            {d.status === 'PENDING' && <Button variant="outline" icon={<KeyRound className="h-4 w-4" />} loading={busy}
              onClick={() => void run(async () => setCode((await api.post<Device>(`/devices/${d.id}/provisioning-code`)).data.provisioning_code ?? null))}>
              New provisioning code</Button>}
            <Button variant="danger" icon={<ShieldAlert className="h-4 w-4" />} onClick={() => setRevoking(true)} data-testid="revoke-device">Revoke</Button>
          </div>
        )} />
      {error && <Callout tone="bad" className="mb-4">{error}</Callout>}
      {d.status === 'REVOKED' && <Callout tone="bad" className="mb-4" title="Revoked">{d.revoke_reason || 'This device can no longer connect.'}
        {d.offline_enabled && ' A disconnected kiosk keeps working only until its offline validity ends or it reconnects.'}</Callout>}
      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Identity & health" />
          <div className="p-5">
            <KeyValue items={[
              ['Status', <Badge key="s" tone={CONNECTION_TONE[d.connection] ?? 'neutral'}>{humanize(d.connection)}</Badge>],
              ['Last seen', d.last_seen_at ? `${fmtDateTime(d.last_seen_at)} (${fmtRelative(d.last_seen_at)})` : 'Never'],
              ['Software', d.software_version || '—'],
              ['Platform', d.platform || '—'],
              ['Key fingerprint', d.key_fingerprint ? <code key="k" className="text-xs">{d.key_fingerprint.slice(0, 32)}…</code> : 'Not provisioned'],
              ['Key protection', `${d.key_protection ?? '—'} (reported by the device; not hardware-attested)`],
              ['Activated', fmtDateTime(d.activated_at)],
              ['Health', d.health ? Object.entries(d.health).map(([k, v]) => `${humanize(k)}: ${String(v)}`).join(' · ') : '—'],
            ]} />
          </div>
        </Card>

        {d.offline_enabled && (
          <Card>
            <CardHeader title="Offline verification" subtitle={`Templates expire ${d.offline_validity_hours} h after each sync.`}
              actions={can('devices:manage') && d.status !== 'REVOKED' && <Button size="sm" variant="outline" icon={<Users className="h-4 w-4" />}
                onClick={() => setSubjects((d.offline_subjects ?? []).map((s) => ({ id: s.user_id, name: s.full_name })))} data-testid="edit-offline-subjects">
                Authorised people</Button>} />
            <div className="space-y-3 p-5 text-sm">
              <KeyValue items={[
                ['Authorised people', String(d.offline_subjects?.length ?? 0)],
                ['Latest bundle', d.latest_bundle ? `${d.latest_bundle.subject_count} people · issued ${fmtRelative(d.latest_bundle.issued_at)}` : 'Not synced yet'],
                ['Valid until', d.latest_bundle ? fmtDateTime(d.latest_bundle.valid_until) : '—'],
                ['Synced events', String(d.offline_events ?? 0)],
              ]} />
              <p className="text-xs text-muted">Only the people listed here are provisioned. Revoking an enrolment or the device takes effect on the kiosk at its next connection.</p>
            </div>
          </Card>
        )}

        {d.offline_enabled && (
          <Card className="xl:col-span-2">
            <CardHeader title="Offline synchronisation log" subtitle="Events recorded on the kiosk while offline, with server-side integrity checks." />
            {!events.data?.events.length ? <EmptyState title="No offline events synced yet" /> : (
              <Table label="Offline events">
                <THead><tr><Th>#</Th><Th>Occurred</Th><Th>Received</Th><Th>Person</Th><Th>Outcome</Th><Th>Integrity</Th></tr></THead>
                <tbody>{events.data.events.map((e) => (
                  <Tr key={e.event_id}>
                    <Td className="tabular">{e.seq}</Td><Td>{fmtDateTime(e.occurred_at)}</Td><Td>{fmtRelative(e.received_at)}</Td>
                    <Td>{e.subject ?? '—'}</Td>
                    <Td>{e.outcome ? <OutcomeBadge outcome={e.outcome} /> : humanize(e.type)}{e.reason_code && <span className="block text-xs text-muted">{humanize(e.reason_code)}</span>}</Td>
                    <Td><Badge tone={e.integrity === 'OK' ? 'good' : 'bad'}>{humanize(e.integrity)}</Badge></Td>
                  </Tr>))}</tbody>
              </Table>
            )}
          </Card>
        )}

        <Card className={d.offline_enabled ? '' : 'xl:col-span-1'}>
          <CardHeader title="Cameras" />
          {!d.cameras?.length ? <EmptyState title="No cameras associated" body={<Link to="/cameras" className="text-primary hover:underline">Manage cameras</Link>} /> : (
            <ul className="divide-y divide-line">{d.cameras.map((c) => (
              <li key={c.id} className="flex items-center justify-between px-5 py-2.5 text-sm"><span>{c.label}</span>
                <Badge tone={c.connection === 'ONLINE' ? 'good' : 'neutral'}>{humanize(c.connection)}</Badge></li>))}</ul>
          )}
        </Card>
        <Card>
          <CardHeader title="Audit history" />
          <ul className="divide-y divide-line">{(d.activity ?? []).map((a, i) => (
            <li key={i} className="flex justify-between gap-2 px-5 py-2.5 text-sm"><span className="text-ink-2">{humanize(a.action)} · {a.actor}</span>
              <span className="text-xs text-muted">{fmtRelative(a.at)}</span></li>))}</ul>
        </Card>
      </div>

      <ConfirmDialog open={revoking} onClose={() => setRevoking(false)} danger confirmLabel="Revoke device" loading={busy}
        title={`Revoke ${d.name}?`} noteLabel="Reason" noteRequired
        body="The device's key stops working immediately. A kiosk that is offline right now keeps its provisioned templates until its validity period ends or it reconnects, when they are erased."
        onConfirm={(note) => void run(async () => { await api.post(`/devices/${d.id}/revoke`, { reason: note }); setRevoking(false); refresh(); })} />
      <Modal open={!!code} onClose={() => setCode(null)} title="New provisioning code" description="Valid for 15 minutes, shown only now."
        footer={<Button onClick={() => setCode(null)}>Done</Button>}>
        {code && <CopyField value={code} label="Provisioning code" />}
      </Modal>
      <Modal open={!!subjects} onClose={() => setSubjects(null)} title="People authorised for offline verification" size="lg"
        description="Only enrolled members can be added. Changing this list asks you to confirm your identity.">
        {subjects && (
          <div className="space-y-4">
            <div className="flex items-end gap-2">
              <div className="flex-1"><UserPicker value={pick} onChange={setPick} enrolledOnly label="Add person" /></div>
              <Button variant="outline" disabled={!pick || subjects.some((s) => s.id === pick.id)}
                onClick={() => { if (pick) { setSubjects([...subjects, { id: pick.id, name: pick.full_name }]); setPick(null); } }}>Add</Button>
            </div>
            <ul className="max-h-72 divide-y divide-line overflow-y-auto rounded-lg border border-line">
              {subjects.length === 0 && <li className="p-3 text-sm text-muted">Nobody yet.</li>}
              {subjects.map((s) => (
                <li key={s.id} className="flex items-center justify-between px-3 py-2 text-sm">{s.name}
                  <Button size="sm" variant="ghost" onClick={() => setSubjects(subjects.filter((x) => x.id !== s.id))}>Remove</Button></li>))}
            </ul>
            <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setSubjects(null)}>Cancel</Button>
              <Button loading={busy} onClick={() => void saveSubjects()} data-testid="save-offline-subjects">Save</Button></div>
          </div>
        )}
      </Modal>
    </>
  );
}
