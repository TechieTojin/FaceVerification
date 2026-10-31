import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { MonitorSmartphone, Plus } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtRelative, humanize } from '../lib/format';
import type { Device } from '../lib/types';
import { useCan } from '../store/auth';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Input, Select, Switch } from '../components/ui/form';
import { Modal } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { CopyField } from '../components/CopyField';

export const CONNECTION_TONE: Record<string, 'good' | 'warn' | 'bad' | 'neutral'> = {
  ONLINE: 'good', OFFLINE: 'warn', NEVER_CONNECTED: 'neutral', PENDING: 'neutral', REVOKED: 'bad',
};

export default function Devices() {
  const can = useCan();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const q = useApi<{ devices: Device[] }>('/devices');
  const [form, setForm] = useState<{ name: string; kind: Device['kind']; location: string; offline: boolean; hours: number } | null>(null);
  const [created, setCreated] = useState<Device | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    if (!form) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.post<Device>('/devices', { name: form.name, kind: form.kind, location: form.location,
        offline_enabled: form.offline, offline_validity_hours: form.hours });
      setCreated(r.data);
      setForm(null);
      void qc.invalidateQueries({ queryKey: ['/devices'] });
    } catch (err) {
      setError(apiError(err).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader title="Devices" description="Managed kiosks and browser terminals. Each device proves possession of a key it generated during provisioning; this is not hardware attestation."
        actions={can('devices:manage') && <Button icon={<Plus className="h-4 w-4" />} data-testid="register-device"
          onClick={() => setForm({ name: '', kind: 'KIOSK', location: '', offline: false, hours: 24 })}>Register device</Button>} />
      <Card>
        {q.isLoading ? <LoadingBlock rows={4} /> : !q.data?.devices.length
          ? <EmptyState icon={<MonitorSmartphone className="h-5 w-5" />} title="No devices registered" body="Register a kiosk to verify people on site, including offline." /> : (
            <Table label="Devices">
              <THead><tr><Th>Device</Th><Th>Type</Th><Th>Status</Th><Th>Last seen</Th><Th>Software</Th><Th>Offline</Th></tr></THead>
              <tbody>
                {q.data.devices.map((d) => (
                  <Tr key={d.id} onClick={() => navigate(`/devices/${d.id}`)} className="cursor-pointer">
                    <Td><span className="font-medium text-ink">{d.name}</span>{d.location && <span className="block text-xs text-muted">{d.location}</span>}</Td>
                    <Td>{d.kind === 'KIOSK' ? 'Managed kiosk' : 'Browser terminal'}</Td>
                    <Td><Badge tone={CONNECTION_TONE[d.connection] ?? 'neutral'}>{humanize(d.connection)}</Badge></Td>
                    <Td>{d.last_seen_at ? fmtRelative(d.last_seen_at) : '—'}</Td>
                    <Td>{d.software_version || '—'}</Td>
                    <Td>{d.offline_enabled ? `${d.offline_validity_hours} h` : 'No'}</Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          )}
      </Card>

      <Modal open={!!form} onClose={() => setForm(null)} title="Register device"
        description="Registering a device is a high-risk action and asks you to confirm your identity.">
        {form && (
          <form onSubmit={create} className="space-y-4">
            {error && <Callout tone="bad">{error}</Callout>}
            <Field label="Name" htmlFor="dev-name"><Input id="dev-name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required /></Field>
            <Field label="Type" htmlFor="dev-kind">
              <Select id="dev-kind" value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value as Device['kind'], offline: false })}>
                <option value="KIOSK">Managed kiosk (kiosk agent)</option><option value="BROWSER_TERMINAL">Browser terminal</option>
              </Select>
            </Field>
            <Field label="Location" htmlFor="dev-loc"><Input id="dev-loc" value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} /></Field>
            {form.kind === 'KIOSK' && (
              <>
                <Switch checked={form.offline} onChange={(v) => setForm({ ...form, offline: v })} label="Allow offline verification"
                  description="Templates of explicitly authorised people are provisioned to the kiosk, encrypted to its key, for a limited time." />
                {form.offline && (
                  <Field label="Offline validity (hours)" htmlFor="dev-hours" hint="Capped by the organisation policy and the server maximum.">
                    <Input id="dev-hours" type="number" min={1} max={72} value={form.hours} onChange={(e) => setForm({ ...form, hours: Number(e.target.value) })} />
                  </Field>
                )}
              </>
            )}
            <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setForm(null)}>Cancel</Button>
              <Button type="submit" loading={busy} disabled={form.name.trim().length < 2}>Register</Button></div>
          </form>
        )}
      </Modal>
      <Modal open={!!created} onClose={() => setCreated(null)} title="Provision the device"
        description="Use this one-time code on the device within 15 minutes. It is shown only now."
        footer={<Button onClick={() => { const id = created?.id; setCreated(null); if (id) navigate(`/devices/${id}`); }}>Done</Button>}>
        {created?.provisioning_code && (
          <div className="space-y-3 text-sm text-ink-2" data-testid="provisioning-code">
            <CopyField value={created.provisioning_code} label="Provisioning code" />
            {created.kind === 'KIOSK' && (
              <p>On the kiosk: <code className="rounded bg-surface-2 px-1 py-0.5 text-xs">python -m kiosk provision --server {window.location.origin} --code {created.provisioning_code}</code></p>
            )}
          </div>
        )}
      </Modal>
    </>
  );
}
