import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Camera as CameraIcon, Plus, Video } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtRelative, humanize } from '../lib/format';
import type { Camera, Device } from '../lib/types';
import { useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, EmptyState, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Input, Select } from '../components/ui/form';
import { Modal } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { OutcomeBadge } from '../components/OutcomeBadge';

const CONN_TONE: Record<Camera['connection'], 'good' | 'warn' | 'neutral' | 'bad'> = {
  ONLINE: 'good', OFFLINE: 'warn', NEVER_SEEN: 'neutral', DISABLED: 'bad',
};
export const connLabel = (c: string) => humanize(c === 'NEVER_SEEN' ? 'never_seen' : c);

interface LocalCam { deviceId: string; label: string }

/** List the cameras this browser may use (labels need camera permission). The server never sees other devices. */
async function listLocalCameras(): Promise<LocalCam[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  let devices = await navigator.mediaDevices.enumerateDevices();
  if (devices.some((d) => d.kind === 'videoinput' && !d.label)) {
    const s = await navigator.mediaDevices.getUserMedia({ video: true });
    s.getTracks().forEach((t) => t.stop());
    devices = await navigator.mediaDevices.enumerateDevices();
  }
  return devices.filter((d) => d.kind === 'videoinput').map((d) => ({ deviceId: d.deviceId, label: d.label || 'Camera' }));
}

export default function Cameras() {
  const can = useCan();
  const qc = useQueryClient();
  const q = useApi<{ cameras: Camera[]; can_manage: boolean }>('/cameras');
  const devices = useApi<{ devices: Device[] }>(can('devices:read') ? '/devices' : null);
  const [form, setForm] = useState<{ id?: string; label: string; location: string; hardware_label: string; device_id: string } | null>(null);
  const [local, setLocal] = useState<LocalCam[] | null>(null);
  const [detail, setDetail] = useState<string | null>(null);
  const detailQ = useApi<Camera>(detail ? `/cameras/${detail}` : null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const refresh = () => void qc.invalidateQueries({ queryKey: ['/cameras'] });

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setError(null);
    try { await fn(); } catch (err) { setError(apiError(err).message); } finally { setBusy(false); }
  };

  const detect = () => run(async () => {
    const cams = await listLocalCameras();
    setLocal(cams);
    if (!cams.length) setError('No camera is available to this browser.');
  });

  const save = (e: FormEvent) => {
    e.preventDefault();
    if (!form) return;
    void run(async () => {
      if (form.id) {
        await api.patch(`/cameras/${form.id}`, { label: form.label, location: form.location,
          ...(form.device_id ? { device_id: form.device_id } : { clear_device: true }) });
        toast.success('Camera updated');
      } else {
        await api.post('/cameras', { label: form.label, location: form.location, hardware_label: form.hardware_label,
          device_id: form.device_id || undefined });
        toast.success('Camera registered');
      }
      setForm(null);
      refresh();
    });
  };

  const toggle = (c: Camera) => run(async () => {
    await api.post(`/cameras/${c.id}/${c.status === 'ACTIVE' ? 'disable' : 'enable'}`);
    refresh();
    void qc.invalidateQueries({ queryKey: [`/cameras/${c.id}`] });
  });

  const d = detailQ.data;
  return (
    <>
      <PageHeader title="Cameras" description="Explicitly registered cameras and verification terminals. A browser can only see cameras on its own machine; the server does not discover or stream cameras."
        actions={q.data?.can_manage && <Button icon={<Plus className="h-4 w-4" />} data-testid="register-camera"
          onClick={() => { setLocal(null); setForm({ label: '', location: '', hardware_label: '', device_id: '' }); }}>Register camera</Button>} />
      {error && <Callout tone="bad" className="mb-4">{error}</Callout>}
      <Card>
        {q.isLoading ? <LoadingBlock rows={4} /> : !q.data?.cameras.length
          ? <EmptyState icon={<CameraIcon className="h-5 w-5" />} title="No cameras registered" body="Register the cameras at your verification desks to track their status and usage." /> : (
            <Table label="Cameras">
              <THead><tr><Th>Camera</Th><Th>Location</Th><Th>Status</Th><Th>Last heartbeat</Th><Th>Last verification</Th><Th className="text-right">Actions</Th></tr></THead>
              <tbody>
                {q.data.cameras.map((c) => (
                  <Tr key={c.id} onClick={() => setDetail(c.id)} className="cursor-pointer">
                    <Td><span className="font-medium text-ink">{c.label}</span>{c.hardware_label && <span className="block text-xs text-muted">{c.hardware_label}</span>}</Td>
                    <Td>{c.location ?? '—'}</Td>
                    <Td><Badge tone={CONN_TONE[c.connection]}>{connLabel(c.connection)}</Badge></Td>
                    <Td>{c.last_heartbeat_at ? fmtRelative(c.last_heartbeat_at) : '—'}</Td>
                    <Td>{c.last_verification_at ? fmtRelative(c.last_verification_at) : '—'}</Td>
                    <Td className="text-right" onClick={(e) => e.stopPropagation()}>
                      {q.data.can_manage && (
                        <div className="flex justify-end gap-1">
                          <Button size="sm" variant="ghost" onClick={() => setForm({ id: c.id, label: c.label, location: c.location ?? '',
                            hardware_label: c.hardware_label ?? '', device_id: c.device_id ?? '' })}>Edit</Button>
                          <Button size="sm" variant="ghost" onClick={() => void toggle(c)}>{c.status === 'ACTIVE' ? 'Disable' : 'Enable'}</Button>
                        </div>
                      )}
                    </Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          )}
      </Card>

      <Modal open={!!form} onClose={() => setForm(null)} title={form?.id ? 'Edit camera' : 'Register camera'}>
        {form && (
          <form onSubmit={save} className="space-y-4">
            <Field label="Label" htmlFor="cam-label"><Input id="cam-label" value={form.label} onChange={(e) => setForm({ ...form, label: e.target.value })} required /></Field>
            <Field label="Location or area" htmlFor="cam-loc"><Input id="cam-loc" value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} /></Field>
            {!form.id && (
              <Field label="Physical camera" htmlFor="cam-hw" hint="Pick a camera attached to this computer, or type its label.">
                <div className="flex gap-2">
                  {local?.length ? (
                    <Select id="cam-hw" value={form.hardware_label} onChange={(e) => setForm({ ...form, hardware_label: e.target.value })}>
                      <option value="">Choose…</option>
                      {local.map((c) => <option key={c.deviceId} value={c.label}>{c.label}</option>)}
                    </Select>
                  ) : (
                    <Input id="cam-hw" value={form.hardware_label} onChange={(e) => setForm({ ...form, hardware_label: e.target.value })} />
                  )}
                  <Button variant="outline" icon={<Video className="h-4 w-4" />} onClick={() => void detect()}>Detect</Button>
                </div>
              </Field>
            )}
            {!!devices.data?.devices.length && (
              <Field label="Associated device (optional)" htmlFor="cam-dev">
                <Select id="cam-dev" value={form.device_id} onChange={(e) => setForm({ ...form, device_id: e.target.value })}>
                  <option value="">None</option>
                  {devices.data.devices.filter((x) => x.status !== 'REVOKED').map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
                </Select>
              </Field>
            )}
            <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setForm(null)}>Cancel</Button>
              <Button type="submit" loading={busy} disabled={!form.label.trim()}>Save</Button></div>
          </form>
        )}
      </Modal>

      <Modal open={!!detail} onClose={() => setDetail(null)} title={d?.label ?? 'Camera'} size="lg">
        {!d ? <LoadingBlock rows={4} /> : (
          <div className="space-y-5" data-testid="camera-detail">
            <KeyValue items={[
              ['Status', <Badge key="s" tone={CONN_TONE[d.connection]}>{connLabel(d.connection)}</Badge>],
              ['Hardware label', d.hardware_label ?? '—'],
              ['Location', d.location ?? '—'],
              ['Last heartbeat', d.last_heartbeat_at ? `${fmtDateTime(d.last_heartbeat_at)} (${d.last_heartbeat_by ?? ''})` : 'Never'],
              ['Capabilities', d.capabilities ? Object.entries(d.capabilities).map(([k, v]) => `${k}: ${String(v)}`).join(', ') : '—'],
            ]} />
            <div>
              <p className="mb-2 text-sm font-semibold text-ink">Recent verifications</p>
              {!d.recent_sessions?.length ? <p className="text-sm text-muted">None yet.</p> : (
                <ul className="divide-y divide-line text-sm">{d.recent_sessions.map((s) => (
                  <li key={s.session_id} className="flex items-center justify-between py-1.5">
                    <Link to={`/history/${s.session_id}`} className="text-primary hover:underline">{fmtDateTime(s.created_at)}</Link>
                    <OutcomeBadge outcome={s.outcome} />
                  </li>))}</ul>
              )}
            </div>
            <div>
              <p className="mb-2 text-sm font-semibold text-ink">Activity</p>
              <ul className="space-y-1 text-sm">{(d.activity ?? []).map((a, i) => (
                <li key={i} className="flex justify-between gap-2"><span className="text-ink-2">{humanize(a.action)} · {a.actor}</span>
                  <span className="text-xs text-muted">{fmtRelative(a.at)}</span></li>))}</ul>
            </div>
          </div>
        )}
      </Modal>
    </>
  );
}
