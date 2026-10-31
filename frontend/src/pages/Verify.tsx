import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { QrCode } from 'lucide-react';
import { api } from '../lib/api';
import { useApi } from '../lib/query';
import type { Camera, SessionView, UserRow } from '../lib/types';
import { Card, PageHeader } from '../components/ui/primitives';
import { Field, Select } from '../components/ui/form';
import { UserPicker } from '../components/UserPicker';
import { LiveCapturePanel } from '../features/camera/LiveCapturePanel';
import { useLiveCapture } from '../features/camera/useLiveCapture';

const HEARTBEAT_MS = 20_000;

export default function Verify() {
  const [subject, setSubject] = useState<UserRow | null>(null);
  const [cameraId, setCameraId] = useState('');
  const cams = useApi<{ cameras: Camera[] }>('/cameras', { include_disabled: false });
  const qc = useQueryClient();
  const start = useCallback(async () => {
    if (!subject) throw new Error('Select a subject first');
    return (await api.post<SessionView>('/sessions', { target_user_id: subject.id, camera_id: cameraId || undefined })).data;
  }, [subject, cameraId]);
  const live = useLiveCapture(start, () => void qc.invalidateQueries({ queryKey: ['/dashboard/summary'] }));
  const busy = live.session.phase === 'running' || live.session.phase === 'starting';
  const registered = cams.data?.cameras.find((c) => c.id === cameraId);
  const { camera } = live;

  // Use the local device whose label matches the registered camera's hardware label.
  useEffect(() => {
    if (!registered?.hardware_label) return;
    const match = camera.devices.find((d) => d.label === registered.hardware_label);
    if (match && match.deviceId !== camera.deviceId) camera.setDeviceId(match.deviceId);
  }, [registered, camera]);

  // Heartbeat: tells administrators this registered camera is in use (no images are sent).
  useEffect(() => {
    if (!cameraId || camera.status !== 'active') return;
    const label = camera.devices.find((d) => d.deviceId === camera.deviceId)?.label;
    const beat = () => void api.post(`/cameras/${cameraId}/heartbeat`, { state: 'streaming', hardware_label: label }).catch(() => undefined);
    beat();
    const t = window.setInterval(beat, HEARTBEAT_MS);
    return () => window.clearInterval(t);
  }, [cameraId, camera.status, camera.deviceId, camera.devices]);

  return (
    <>
      <PageHeader title="Live verification"
        description="1:1 check of the person in front of the camera against the selected user's enrolled template. The server issues a random liveness challenge and makes every decision."
        actions={<Link to="/verify/qr" className="inline-flex items-center gap-1.5 text-sm font-medium text-primary hover:underline">
          <QrCode className="h-4 w-4" /> Verify on the person's phone instead</Link>} />
      <LiveCapturePanel live={live} startLabel={subject ? `Verify ${subject.full_name}` : 'Select a subject to verify'}
        canStart={!!subject}
        side={(
          <Card className="space-y-4 p-5">
            <UserPicker value={subject} onChange={setSubject} enrolledOnly disabled={busy} label="Person to verify" />
            {!!cams.data?.cameras.length && (
              <Field label="Registered camera (optional)" htmlFor="reg-cam" hint="Records which desk camera was used.">
                <Select id="reg-cam" value={cameraId} disabled={busy} onChange={(e) => setCameraId(e.target.value)}>
                  <option value="">Not recorded</option>
                  {cams.data.cameras.map((c) => <option key={c.id} value={c.id}>{c.label}{c.location ? ` — ${c.location}` : ''}</option>)}
                </Select>
              </Field>
            )}
          </Card>
        )}
        idleHint={<p className="text-xs text-muted">The camera starts when you begin. Face images are processed in memory and never stored.</p>} />
    </>
  );
}
