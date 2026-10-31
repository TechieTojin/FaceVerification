import { useCallback, useEffect, useState } from 'react';
import { CloudOff, Cloud, RefreshCw, ShieldAlert, ShieldCheck } from 'lucide-react';
import { apiError } from '../lib/api';
import { fmtDateTime, fmtRelative, humanize } from '../lib/format';
import type { KioskStatus, SessionView } from '../lib/types';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, KeyValue, LoadingBlock } from '../components/ui/primitives';
import { Select } from '../components/ui/form';
import { OutcomeBadge } from '../components/OutcomeBadge';
import { FaceCheck } from '../features/camera/FaceCheck';
import { kioskHttp, kioskTransport } from '../features/camera/transport';

interface KioskEvent { event_id: string; seq: number; type: string; occurred_at: string; outcome: string | null;
  reason_code: string | null; subject_user_id: string | null; synced: boolean }

/**
 * Offline kiosk mode. Served by the kiosk agent on 127.0.0.1; every decision is made on this device by the
 * agent's local models. Works with no connection to the central server.
 */
export default function Kiosk() {
  const [status, setStatus] = useState<KioskStatus | null>(null);
  const [subjects, setSubjects] = useState<{ user_id: string; display_name: string }[]>([]);
  const [events, setEvents] = useState<KioskEvent[]>([]);
  const [subject, setSubject] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [run, setRun] = useState(0);

  const load = useCallback(async () => {
    try {
      const [s, sub, ev] = await Promise.all([
        kioskHttp.get<KioskStatus>('/status'),
        kioskHttp.get<{ subjects: { user_id: string; display_name: string }[] }>('/subjects'),
        kioskHttp.get<{ events: KioskEvent[] }>('/events'),
      ]);
      setStatus(s.data);
      setSubjects(sub.data.subjects);
      setEvents(ev.data.events);
      setError(null);
    } catch (err) {
      setError(apiError(err, 'The kiosk agent is not responding').message);
    }
  }, []);

  useEffect(() => {
    void load();
    const t = window.setInterval(() => void load(), 5000);
    return () => window.clearInterval(t);
  }, [load]);

  const sync = async () => {
    setSyncing(true);
    try { await kioskHttp.post('/sync'); } catch (err) { setError(apiError(err).message); } finally { setSyncing(false); void load(); }
  };

  const start = useCallback(async (): Promise<SessionView> =>
    (await kioskHttp.post<SessionView>('/sessions', { user_id: subject })).data, [subject]);

  const names = Object.fromEntries(subjects.map((s) => [s.user_id, s.display_name]));
  const valid = status?.bundle.valid && !status.revoked;

  return (
    <div className="min-h-full bg-page">
      <header className="flex flex-wrap items-center gap-3 border-b border-line bg-surface px-4 py-3 sm:px-6">
        <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-white"><ShieldCheck className="h-5 w-5" /></span>
        <div className="min-w-0 flex-1 leading-tight">
          <p className="font-semibold text-ink">{status?.name || 'Verification kiosk'}</p>
          <p className="truncate text-xs text-muted">{status?.organization}</p>
        </div>
        {status && (status.online
          ? <Badge tone="good"><Cloud className="mr-1 inline h-3.5 w-3.5" />Online</Badge>
          : <Badge tone="warn" className="text-sm" ><CloudOff className="mr-1 inline h-3.5 w-3.5" />Offline mode</Badge>)}
        <Button size="sm" variant="outline" icon={<RefreshCw className="h-3.5 w-3.5" />} loading={syncing} onClick={() => void sync()} data-testid="kiosk-sync">Sync now</Button>
      </header>
      <main className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6" data-testid="kiosk-page">
        {error && <Callout tone="bad">{error}</Callout>}
        {!status ? <LoadingBlock rows={4} /> : (
          <>
            {status.revoked && <Callout tone="bad" title="This kiosk has been revoked">Its provisioned templates were erased. Contact an administrator.</Callout>}
            {!status.revoked && !status.bundle.valid && (
              <Callout tone="warn" title={status.bundle.reason === 'EXPIRED' ? 'Offline authorisation expired' : 'Not provisioned yet'}>
                {status.bundle.reason === 'EXPIRED'
                  ? 'Reconnect this kiosk to the network so it can renew its authorisation. Verification is blocked until then.'
                  : 'Connect to the network and sync to receive the people authorised for this kiosk.'}
              </Callout>
            )}
            {valid && (
              <FaceCheck key={run} transport={kioskTransport} start={start} startLabel={subject ? `Verify ${names[subject] ?? ''}` : 'Choose a person first'}
                canStart={!!subject} onTerminal={() => void load()} onAgain={() => setRun((n) => n + 1)}
                side={(
                  <Card className="space-y-3 p-5">
                    <label htmlFor="kiosk-subject" className="text-sm font-medium text-ink">Who is being verified?</label>
                    <Select id="kiosk-subject" value={subject} onChange={(e) => setSubject(e.target.value)}>
                      <option value="">Select a person…</option>
                      {subjects.map((s) => <option key={s.user_id} value={s.user_id}>{s.display_name}</option>)}
                    </Select>
                    <p className="text-xs text-muted">Only people authorised for this kiosk are listed. Matching and liveness run on this device.</p>
                  </Card>
                )} />
            )}
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader title="Kiosk status" />
                <div className="p-5">
                  <KeyValue items={[
                    ['Connection', status.online ? 'Connected to server' : 'Offline — decisions are made locally'],
                    ['Authorisation valid until', status.bundle.valid_until ? fmtDateTime(status.bundle.valid_until) : '—'],
                    ['Authorised people', String(status.bundle.subject_count ?? 0)],
                    ['Events waiting to sync', String(status.events.pending)],
                    ['Last sync', status.last_sync_at ? fmtRelative(status.last_sync_at) : 'Never'],
                    ['Inference', `${status.inference} — ${status.inference_location}`],
                    ['Key protection', status.key_protection],
                  ]} />
                  {!status.hardware_backed && (
                    <p className="mt-3 flex gap-2 text-xs text-muted"><ShieldAlert className="h-4 w-4 shrink-0" />Keys are protected by the operating system or a
                      passphrase, not by tamper-resistant hardware. Revocations reach this kiosk only when it reconnects.</p>
                  )}
                  {status.last_sync_error && <p className="mt-2 text-xs text-muted">Last sync: {status.last_sync_error}</p>}
                </div>
              </Card>
              <Card>
                <CardHeader title="Local audit log" subtitle="Hash-chained; uploaded to the server when online." />
                <ul className="max-h-80 divide-y divide-line overflow-y-auto" data-testid="kiosk-events">
                  {events.length === 0 && <li className="p-5 text-sm text-muted">No events yet.</li>}
                  {events.map((e) => (
                    <li key={e.event_id} className="flex items-center gap-3 px-5 py-2.5 text-sm">
                      <span className="tabular w-8 text-xs text-muted">#{e.seq}</span>
                      <span className="min-w-0 flex-1 truncate text-ink-2">{(e.subject_user_id && names[e.subject_user_id]) || humanize(e.type)}
                        {e.reason_code && <span className="block text-xs text-muted">{humanize(e.reason_code)}</span>}</span>
                      {e.outcome ? <OutcomeBadge outcome={e.outcome} /> : <Badge tone="neutral">{humanize(e.type)}</Badge>}
                      <Badge tone={e.synced ? 'good' : 'warn'}>{e.synced ? 'Synced' : 'Pending'}</Badge>
                    </li>
                  ))}
                </ul>
              </Card>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
