import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { CheckCircle2, Circle, Loader2, QrCode, RefreshCw, Smartphone, X } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { cn } from '../lib/cn';
import { fmtDuration, humanize, STEP_LABEL } from '../lib/format';
import { useEventStream } from '../lib/sse';
import type { Pairing, UserRow } from '../lib/types';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, PageHeader } from '../components/ui/primitives';
import { Field, Input } from '../components/ui/form';
import { UserPicker } from '../components/UserPicker';
import { QrImage } from '../components/QrImage';
import { OutcomeBadge } from '../components/OutcomeBadge';

interface Progress { step?: string | null; step_index?: number | null; steps_total?: number | null; face_count?: number | null;
  message?: string | null; terminal?: boolean }

function useCountdown(until: string | undefined) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);
  if (!until) return null;
  return Math.max(0, Math.round((Date.parse(until.endsWith('Z') ? until : `${until}Z`) - now) / 1000));
}

const OPEN = ['WAITING', 'AWAITING_CONFIRMATION', 'READY', 'IN_PROGRESS'];

export default function QrVerify() {
  const [subject, setSubject] = useState<UserRow | null>(null);
  const [pairing, setPairing] = useState<Pairing | null>(null);
  const [qrUrl, setQrUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress>({});
  const [code, setCode] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const remaining = useCountdown(pairing?.status === 'WAITING' || pairing?.status === 'AWAITING_CONFIRMATION' ? pairing.expires_at : undefined);

  const reload = useCallback(async (id: string) => {
    try {
      setPairing((await api.get<Pairing>(`/pairings/${id}`)).data);
    } catch { /* transient; the stream or next poll will catch up */ }
  }, []);

  const { state: stream } = useEventStream(['pairing'], (m) => {
    const d = m.data as Progress & { pairing_id?: string };
    if (!pairing || d.pairing_id !== pairing.id) return;
    if (m.event === 'pairing_progress') setProgress(d);
    else void reload(pairing.id);
  }, !!pairing && OPEN.includes(pairing.status));

  // Polling fallback (reconnect safety); the stream normally delivers changes immediately.
  useEffect(() => {
    if (!pairing || !OPEN.includes(pairing.status)) return;
    const t = window.setInterval(() => void reload(pairing.id), stream === 'live' ? 10_000 : 3000);
    return () => window.clearInterval(t);
  }, [pairing, stream, reload]);

  useEffect(() => {
    if (remaining === 0 && pairing && (pairing.status === 'WAITING' || pairing.status === 'AWAITING_CONFIRMATION')) void reload(pairing.id);
  }, [remaining, pairing, reload]);

  const act = async (key: string, fn: () => Promise<void>) => {
    setBusy(key);
    setError(null);
    try { await fn(); } catch (err) { setError(apiError(err).message); } finally { setBusy(null); }
  };

  const create = () => act('create', async () => {
    const r = await api.post<Pairing>('/pairings', { target_user_id: subject!.id });
    setPairing(r.data);
    setQrUrl(r.data.qr_url ?? null);
    setProgress({});
    setCode('');
  });
  const regenerate = () => pairing && act('regen', async () => {
    const r = await api.post<Pairing>(`/pairings/${pairing.id}/regenerate`);
    setPairing(r.data);
    setQrUrl(r.data.qr_url ?? null);
  });
  const confirm = (e: FormEvent) => {
    e.preventDefault();
    if (pairing) void act('confirm', async () => {
      setPairing((await api.post<Pairing>(`/pairings/${pairing.id}/confirm`, { code: code.trim() })).data);
      setCode('');
    });
  };
  const cancel = () => pairing && act('cancel', async () => {
    setPairing((await api.post<Pairing>(`/pairings/${pairing.id}/cancel`)).data);
  });
  const reset = () => { setPairing(null); setQrUrl(null); setProgress({}); setError(null); };

  const st = pairing?.status;
  const stages: { label: string; done: boolean; active: boolean }[] = pairing ? [
    { label: 'QR code displayed', done: st !== 'WAITING', active: st === 'WAITING' },
    { label: pairing.device ? `Phone connected (${pairing.device})` : 'Phone connected', done: !!pairing.claimed_at, active: false },
    { label: 'Pairing confirmed by you', done: !!pairing.confirmed_at, active: st === 'AWAITING_CONFIRMATION' },
    { label: 'Camera ready', done: pairing.sessions_started > 0, active: st === 'READY' },
    { label: 'Face detected', done: (progress.face_count ?? 0) > 0 || st === 'COMPLETED', active: st === 'IN_PROGRESS' && !progress.face_count },
    { label: progress.step ? `Liveness: ${STEP_LABEL[progress.step] ?? humanize(progress.step)} (${(progress.step_index ?? 0) + 1}/${progress.steps_total ?? '?'})` : 'Liveness challenge',
      done: st === 'COMPLETED', active: st === 'IN_PROGRESS' && (progress.face_count ?? 0) > 0 },
    { label: 'Decision', done: st === 'COMPLETED', active: false },
  ] : [];

  return (
    <>
      <PageHeader title="QR verification" description="Verify someone on their own phone. The QR holds only a one-time reference; you confirm the pairing with the code shown on the phone." />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card className="p-6">
          {!pairing ? (
            <div className="flex flex-col items-center gap-4 py-10 text-center">
              <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-soft text-primary"><QrCode className="h-7 w-7" /></span>
              <p className="max-w-md text-sm text-ink-2">Select the person, then show them the QR code. They scan it with their phone camera, confirm, and complete the live check there.</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-5" data-testid="pairing-panel">
              {(st === 'WAITING') && qrUrl && (
                <>
                  <QrImage value={qrUrl} size={260} label="Verification QR code" />
                  <p className="text-sm text-ink-2">Scan with the phone camera · expires in <span className="tabular font-semibold">{fmtDuration(remaining ?? 0)}</span></p>
                  <details className="w-full max-w-md text-xs text-muted"><summary className="cursor-pointer">Can't scan? Open the link on the phone</summary>
                    <p className="mt-1 break-all font-mono" data-testid="pairing-link">{qrUrl}</p></details>
                </>
              )}
              {st === 'AWAITING_CONFIRMATION' && (
                <form onSubmit={confirm} className="w-full max-w-sm space-y-3">
                  <Callout tone="info" title="Phone connected">Ask the person to read the 6-digit code shown on their phone and type it here.
                    Do not confirm a code you did not see on their device.</Callout>
                  <Field label="Code shown on the phone" htmlFor="pair-code">
                    <Input id="pair-code" inputMode="numeric" maxLength={6} value={code} autoFocus className="text-center text-2xl tracking-[0.4em]"
                      onChange={(e) => setCode(e.target.value.replace(/\D/g, ''))} />
                  </Field>
                  <Button type="submit" className="w-full" loading={busy === 'confirm'} disabled={code.length !== 6}>Confirm pairing</Button>
                </form>
              )}
              {(st === 'READY' || st === 'IN_PROGRESS') && (
                <div className="flex flex-col items-center gap-3 py-6 text-center">
                  <Smartphone className="h-10 w-10 text-primary" />
                  <p className="text-lg font-semibold text-ink">{st === 'READY' ? 'Waiting for the phone camera…' : 'Verification in progress on the phone'}</p>
                  {progress.message && <p className="text-sm text-ink-2">{progress.message}</p>}
                </div>
              )}
              {st === 'COMPLETED' && (
                <div className="flex flex-col items-center gap-3 py-6 text-center" data-testid="pairing-result">
                  <OutcomeBadge outcome={pairing.result_outcome} />
                  <p className="text-lg font-semibold text-ink">{pairing.target_name}: {humanize(pairing.result_outcome ?? '')}</p>
                  {pairing.session?.review_status && <Badge tone="warn">Manual review: {humanize(pairing.session.review_status)}</Badge>}
                  {pairing.session_id && <Link to={`/history/${pairing.session_id}`} className="text-sm font-medium text-primary hover:underline">View session details</Link>}
                </div>
              )}
              {st === 'EXPIRED' && (
                <div className="flex flex-col items-center gap-3 py-6 text-center" data-testid="pairing-expired">
                  <p className="text-lg font-semibold text-ink">This QR code expired</p>
                  <p className="text-sm text-muted">It was not used in time. Generate a new one to continue.</p>
                  <Button icon={<RefreshCw className="h-4 w-4" />} loading={busy === 'regen'} onClick={() => void regenerate()}>Generate new QR</Button>
                </div>
              )}
              {st === 'CANCELLED' && (
                <div className="flex flex-col items-center gap-3 py-6 text-center">
                  <p className="text-lg font-semibold text-ink">Pairing cancelled</p>
                  <p className="text-sm text-muted">{humanize(pairing.cancelled_reason ?? '')}</p>
                </div>
              )}
            </div>
          )}
          {error && <Callout tone="bad" className="mt-4">{error}</Callout>}
        </Card>

        <div className="space-y-4">
          <Card className="p-5">
            <UserPicker value={subject} onChange={setSubject} enrolledOnly disabled={!!pairing && OPEN.includes(st ?? '')} label="Person to verify" />
            <div className="mt-4 flex gap-2">
              {!pairing || !OPEN.includes(st ?? '') ? (
                <Button className="w-full" disabled={!subject} loading={busy === 'create'} onClick={() => { reset(); void create(); }}
                  icon={<QrCode className="h-4 w-4" />} data-testid="create-pairing">Show QR code</Button>
              ) : (
                <Button className="w-full" variant="outline" icon={<X className="h-4 w-4" />} loading={busy === 'cancel'} onClick={() => void cancel()}
                  data-testid="cancel-pairing">Cancel</Button>
              )}
            </div>
          </Card>
          {pairing && (
            <Card>
              <CardHeader title="Progress" subtitle={!OPEN.includes(pairing.status) ? 'Finished'
                : stream === 'live' ? 'Live updates' : 'Reconnecting — refreshing periodically'} />
              <ol className="space-y-2 p-5" data-testid="pairing-stages">
                {stages.map((s) => (
                  <li key={s.label} className="flex items-center gap-2.5 text-sm">
                    {s.done ? <CheckCircle2 className="h-4 w-4 text-good-text" /> : s.active ? <Loader2 className="h-4 w-4 animate-spin text-primary" /> : <Circle className="h-4 w-4 text-line" />}
                    <span className={cn(s.done ? 'text-ink' : s.active ? 'font-medium text-ink' : 'text-muted')}>{s.label}</span>
                  </li>
                ))}
              </ol>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
