import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { CheckCircle2, ShieldCheck, Smartphone, XCircle } from 'lucide-react';
import { apiError } from '../lib/api';
import { humanize } from '../lib/format';
import type { LinkInfo, PhoneState, SessionView } from '../lib/types';
import { Button } from '../components/ui/Button';
import { Callout, Card, LoadingBlock } from '../components/ui/primitives';
import { Checkbox } from '../components/ui/form';
import { FaceCheck } from '../features/camera/FaceCheck';
import { headerTransport } from '../features/camera/transport';

const storeKey = (token: string) => `fv_pair_${token.slice(0, 16)}`;
const remote = axios.create({ baseURL: '/api/v1/remote', timeout: 20_000 });

interface Claimed { secret: string; code: string | null }

/** The phone side of a QR pairing or hosted verification link. No sign-in; a pairing secret (memory + tab storage) authorises it. */
export default function MobileVerify() {
  const { token = '' } = useParams();
  const [info, setInfo] = useState<LinkInfo | null>(null);
  const [claimed, setClaimed] = useState<Claimed | null>(() => {
    try { return JSON.parse(sessionStorage.getItem(storeKey(token)) ?? 'null') as Claimed | null; } catch { return null; }
  });
  const [state, setState] = useState<PhoneState | null>(null);
  const [agree, setAgree] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [offline, setOffline] = useState(false);
  const transport = useMemo(() => (claimed ? headerTransport('X-Pairing-Secret', claimed.secret) : null), [claimed]);

  useEffect(() => {
    if (claimed) return;
    remote.get<LinkInfo>(`/link/${encodeURIComponent(token)}`).then((r) => setInfo(r.data))
      .catch((e) => setError(apiError(e, 'This link is not valid').message));
  }, [token, claimed]);

  const poll = useCallback(async () => {
    if (!claimed) return;
    try {
      const r = await remote.get<PhoneState>('/state', { headers: { 'X-Pairing-Secret': claimed.secret } });
      setState(r.data);
      setOffline(false);
    } catch (e) {
      const info = apiError(e);
      if (info.network) setOffline(true);
      else if (info.status === 401) { sessionStorage.removeItem(storeKey(token)); setClaimed(null); setError('This pairing is no longer valid.'); }
    }
  }, [claimed, token]);

  useEffect(() => {
    void poll();
    if (!claimed) return;
    const t = window.setInterval(() => void poll(), 2000);
    return () => window.clearInterval(t);
  }, [claimed, poll]);

  const claim = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await remote.post<{ pairing_secret: string; confirmation_code: string | null; state: PhoneState }>(
        `/link/${encodeURIComponent(token)}/claim`, { accept: true });
      const c = { secret: r.data.pairing_secret, code: r.data.confirmation_code };
      sessionStorage.setItem(storeKey(token), JSON.stringify(c));
      setClaimed(c);
      setState(r.data.state);
    } catch (e) {
      setError(apiError(e, 'Could not open this verification').message);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    if (!claimed) return;
    await remote.post('/cancel', {}, { headers: { 'X-Pairing-Secret': claimed.secret } }).catch(() => undefined);
    void poll();
  };

  const start = useCallback(async (): Promise<SessionView> => {
    const r = await remote.post<SessionView>('/session', {}, { headers: { 'X-Pairing-Secret': claimed!.secret } });
    return r.data;
  }, [claimed]);

  const s = state?.status;
  return (
    <div className="min-h-full bg-page">
      <header className="flex items-center gap-2 border-b border-line bg-surface px-4 py-3">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-white"><ShieldCheck className="h-4 w-4" /></span>
        <div className="min-w-0 leading-tight">
          <p className="text-sm font-semibold text-ink">Identity check</p>
          <p className="truncate text-xs text-muted">{state?.organization ?? info?.organization ?? 'FaceVerification'}</p>
        </div>
        {offline && <span className="ml-auto rounded-full bg-warning/20 px-2 py-0.5 text-xs text-ink">Reconnecting…</span>}
      </header>
      <main className="mx-auto w-full max-w-3xl p-4">
        {error && !claimed && <Callout tone="bad" title="Cannot continue">{error}</Callout>}

        {!claimed && !error && (!info ? <LoadingBlock rows={3} /> : (
          <Card className="space-y-4 p-5" data-testid="mobile-intro">
            <Smartphone className="h-8 w-8 text-primary" />
            <h1 className="text-xl font-semibold text-ink">Verify your identity</h1>
            <p className="text-sm text-ink-2">
              {info.requested_by ? `${info.requested_by} at ` : ''}{info.organization} asks <strong>{info.subject}</strong> to complete a live face check
              with this phone's front camera. Images are processed on the server in memory and never stored.
            </p>
            <Checkbox id="agree" checked={agree} onChange={setAgree}>
              I am {info.subject} and I agree to this face check.
            </Checkbox>
            <Button className="w-full" size="lg" disabled={!agree} loading={busy} onClick={() => void claim()} data-testid="claim-pairing">Continue</Button>
          </Card>
        ))}

        {claimed && !state && <LoadingBlock rows={3} />}

        {claimed && s === 'AWAITING_CONFIRMATION' && (
          <Card className="space-y-4 p-5 text-center" data-testid="mobile-code">
            <p className="text-sm text-ink-2">Tell the person who showed you the QR code this number:</p>
            <p className="tabular text-5xl font-semibold tracking-[0.2em] text-ink" data-testid="confirmation-code">{claimed.code}</p>
            <p className="text-sm text-muted">Waiting for them to confirm…</p>
            <Button variant="outline" className="w-full" onClick={() => void cancel()}>Cancel</Button>
          </Card>
        )}

        {claimed && transport && (s === 'READY' || s === 'IN_PROGRESS') && (
          <div className="space-y-3" data-testid="mobile-capture">
            <Callout tone="info">Hold the phone at eye level in good light and follow the instructions. Attempts used: {state?.attempts_used ?? 0}.</Callout>
            <FaceCheck transport={transport} start={start} startLabel="Start camera" onTerminal={() => void poll()}
              onAgain={() => void poll()} />
            <Button variant="ghost" className="w-full" onClick={() => void cancel()}>Cancel verification</Button>
          </div>
        )}

        {claimed && s === 'COMPLETED' && (
          <Card className="space-y-3 p-6 text-center" data-testid="mobile-result">
            {state?.result_outcome === 'VERIFIED'
              ? <CheckCircle2 className="mx-auto h-12 w-12 text-good-text" />
              : <XCircle className="mx-auto h-12 w-12 text-critical" />}
            <h1 className="text-xl font-semibold text-ink">{state?.result_outcome === 'VERIFIED' ? 'Verified' : humanize(state?.result_outcome ?? '')}</h1>
            {state?.review_status && <p className="text-sm text-muted">This result will be reviewed by a person.</p>}
            <p className="text-sm text-muted">You can close this page.</p>
          </Card>
        )}

        {claimed && (s === 'EXPIRED' || s === 'CANCELLED') && (
          <Card className="space-y-2 p-6 text-center" data-testid="mobile-ended">
            <XCircle className="mx-auto h-10 w-10 text-muted" />
            <h1 className="text-lg font-semibold text-ink">{s === 'EXPIRED' ? 'This verification expired' : 'This verification was cancelled'}</h1>
            <p className="text-sm text-muted">Ask for a new QR code if you still need to verify.</p>
          </Card>
        )}
      </main>
    </div>
  );
}
