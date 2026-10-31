import { useEffect, useMemo, useState, type FormEvent } from 'react';
import { KeyRound, ShieldCheck } from 'lucide-react';
import { api, apiError } from '../lib/api';
import type { SecurityOverview, SessionView } from '../lib/types';
import { getPasskey, passkeyErrorMessage, type Envelope } from '../lib/webauthn';
import { useAuth } from '../store/auth';
import { useStepUp } from '../store/stepUp';
import { FaceCheck } from '../features/camera/FaceCheck';
import { deferredHeaderTransport } from '../features/camera/transport';
import { Button } from './ui/Button';
import { Field, Input } from './ui/form';
import { Modal } from './ui/Modal';
import { Callout, LoadingBlock } from './ui/primitives';
import { Tabs } from './ui/Tabs';

type Method = 'webauthn' | 'totp' | 'recovery_code' | 'password';

/** Re-authentication for sensitive actions, opened automatically by the API client. */
export function StepUpDialog() {
  const { open, level, finish } = useStepUp();
  const setUser = useAuth((s) => s.setUser);
  const [overview, setOverview] = useState<SecurityOverview | null>(null);
  const [method, setMethod] = useState<Method>('password');
  const [secret, setSecret] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [faceToken, setFaceToken] = useState<string | null>(null);
  const [facePassed, setFacePassed] = useState(false);

  useEffect(() => {
    if (!open) return;
    setSecret(''); setError(null); setFaceToken(null); setFacePassed(false);
    api.get<SecurityOverview>('/auth/security').then((r) => {
      setOverview(r.data);
      setMethod(r.data.passkeys.length ? 'webauthn' : r.data.factors.some((f) => f.kind === 'TOTP') ? 'totp' : 'password');
    }).catch((e) => setError(apiError(e).message));
  }, [open]);

  const methods = useMemo(() => {
    if (!overview) return [] as { value: Method; label: string }[];
    const strong = overview.passkeys.length > 0 || overview.factors.some((f) => f.kind === 'TOTP');
    const m: { value: Method; label: string }[] = [];
    if (overview.passkeys.length) m.push({ value: 'webauthn', label: 'Passkey' });
    if (overview.factors.some((f) => f.kind === 'TOTP')) m.push({ value: 'totp', label: 'Authenticator' });
    if (overview.recovery_codes_remaining > 0) m.push({ value: 'recovery_code', label: 'Recovery code' });
    if (!strong) m.push({ value: 'password', label: 'Password' });
    return m;
  }, [overview]);

  const needFace = level === 'face_and_factor';
  const face = useMemo(() => deferredHeaderTransport('X-Login-Token'), []);

  const submit = async (e?: FormEvent) => {
    e?.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { method };
      if (method === 'password') body.password = secret;
      if (method === 'totp' || method === 'recovery_code') body.code = secret;
      if (method === 'webauthn') {
        const env = (await api.post<Envelope<never>>('/auth/step-up/webauthn/options')).data;
        body.challenge_id = env.challenge_id;
        body.credential = await getPasskey(env);
      }
      if (needFace) body.face_token = faceToken;
      const r = await api.post<{ user: Parameters<typeof setUser>[0] }>('/auth/step-up', body);
      setUser(r.data.user);
      finish(true);
    } catch (err) {
      setError((err as { name?: string })?.name?.endsWith('Error') && !(err as { response?: unknown }).response
        ? passkeyErrorMessage(err) : apiError(err, 'Confirmation failed').message);
    } finally {
      setBusy(false);
    }
  };

  const startFace = async (): Promise<SessionView> => {
    const r = await api.post<{ face_token: string; session: SessionView }>('/auth/step-up/face/session');
    face.set(r.data.face_token);
    setFaceToken(r.data.face_token);
    return r.data.session;
  };

  return (
    <Modal open={open} onClose={() => finish(false)} size={needFace && !facePassed ? 'xl' : 'md'}
      title="Confirm it's you"
      description={needFace
        ? 'This high-risk action needs a live face check and a second factor.'
        : 'This action is sensitive. Confirm your identity to continue.'}>
      {!overview ? <LoadingBlock rows={2} /> : (
        <div className="space-y-4" data-testid="step-up-dialog">
          {needFace && !facePassed && (
            <>
              <Callout tone="info" title="Step 1 of 2 — live face check">
                Look at the camera and follow the instructions. Your template in this organisation is used.
              </Callout>
              <FaceCheck startLabel="Start face check" transport={face.transport}
                start={startFace} onTerminal={(s) => {
                  if (s.final_outcome === 'VERIFIED') setFacePassed(true);
                  else setError(`Face check did not pass (${s.final_outcome?.toLowerCase().replace(/_/g, ' ')}).`);
                }} />
            </>
          )}
          {(!needFace || facePassed) && (
            <form onSubmit={submit} className="space-y-4">
              {needFace && <Callout tone="good" title="Face check passed">Now confirm a second factor.</Callout>}
              {methods.length > 1 && <Tabs label="Confirmation method" value={method} onChange={(v) => { setMethod(v); setSecret(''); }}
                items={methods} />}
              {method === 'webauthn' && (
                <p className="text-sm text-ink-2">Use the passkey on this device or your security key.</p>
              )}
              {method === 'password' && (
                <Field label="Password" htmlFor="stepup-password">
                  <Input id="stepup-password" type="password" autoComplete="current-password" value={secret}
                    onChange={(e) => setSecret(e.target.value)} autoFocus />
                </Field>
              )}
              {(method === 'totp' || method === 'recovery_code') && (
                <Field label={method === 'totp' ? '6-digit code from your authenticator app' : 'Recovery code'} htmlFor="stepup-code">
                  <Input id="stepup-code" inputMode={method === 'totp' ? 'numeric' : 'text'} autoComplete="one-time-code"
                    value={secret} onChange={(e) => setSecret(e.target.value)} autoFocus />
                </Field>
              )}
              {error && <Callout tone="bad">{error}</Callout>}
              <div className="flex justify-end gap-2">
                <Button type="button" variant="ghost" onClick={() => finish(false)}>Cancel</Button>
                <Button type="submit" loading={busy} icon={method === 'webauthn' ? <KeyRound className="h-4 w-4" /> : <ShieldCheck className="h-4 w-4" />}
                  disabled={(method !== 'webauthn' && !secret) || (needFace && !faceToken)} data-testid="step-up-confirm">
                  {method === 'webauthn' ? 'Use passkey' : 'Confirm'}
                </Button>
              </div>
            </form>
          )}
          {needFace && !facePassed && error && <Callout tone="bad">{error}</Callout>}
        </div>
      )}
    </Modal>
  );
}
