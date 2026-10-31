import { useMemo, useState, type FormEvent } from 'react';
import { Navigate, useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, Fingerprint, KeyRound, Lock, Mail, ScanFace, ShieldCheck } from 'lucide-react';
import { api, apiError } from '../lib/api';
import type { LoginResponse, Me, MfaMethod, SessionView } from '../lib/types';
import { getPasskey, passkeyErrorMessage, passkeysSupported, type Envelope } from '../lib/webauthn';
import { useAuth } from '../store/auth';
import { Button } from '../components/ui/Button';
import { Callout } from '../components/ui/primitives';
import { Field, Input } from '../components/ui/form';
import { Tabs } from '../components/ui/Tabs';
import { QrImage } from '../components/QrImage';
import { FaceCheck } from '../features/camera/FaceCheck';
import { headerTransport } from '../features/camera/transport';


type Step =
  | { kind: 'credentials' }
  | { kind: 'mfa'; token: string; methods: MfaMethod[] }
  | { kind: 'enrol'; token: string; setup?: { factor_id: string; secret: string; otpauth_uri: string } }
  | { kind: 'codes'; codes: string[]; user: Me }
  | { kind: 'face'; token: string; name: string };

const METHOD_LABEL: Record<MfaMethod, string> = {
  webauthn: 'Passkey', totp: 'Authenticator app', recovery_code: 'Recovery code', email_otp: 'Email code',
};

function Panel() {
  return (
    <div className="relative hidden overflow-hidden bg-[#0d366b] p-12 text-white lg:flex lg:flex-col lg:justify-between">
      <div className="flex items-center gap-2.5">
        <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-white/15"><ShieldCheck className="h-5 w-5" /></span>
        <span className="text-base font-semibold">FaceVerification</span>
      </div>
      <div className="max-w-md">
        <h2 className="text-3xl font-semibold leading-tight">Supervised biometric identity verification.</h2>
        <ul className="mt-8 space-y-4 text-sm text-white/85">
          <li className="flex gap-3"><Fingerprint className="h-5 w-5 shrink-0" />Passwordless sign-in with passkeys, plus a second factor where your organisation requires it.</li>
          <li className="flex gap-3"><ScanFace className="h-5 w-5 shrink-0" />Randomised liveness challenges validated on the server.</li>
          <li className="flex gap-3"><Lock className="h-5 w-5 shrink-0" />Encrypted templates only — face images are never stored.</li>
        </ul>
      </div>
      <p className="text-xs text-white/60">Authorised personnel only. Activity is logged.</p>
    </div>
  );
}

export default function LoginPage() {
  const status = useAuth((s) => s.status);
  const setUser = useAuth((s) => s.setUser);
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from ?? '/';
  const [step, setStep] = useState<Step>({ kind: 'credentials' });
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  const [method, setMethod] = useState<MfaMethod>('totp');
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const supported = useMemo(() => passkeysSupported(), []);

  if (status === 'authenticated' && step.kind !== 'codes') return <Navigate to={from} replace />;

  const done = (user: Me) => {
    setUser(user);
    navigate(from === '/login' ? '/' : from, { replace: true });
  };

  const handle = (data: LoginResponse) => {
    setCode('');
    setError(null);
    if (data.status === 'ok') {
      if (data.recovery_codes?.length) setStep({ kind: 'codes', codes: data.recovery_codes, user: data.user });
      else done(data.user);
    } else if (data.status === 'mfa_required') {
      if (data.enrol_required) setStep({ kind: 'enrol', token: data.pending_token });
      else {
        setMethod(data.methods.includes('webauthn') ? 'webauthn' : data.methods[0] ?? 'totp');
        setStep({ kind: 'mfa', token: data.pending_token, methods: data.methods });
      }
    } else {
      setStep({ kind: 'face', token: data.pending_token, name: data.user_name });
    }
  };

  const run = async (key: string, fn: () => Promise<void>, fallback: string) => {
    setBusy(key);
    setError(null);
    try {
      await fn();
    } catch (err) {
      const isBrowser = !(err as { response?: unknown }).response && (err as { name?: string }).name?.endsWith('Error');
      const e = apiError(err, fallback);
      setError(isBrowser ? passkeyErrorMessage(err) : e.status === 401 && key === 'password' ? 'Incorrect email or password.' : e.message);
    } finally {
      setBusy(null);
    }
  };

  const submitPassword = (e: FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password) { setError('Enter your email and password.'); return; }
    void run('password', async () => {
      handle((await api.post<LoginResponse>('/auth/login', { email: email.trim(), password })).data);
    }, 'Sign-in failed');
  };

  const passkeySignIn = () => run('passkey', async () => {
    const env = (await api.post<Envelope<never>>('/auth/passkey/options')).data;
    const credential = await getPasskey(env);
    handle((await api.post<LoginResponse>('/auth/passkey/verify', { challenge_id: env.challenge_id, credential })).data);
  }, 'Passkey sign-in failed');

  const verifyMfa = (e?: FormEvent) => {
    e?.preventDefault();
    if (step.kind !== 'mfa') return;
    void run('mfa', async () => {
      const body: Record<string, unknown> = { pending_token: step.token, method, code };
      if (method === 'webauthn') {
        const env = (await api.post<Envelope<never>>('/auth/mfa/webauthn/options', { pending_token: step.token })).data;
        body.challenge_id = env.challenge_id;
        body.credential = await getPasskey(env);
      }
      handle((await api.post<LoginResponse>('/auth/mfa/verify', body)).data);
    }, 'Verification failed');
  };

  const sendEmailCode = () => step.kind === 'mfa' && run('email', async () => {
    await api.post('/auth/mfa/email-otp', { pending_token: step.token });
    setInfo('We sent a 6-digit code to your email. It expires in 10 minutes.');
  }, 'Could not send the email');

  const startEnrol = () => step.kind === 'enrol' && run('enrol', async () => {
    const r = await api.post<{ factor_id: string; secret: string; otpauth_uri: string }>('/auth/mfa/enrol/totp', { pending_token: step.token });
    setStep({ ...step, setup: r.data });
  }, 'Could not start set-up');

  const confirmEnrol = (e: FormEvent) => {
    e.preventDefault();
    if (step.kind !== 'enrol' || !step.setup) return;
    void run('enrol', async () => {
      handle((await api.post<LoginResponse>('/auth/mfa/enrol/totp/confirm',
        { pending_token: step.token, factor_id: step.setup!.factor_id, code })).data);
    }, 'That code did not work');
  };

  const back = () => { setStep({ kind: 'credentials' }); setError(null); setInfo(null); setCode(''); };

  return (
    <div className="grid min-h-full lg:grid-cols-2">
      <Panel />
      <div className="flex items-center justify-center p-6">
        <div className={step.kind === 'face' ? 'w-full max-w-5xl' : 'w-full max-w-sm'}>
          {step.kind === 'credentials' && (
            <form onSubmit={submitPassword} className="space-y-5" noValidate aria-label="Sign in">
              <div>
                <h1 className="text-2xl font-semibold tracking-tight text-ink">Sign in</h1>
                <p className="mt-1 text-sm text-muted">Use your organisation account.</p>
              </div>
              {error && <Callout tone="bad">{error}</Callout>}
              {supported && (
                <>
                  <Button type="button" variant="outline" size="lg" className="w-full" loading={busy === 'passkey'}
                    icon={<Fingerprint className="h-4 w-4" />} onClick={() => void passkeySignIn()} data-testid="passkey-signin">
                    Sign in with a passkey
                  </Button>
                  <div className="flex items-center gap-3 text-xs text-muted"><span className="h-px flex-1 bg-line" />or<span className="h-px flex-1 bg-line" /></div>
                </>
              )}
              <Field label="Email" htmlFor="email">
                <Input id="email" type="email" autoComplete="username webauthn" value={email} onChange={(e) => setEmail(e.target.value)} required />
              </Field>
              <Field label="Password" htmlFor="password">
                <Input id="password" type="password" autoComplete="current-password" value={password}
                  onChange={(e) => setPassword(e.target.value)} required />
              </Field>
              <Button type="submit" className="w-full" size="lg" loading={busy === 'password'}>Sign in</Button>
              <details className="text-xs text-muted">
                <summary className="cursor-pointer select-none hover:text-ink-2">Can't use your passkey or authenticator?</summary>
                <p className="mt-2 leading-relaxed">Sign in with your password, then choose <strong>Recovery code</strong>.
                  If you have no recovery codes left, ask an administrator of your organisation to reset your sign-in factors.</p>
              </details>
            </form>
          )}

          {step.kind === 'mfa' && (
            <form onSubmit={verifyMfa} className="space-y-5" aria-label="Second factor">
              <button type="button" onClick={back} className="inline-flex items-center gap-1 text-xs text-muted hover:text-ink">
                <ArrowLeft className="h-3.5 w-3.5" /> Back
              </button>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight text-ink">Two-step verification</h1>
                <p className="mt-1 text-sm text-muted">Your password was accepted. Confirm with a second factor.</p>
              </div>
              {step.methods.length > 1 && (
                <Tabs label="Verification method" value={method} onChange={(m) => { setMethod(m); setCode(''); setInfo(null); }}
                  items={step.methods.map((m) => ({ value: m, label: METHOD_LABEL[m] }))} />
              )}
              {error && <Callout tone="bad">{error}</Callout>}
              {info && <Callout tone="info">{info}</Callout>}
              {method === 'webauthn' ? (
                <Button type="submit" size="lg" className="w-full" loading={busy === 'mfa'} icon={<KeyRound className="h-4 w-4" />}>Use passkey</Button>
              ) : (
                <>
                  {method === 'email_otp' && (
                    <Button type="button" variant="outline" className="w-full" loading={busy === 'email'} icon={<Mail className="h-4 w-4" />}
                      onClick={() => void sendEmailCode()}>Email me a code</Button>
                  )}
                  <Field label={method === 'recovery_code' ? 'Recovery code' : '6-digit code'} htmlFor="mfa-code"
                    hint={method === 'recovery_code' ? 'Each recovery code works once.' : undefined}>
                    <Input id="mfa-code" autoComplete="one-time-code" inputMode={method === 'recovery_code' ? 'text' : 'numeric'}
                      value={code} onChange={(e) => setCode(e.target.value)} autoFocus />
                  </Field>
                  <Button type="submit" size="lg" className="w-full" loading={busy === 'mfa'} disabled={!code.trim()}>Verify</Button>
                </>
              )}
            </form>
          )}

          {step.kind === 'enrol' && (
            <div className="space-y-5">
              <button type="button" onClick={back} className="inline-flex items-center gap-1 text-xs text-muted hover:text-ink">
                <ArrowLeft className="h-3.5 w-3.5" /> Back
              </button>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight text-ink">Set up two-step verification</h1>
                <p className="mt-1 text-sm text-muted">Your organisation requires a second factor for administrators. Add an authenticator app to continue.</p>
              </div>
              {error && <Callout tone="bad">{error}</Callout>}
              {!step.setup ? (
                <Button size="lg" className="w-full" loading={busy === 'enrol'} onClick={() => void startEnrol()}>Set up authenticator app</Button>
              ) : (
                <form onSubmit={confirmEnrol} className="space-y-4">
                  <div className="flex flex-col items-center gap-3 rounded-xl border border-line p-4">
                    <QrImage value={step.setup.otpauth_uri} size={180} label="Authenticator set-up QR code" />
                    <p className="text-center text-xs text-muted">Scan with your authenticator app, or enter this key:</p>
                    <code className="break-all rounded bg-surface-2 px-2 py-1 text-xs" data-testid="totp-secret">{step.setup.secret}</code>
                  </div>
                  <Field label="Code from the app" htmlFor="enrol-code">
                    <Input id="enrol-code" inputMode="numeric" autoComplete="one-time-code" value={code} onChange={(e) => setCode(e.target.value)} />
                  </Field>
                  <Button type="submit" size="lg" className="w-full" loading={busy === 'enrol'} disabled={!code.trim()}>Confirm and sign in</Button>
                </form>
              )}
            </div>
          )}

          {step.kind === 'codes' && (
            <div className="space-y-5" data-testid="recovery-codes">
              <div>
                <h1 className="text-2xl font-semibold tracking-tight text-ink">Save your recovery codes</h1>
                <p className="mt-1 text-sm text-muted">Each code signs you in once if you lose your authenticator. They are shown only now.</p>
              </div>
              <ul className="grid grid-cols-2 gap-2 rounded-xl border border-line bg-surface-2 p-4 font-mono text-sm">
                {step.codes.map((c) => <li key={c}>{c}</li>)}
              </ul>
              <Button size="lg" className="w-full" onClick={() => done(step.user)}>I have saved them — continue</Button>
            </div>
          )}

          {step.kind === 'face' && (
            <div className="space-y-5">
              <button type="button" onClick={back} className="inline-flex items-center gap-1 text-xs text-muted hover:text-ink">
                <ArrowLeft className="h-3.5 w-3.5" /> Cancel sign-in
              </button>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight text-ink">Live face check</h1>
                <p className="mt-1 text-sm text-muted">Passkey accepted for {step.name}. Your organisation also requires a live face check.
                  The face check alone never signs anyone in.</p>
              </div>
              {error && <Callout tone="bad">{error}</Callout>}
              <FaceCheck transport={headerTransport('X-Login-Token', step.token)} startLabel="Start face check"
                start={async () => (await api.post<SessionView>('/auth/login/face/session', { pending_token: step.token })).data}
                onTerminal={(s) => {
                  if (s.final_outcome === 'VERIFIED') {
                    void run('face', async () => {
                      handle((await api.post<LoginResponse>('/auth/login/face/complete', { pending_token: step.token })).data);
                    }, 'Face check failed');
                  } else {
                    setError(`The face check did not pass (${String(s.final_outcome).toLowerCase().replace(/_/g, ' ')}). You can try again.`);
                  }
                }} />
            </div>
          )}

          <p className="mt-8 text-center text-xs text-muted">Joining an organisation? Open the invitation link you received.</p>
        </div>
      </div>
    </div>
  );
}
