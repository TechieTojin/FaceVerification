import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Fingerprint, KeyRound, LifeBuoy, Pencil, Plus, ShieldCheck, Smartphone, Trash2 } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtRelative } from '../lib/format';
import type { SecurityOverview } from '../lib/types';
import { createPasskey, passkeyErrorMessage, passkeysSupported, type Envelope } from '../lib/webauthn';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Input } from '../components/ui/form';
import { ConfirmDialog, Modal } from '../components/ui/Modal';
import { QrImage } from '../components/QrImage';

function CodesModal({ codes, onClose }: { codes: string[] | null; onClose: () => void }) {
  return (
    <Modal open={!!codes} onClose={onClose} title="Your recovery codes"
      description="Each code signs you in once if you lose your passkey or authenticator. They are shown only now — store them safely."
      footer={<Button onClick={onClose}>I have saved them</Button>}>
      <ul className="grid grid-cols-2 gap-2 rounded-xl border border-line bg-surface-2 p-4 font-mono text-sm" data-testid="recovery-code-list">
        {codes?.map((c) => <li key={c}>{c}</li>)}
      </ul>
    </Modal>
  );
}

export default function AccountSecurity() {
  const qc = useQueryClient();
  const q = useApi<SecurityOverview>('/auth/security');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [passkeyName, setPasskeyName] = useState('');
  const [adding, setAdding] = useState(false);
  const [rename, setRename] = useState<{ id: string; name: string } | null>(null);
  const [removing, setRemoving] = useState<{ kind: 'passkey' | 'factor'; id: string; name: string } | null>(null);
  const [totp, setTotp] = useState<{ factor_id: string; secret: string; otpauth_uri: string } | null>(null);
  const [code, setCode] = useState('');
  const [codes, setCodes] = useState<string[] | null>(null);
  const refresh = () => void qc.invalidateQueries({ queryKey: ['/auth/security'] });

  const act = async (key: string, fn: () => Promise<void>) => {
    setBusy(key);
    setError(null);
    try {
      await fn();
    } catch (err) {
      const browser = !(err as { response?: unknown }).response && (err as { name?: string }).name?.endsWith('Error');
      setError(browser ? passkeyErrorMessage(err) : apiError(err).message);
    } finally {
      setBusy(null);
    }
  };

  const addPasskey = (e: FormEvent) => {
    e.preventDefault();
    void act('passkey', async () => {
      const env = (await api.post<Envelope<never>>('/auth/passkeys/register/options')).data;
      const credential = await createPasskey(env);
      await api.post('/auth/passkeys/register/verify', { challenge_id: env.challenge_id, credential, name: passkeyName || 'Passkey' });
      toast.success('Passkey added', 'You can now sign in without a password.');
      setAdding(false);
      setPasskeyName('');
      refresh();
    });
  };

  const startTotp = () => act('totp', async () => {
    setTotp((await api.post<{ factor_id: string; secret: string; otpauth_uri: string }>('/auth/totp/setup')).data);
    setCode('');
  });

  const confirmTotp = (e: FormEvent) => {
    e.preventDefault();
    if (!totp) return;
    void act('totp', async () => {
      const r = await api.post<{ recovery_codes: string[] }>('/auth/totp/confirm', { factor_id: totp.factor_id, code });
      setTotp(null);
      toast.success('Authenticator app enabled');
      if (r.data.recovery_codes.length) setCodes(r.data.recovery_codes);
      refresh();
    });
  };

  const doRemove = () => removing && act('remove', async () => {
    if (removing.kind === 'passkey') await api.delete(`/auth/passkeys/${removing.id}`);
    else await api.delete(`/auth/factors/${removing.id}`);
    toast.success(`${removing.name} removed`);
    setRemoving(null);
    refresh();
  });

  const regenerate = () => act('codes', async () => {
    setCodes((await api.post<{ recovery_codes: string[] }>('/auth/recovery-codes')).data.recovery_codes);
    refresh();
  });

  const doRename = (e: FormEvent) => {
    e.preventDefault();
    if (!rename) return;
    void act('rename', async () => {
      await api.patch(`/auth/passkeys/${rename.id}`, { name: rename.name });
      setRename(null);
      refresh();
    });
  };

  const d = q.data;
  const totpFactor = d?.factors.find((f) => f.kind === 'TOTP');

  return (
    <>
      <PageHeader title="Account security" description="Passkeys, two-step verification and recovery for your account (applies in every organisation you belong to)."
        actions={<Link to="/profile" className="text-sm font-medium text-primary hover:underline">Password & signed-in devices</Link>} />
      {error && <Callout tone="bad" className="mb-4">{error}</Callout>}
      {q.isLoading || !d ? <LoadingBlock rows={6} /> : (
        <div className="grid gap-6 xl:grid-cols-2">
          <Card className="xl:col-span-2 p-5">
            <div className="flex flex-wrap items-center gap-3">
              <ShieldCheck className={d.second_factor_enabled ? 'h-6 w-6 text-good-text' : 'h-6 w-6 text-muted'} />
              <div className="min-w-0 flex-1">
                <p className="font-medium text-ink">{d.second_factor_enabled ? 'Two-step verification is on' : 'Two-step verification is off'}</p>
                <p className="text-sm text-muted">{d.second_factor_enabled
                  ? 'Password sign-ins also need your passkey, authenticator app or a recovery code.'
                  : 'Add a passkey or authenticator app to protect your account beyond the password.'}</p>
              </div>
              {d.mfa_required_by_policy && <Badge tone="warn">Required by your organisation</Badge>}
            </div>
          </Card>

          <Card>
            <CardHeader title="Passkeys" subtitle="Sign in with your device's screen lock or a security key. Your fingerprint or face never leaves the device."
              actions={passkeysSupported() && <Button size="sm" icon={<Plus className="h-4 w-4" />} onClick={() => setAdding(true)} data-testid="add-passkey">Add passkey</Button>} />
            {!passkeysSupported() && <Callout tone="warn" className="m-4">This browser does not support passkeys.</Callout>}
            {d.passkeys.length === 0 ? <EmptyState icon={<Fingerprint className="h-5 w-5" />} title="No passkeys yet" /> : (
              <ul className="divide-y divide-line">
                {d.passkeys.map((p) => (
                  <li key={p.id} className="flex items-center gap-3 px-5 py-3" data-testid="passkey-row">
                    <KeyRound className="h-4 w-4 shrink-0 text-muted" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-ink">{p.name}</p>
                      <p className="text-xs text-muted">Added {fmtDateTime(p.created_at)} · {p.last_used_at ? `last used ${fmtRelative(p.last_used_at)}` : 'never used'}
                        {p.backed_up ? ' · synced' : ''}</p>
                    </div>
                    <Button size="sm" variant="ghost" aria-label={`Rename ${p.name}`} onClick={() => setRename({ id: p.id, name: p.name })}><Pencil className="h-4 w-4" /></Button>
                    <Button size="sm" variant="ghost" aria-label={`Remove ${p.name}`} onClick={() => setRemoving({ kind: 'passkey', id: p.id, name: p.name })}><Trash2 className="h-4 w-4" /></Button>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card>
            <CardHeader title="Authenticator app" subtitle="Time-based one-time codes (TOTP). Each code is accepted once." />
            <div className="p-5">
              {totpFactor ? (
                <div className="flex items-center gap-3">
                  <Smartphone className="h-5 w-5 text-good-text" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-ink">{totpFactor.label}</p>
                    <p className="text-xs text-muted">Enabled {fmtDateTime(totpFactor.confirmed_at)}{totpFactor.last_used_at ? ` · last used ${fmtRelative(totpFactor.last_used_at)}` : ''}</p>
                  </div>
                  <Button size="sm" variant="outline" onClick={() => setRemoving({ kind: 'factor', id: totpFactor.id, name: totpFactor.label })}>Remove</Button>
                </div>
              ) : totp ? (
                <form onSubmit={confirmTotp} className="space-y-4">
                  <div className="flex flex-col items-center gap-3 rounded-xl border border-line p-4">
                    <QrImage value={totp.otpauth_uri} size={176} label="Authenticator set-up QR code" />
                    <code className="break-all rounded bg-surface-2 px-2 py-1 text-xs" data-testid="totp-secret">{totp.secret}</code>
                  </div>
                  <Field label="Enter the 6-digit code from the app" htmlFor="totp-code">
                    <Input id="totp-code" inputMode="numeric" autoComplete="one-time-code" value={code} onChange={(e) => setCode(e.target.value)} />
                  </Field>
                  <div className="flex justify-end gap-2">
                    <Button variant="ghost" onClick={() => setTotp(null)}>Cancel</Button>
                    <Button type="submit" loading={busy === 'totp'} disabled={!code.trim()}>Enable</Button>
                  </div>
                </form>
              ) : (
                <Button icon={<Smartphone className="h-4 w-4" />} loading={busy === 'totp'} onClick={() => void startTotp()} data-testid="setup-totp">
                  Set up authenticator app
                </Button>
              )}
            </div>
          </Card>

          <Card>
            <CardHeader title="Recovery codes" subtitle="One-time codes for when you cannot use your passkey or app." />
            <div className="flex flex-wrap items-center gap-3 p-5">
              <LifeBuoy className="h-5 w-5 text-muted" />
              <p className="min-w-0 flex-1 text-sm text-ink-2"><span className="tabular font-semibold text-ink">{d.recovery_codes_remaining}</span> unused codes</p>
              <Button size="sm" variant="outline" loading={busy === 'codes'} onClick={() => void regenerate()}>Generate new codes</Button>
            </div>
          </Card>

          <Card>
            <CardHeader title="Email codes" subtitle="One-time codes sent to your email address during sign-in." />
            <div className="p-5 text-sm text-ink-2">
              {d.email_otp_available
                ? 'Available as a sign-in option once two-step verification is on.'
                : <Callout tone="neutral">Not available: no mail server is configured for this deployment.</Callout>}
            </div>
          </Card>
        </div>
      )}

      <Modal open={adding} onClose={() => setAdding(false)} title="Add a passkey"
        description="Your browser will ask you to confirm with your screen lock, fingerprint, face or a security key.">
        <form onSubmit={addPasskey} className="space-y-4">
          <Field label="Name" htmlFor="passkey-name" hint="Helps you recognise it later, e.g. 'Work laptop'.">
            <Input id="passkey-name" value={passkeyName} maxLength={100} onChange={(e) => setPasskeyName(e.target.value)} />
          </Field>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setAdding(false)}>Cancel</Button>
            <Button type="submit" loading={busy === 'passkey'} icon={<Fingerprint className="h-4 w-4" />} data-testid="create-passkey">Create passkey</Button>
          </div>
        </form>
      </Modal>
      <Modal open={!!rename} onClose={() => setRename(null)} title="Rename passkey">
        <form onSubmit={doRename} className="space-y-4">
          <Field label="Name" htmlFor="rename-passkey">
            <Input id="rename-passkey" value={rename?.name ?? ''} maxLength={100}
              onChange={(e) => setRename((r) => (r ? { ...r, name: e.target.value } : r))} />
          </Field>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setRename(null)}>Cancel</Button>
            <Button type="submit" loading={busy === 'rename'} disabled={!rename?.name.trim()}>Save</Button>
          </div>
        </form>
      </Modal>
      <ConfirmDialog open={!!removing} onClose={() => setRemoving(null)} onConfirm={() => void doRemove()} danger
        title={`Remove ${removing?.name ?? ''}?`} confirmLabel="Remove"
        body="You will be asked to confirm your identity. If this is your last factor, password sign-in will no longer require a second step." />
      <CodesModal codes={codes} onClose={() => setCodes(null)} />
    </>
  );
}
