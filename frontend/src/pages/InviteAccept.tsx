import { useEffect, useState, type FormEvent } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Building2, ShieldCheck } from 'lucide-react';
import { api, apiError } from '../lib/api';
import type { Me } from '../lib/types';
import { useAuth } from '../store/auth';
import { Button } from '../components/ui/Button';
import { Callout, Card, LoadingBlock } from '../components/ui/primitives';
import { Field, Input } from '../components/ui/form';

interface InviteInfo { organization: string; role: string; email: string; account_exists: boolean; expires_at: string }

/** Public page for invitation links. Existing accounts must be signed in as the invited email. */
export default function InviteAccept() {
  const { token = '' } = useParams();
  const user = useAuth((s) => s.user);
  const setUser = useAuth((s) => s.setUser);
  const navigate = useNavigate();
  const [info, setInfo] = useState<InviteInfo | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.get<InviteInfo>(`/invitations/${encodeURIComponent(token)}`).then((r) => setInfo(r.data))
      .catch((e) => setLoadError(apiError(e, 'This invitation link is not valid').message));
  }, [token]);

  const accept = async (e?: FormEvent) => {
    e?.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await api.post<{ session?: { user: Me } }>(`/invitations/${encodeURIComponent(token)}/accept`,
        info?.account_exists ? {} : { full_name: name, password });
      const me = r.data.session?.user ?? (await api.get<Me>('/auth/me')).data;
      setUser(me);
      navigate('/', { replace: true });
    } catch (err) {
      setError(apiError(err).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <Card className="w-full max-w-md p-6">
        <div className="mb-5 flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-white"><ShieldCheck className="h-5 w-5" /></span>
          <span className="font-semibold text-ink">FaceVerification</span>
        </div>
        {loadError ? <Callout tone="bad" title="Invitation unavailable">{loadError}</Callout> : !info ? <LoadingBlock rows={3} /> : (
          <div className="space-y-4" data-testid="invite-page">
            <div>
              <h1 className="flex items-center gap-2 text-xl font-semibold text-ink"><Building2 className="h-5 w-5 text-muted" />Join {info.organization}</h1>
              <p className="mt-1 text-sm text-muted">Invitation for {info.email} as <strong>{info.role}</strong>.</p>
            </div>
            {error && <Callout tone="bad">{error}</Callout>}
            {info.account_exists ? (
              user ? (
                <Button className="w-full" size="lg" loading={busy} onClick={() => void accept()}>Accept invitation</Button>
              ) : (
                <Callout tone="info" title="Sign in first">
                  This email already has an account. <Link to="/login" state={{ from: `/invite/${token}` }} className="font-medium text-primary hover:underline">Sign in</Link>{' '}
                  as {info.email}, then open this link again.
                </Callout>
              )
            ) : (
              <form onSubmit={accept} className="space-y-4">
                <Field label="Your full name" htmlFor="inv-name"><Input id="inv-name" value={name} onChange={(e) => setName(e.target.value)} required /></Field>
                <Field label="Choose a password" htmlFor="inv-password" hint="At least 10 characters with three of: lowercase, uppercase, digit, symbol.">
                  <Input id="inv-password" type="password" autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
                </Field>
                <Button type="submit" className="w-full" size="lg" loading={busy} disabled={!name.trim() || !password}>Create account and join</Button>
              </form>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
