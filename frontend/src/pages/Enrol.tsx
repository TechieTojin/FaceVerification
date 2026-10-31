import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Check, ShieldCheck } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { REASON_TEXT, fmtDateTime, humanize } from '../lib/format';
import type { EnrolmentDetail, SessionView, UserRow } from '../lib/types';
import { cn } from '../lib/cn';
import { useAuth } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Checkbox, Field, Input } from '../components/ui/form';
import { UserPicker } from '../components/UserPicker';
import { LiveCapturePanel } from '../features/camera/LiveCapturePanel';
import { useLiveCapture } from '../features/camera/useLiveCapture';

const STEPS = ['Subject', 'Consent', 'Capture', 'Result'] as const;

function Stepper({ current }: { current: number }) {
  return (
    <ol className="mb-6 flex flex-wrap items-center gap-2" aria-label="Enrolment steps">
      {STEPS.map((s, i) => (
        <li key={s} className="flex items-center gap-2" aria-current={i === current ? 'step' : undefined}>
          <span className={cn('flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold',
            i < current ? 'bg-good/15 text-good-text' : i === current ? 'bg-primary text-white' : 'bg-surface-2 text-muted')}>
            {i < current ? <Check className="h-4 w-4" /> : i + 1}
          </span>
          <span className={cn('text-sm', i === current ? 'font-medium text-ink' : 'text-muted')}>{s}</span>
          {i < STEPS.length - 1 && <span className="mx-1 hidden h-px w-8 bg-line sm:block" aria-hidden />}
        </li>
      ))}
    </ol>
  );
}

export default function Enrol() {
  const { userId } = useParams();
  const navigate = useNavigate();
  const me = useAuth((s) => s.user);
  const qc = useQueryClient();
  const [subject, setSubject] = useState<UserRow | null>(null);
  const [step, setStep] = useState(0);
  const preset = useApi<EnrolmentDetail>(userId ? `/enrolments/${userId}` : null, undefined, { retry: false });
  useEffect(() => {
    const p = preset.data;
    if (p && !subject) {
      setSubject({ id: p.user_id, email: p.user_email, full_name: p.user_name, role: p.user_role, is_active: true,
        enrolment_status: p.status, enrolment_usable: p.usable, reenrolment_requested: p.reenrolment_requested });
    }
  }, [preset.data, subject]);

  const status = useApi<EnrolmentDetail>(subject ? `/enrolments/${subject.id}` : null);
  const consentText = useApi<{ version: string; text: string }>('/enrolments/consent-text');

  const start = useCallback(async () => {
    if (!subject) throw new Error('No subject');
    return (await api.post<SessionView>(`/enrolments/${subject.id}/sessions`)).data;
  }, [subject]);
  const live = useLiveCapture(start, () => {
    setStep(3);
    void qc.invalidateQueries({ queryKey: [`/enrolments/${subject?.id}`] });
  });

  const privileged = subject?.role === 'ADMIN' && me?.role !== 'ADMIN';
  const st = status.data;
  const replacing = !!st?.usable && !st.reenrolment_requested;

  return (
    <>
      <PageHeader title="Enrolment" description="Create or replace a person's encrypted biometric template with a liveness-checked, multi-sample capture." />
      <Stepper current={step} />

      {step === 0 && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card className="p-5">
            <UserPicker value={subject} onChange={setSubject} label="Person to enrol" />
          </Card>
          <Card>
            <CardHeader title="Authorisation check" subtitle="Rules are enforced by the server; this is a preview." />
            <div className="space-y-4 px-5 py-4">
              {!subject ? <p className="text-sm text-muted">Select a person to see their enrolment status.</p> : status.isLoading ? <LoadingBlock rows={3} /> : st && (
                <>
                  <KeyValue items={[
                    ['Name', subject.full_name], ['Role', subject.role],
                    ['Enrolment', <Badge key="s" tone={st.usable ? 'good' : 'neutral'}>{humanize(st.status)}</Badge>],
                    ['Consent', st.consent_given ? `Given ${fmtDateTime(st.consent_timestamp)}` : 'Not given'],
                  ]} />
                  {privileged && <Callout tone="bad" title="Not permitted">Only administrators may enrol administrator accounts.</Callout>}
                  {!subject.is_active && <Callout tone="bad">This account is disabled.</Callout>}
                  {replacing && <Callout tone="warn" title="Replacement needs approval">
                    {subject.full_name} already has an active template. A new capture creates a replacement request that an administrator must approve; the current template stays in use until then.
                  </Callout>}
                  {st.status === 'REENROLMENT_REQUIRED' && <Callout tone="info">The existing template predates v3 or was flagged for re-enrolment.</Callout>}
                  {me?.id === subject.id && me.role !== 'ADMIN' && <Callout tone="warn">Self-enrolment by an operator requires administrator approval.</Callout>}
                </>
              )}
              <Button disabled={!subject || privileged || !subject.is_active} onClick={() => setStep(1)}>Continue</Button>
            </div>
          </Card>
        </div>
      )}

      {step === 1 && subject && (
        <ConsentStep subject={subject} text={consentText.data?.text ?? ''} version={consentText.data?.version ?? ''}
          alreadyGiven={!!st?.consent_given} onBack={() => setStep(0)}
          onDone={() => { void status.refetch(); setStep(2); }} />
      )}

      {step === 2 && subject && (
        <LiveCapturePanel live={live} startLabel="Start capture" canStart
          side={(
            <Card className="p-5 text-sm text-ink-2">
              <p className="font-medium text-ink">{subject.full_name}</p>
              <p className="mt-1 text-xs text-muted">Follow the on-screen actions, then hold still while several samples are captured. Only an encrypted template is stored.</p>
            </Card>
          )} />
      )}

      {step === 3 && subject && live.session.session && (
        <Card className="max-w-2xl p-6" data-testid="enrol-result">
          {live.session.session.final_outcome === 'ENROLMENT_CAPTURED' && live.session.session.frame?.activated && (
            <Callout tone="good" title="Enrolment complete">{subject.full_name}’s template is active and ready for verification.</Callout>
          )}
          {live.session.session.final_outcome === 'ENROLMENT_CAPTURED' && live.session.session.frame && !live.session.session.frame.activated && (
            <Callout tone="warn" title="Awaiting administrator approval">
              <p>The capture succeeded but needs review:</p>
              <ul className="mt-1 list-disc pl-5">{(live.session.session.frame.approval_reasons ?? []).map((r) => <li key={r}>{REASON_TEXT[r] ?? humanize(r)}</li>)}</ul>
            </Callout>
          )}
          {live.session.session.final_outcome !== 'ENROLMENT_CAPTURED' && (
            <Callout tone="bad" title={`Enrolment not completed — ${humanize(live.session.session.final_outcome)}`}>
              {live.session.session.frame?.message || humanize(live.session.session.reason_code)}
            </Callout>
          )}
          <div className="mt-5 flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => { live.reset(); setStep(2); }}>Capture again</Button>
            <Button variant="outline" onClick={() => navigate(`/enrolments/${subject.id}`)}>View enrolment</Button>
            <Button onClick={() => { live.reset(); setSubject(null); setStep(0); navigate('/enrol'); }}>Enrol someone else</Button>
          </div>
        </Card>
      )}
    </>
  );
}

function ConsentStep({ subject, text, version, alreadyGiven, onBack, onDone }: {
  subject: UserRow; text: string; version: string; alreadyGiven: boolean; onBack: () => void; onDone: () => void;
}) {
  const me = useAuth((s) => s.user);
  const [accept, setAccept] = useState(false);
  const [pw, setPw] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const self = me?.id === subject.id;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!accept) { setError('The subject must accept the consent statement.'); return; }
    if (!pw) { setError(`${self ? 'Your' : 'The subject’s'} password is required to confirm identity.`); return; }
    setBusy(true);
    try {
      await api.post(`/enrolments/${subject.id}/consent`, { subject_password: pw, accept: true });
      toast.success('Consent recorded');
      setPw('');
      onDone();
    } catch (err) {
      setError(apiError(err).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="max-w-2xl">
      <CardHeader title="Informed consent" subtitle={`Consent statement version ${version}`} />
      <form onSubmit={submit} className="space-y-5 px-5 py-5" noValidate>
        <div className="flex gap-3 rounded-lg bg-surface-2 p-4 text-sm text-ink-2">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden />
          <p>{text}</p>
        </div>
        {alreadyGiven && <Callout tone="info">Consent is already on record; confirming again refreshes it.</Callout>}
        <Checkbox checked={accept} onChange={setAccept}>
          {self ? 'I' : subject.full_name} have read and accept this statement.
        </Checkbox>
        <Field label={self ? 'Your password' : `${subject.full_name}’s password (entered by the subject)`} htmlFor="subject-pw"
          hint="Confirms the person present is the account holder. The operator should not know this password.">
          <Input id="subject-pw" type="password" autoComplete="off" value={pw} onChange={(e) => setPw(e.target.value)} />
        </Field>
        {error && <Callout tone="bad">{error}</Callout>}
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={onBack}>Back</Button>
          <Button type="submit" loading={busy}>Record consent</Button>
          {alreadyGiven && <Button variant="ghost" onClick={onDone}>Skip — consent already given</Button>}
        </div>
        <p className="text-xs text-muted">Consent can be withdrawn at any time from the person’s profile; withdrawal deletes the template. <Link to={`/enrolments/${subject.id}`} className="text-primary hover:underline">Enrolment record</Link></p>
      </form>
    </Card>
  );
}
