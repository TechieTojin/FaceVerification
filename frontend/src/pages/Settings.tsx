import { useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { RotateCcw, Save, ShieldAlert } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime } from '../lib/format';
import type { PolicyResponse, PolicySpec, PolicyValue, PolicyVersionRow } from '../lib/types';
import { useCan } from '../store/auth';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Input, Select, Switch, Textarea } from '../components/ui/form';
import { Modal } from '../components/ui/Modal';

const GROUPS: { key: string; title: string; subtitle: string }[] = [
  { key: 'matching', title: 'Matching', subtitle: 'How identity decisions are made' },
  { key: 'liveness', title: 'Liveness', subtitle: 'Passive anti-spoofing and the active challenge' },
  { key: 'session', title: 'Sessions', subtitle: 'Lifetime and processing limits' },
  { key: 'quality', title: 'Quality & enrolment', subtitle: 'Frame quality gates and enrolment rules' },
  { key: 'capture', title: 'Capture', subtitle: 'Sent to browsers at session start' },
  { key: 'alerts', title: 'Alert thresholds', subtitle: 'Suspicious-activity detection rules' },
  { key: 'retention', title: 'Retention', subtitle: 'Automatic deletion of records' },
  { key: 'review', title: 'Manual review', subtitle: 'Which results are routed to a person' },
  { key: 'authentication', title: 'Sign-in & step-up', subtitle: 'Multi-factor requirements for this organisation' },
  { key: 'cross_device', title: 'QR & phone verification', subtitle: 'Pairing lifetime and retries' },
  { key: 'offline', title: 'Offline kiosks', subtitle: 'How long and for how many people kiosks may work offline' },
];

function weakens(spec: PolicySpec, oldV: PolicyValue, newV: PolicyValue): boolean {
  if (!spec.weakening || oldV === newV) return false;
  if (spec.weakening === 'disable') return oldV === true && newV === false;
  if (spec.kind === 'bool') return oldV === false && newV === true;
  if (spec.weakening === 'increase') return Number(newV) > Number(oldV);
  return Number(newV) < Number(oldV);
}

function validate(spec: PolicySpec, raw: string): string | null {
  if (spec.kind === 'bool' || spec.kind === 'choice') return null;
  if (raw.trim() === '' || Number.isNaN(Number(raw))) return 'Enter a number';
  const v = Number(raw);
  if (spec.kind === 'int' && !Number.isInteger(v)) return 'Must be a whole number';
  if (spec.min !== null && v < spec.min) return `Minimum ${spec.min}`;
  if (spec.max !== null && v > spec.max) return `Maximum ${spec.max}`;
  return null;
}

export default function SettingsPage() {
  const can = useCan();
  const editable = can('policy:manage');
  const qc = useQueryClient();
  const q = useApi<PolicyResponse>('/policy');
  const versions = useApi<{ versions: PolicyVersionRow[] }>(editable ? '/policy/versions' : null);
  const [draft, setDraft] = useState<Record<string, string | boolean>>({});
  const [serverErrors, setServerErrors] = useState<Record<string, string>>({});
  const [confirm, setConfirm] = useState<string[] | null>(null);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const p = q.data;

  const reset = () => {
    if (!p) return;
    const d: Record<string, string | boolean> = {};
    for (const s of p.specs) d[s.key] = s.kind === 'bool' ? Boolean(p.values[s.key]) : String(p.values[s.key]);
    setDraft(d);
    setServerErrors({});
  };
  useEffect(reset, [p]); // eslint-disable-line react-hooks/exhaustive-deps

  const specs = useMemo(() => new Map((p?.specs ?? []).map((s) => [s.key, s])), [p]);
  const parsed = (s: PolicySpec): PolicyValue => (s.kind === 'bool' ? Boolean(draft[s.key]) : s.kind === 'choice' ? String(draft[s.key]) : Number(draft[s.key]));
  const errors: Record<string, string> = {};
  const changes: Record<string, PolicyValue> = {};
  if (p) {
    for (const s of p.specs) {
      if (draft[s.key] === undefined) continue;
      const err = s.kind === 'bool' ? null : validate(s, String(draft[s.key]));
      if (err) { errors[s.key] = err; continue; }
      const v = parsed(s);
      if (v !== p.values[s.key]) changes[s.key] = v;
    }
  }
  const weakening = p ? Object.keys(changes).filter((k) => weakens(specs.get(k)!, p.values[k], changes[k])) : [];
  const dirty = Object.keys(changes).length > 0;

  const save = async (confirmed: boolean) => {
    setBusy(true);
    setServerErrors({});
    try {
      const r = await api.put('/policy', { changes, confirm: confirmed, reason });
      toast.success(`Policy saved as version ${r.data.version}`, 'New sessions use it immediately; running sessions keep their snapshot.');
      setConfirm(null);
      setReason('');
      void qc.invalidateQueries({ queryKey: ['/policy'] });
      void qc.invalidateQueries({ queryKey: ['/policy/versions'] });
    } catch (err) {
      const info = apiError(err);
      if (info.code === 'CONFIRMATION_REQUIRED') {
        setConfirm(weakening.length ? weakening : Object.keys(changes));
      } else {
        setServerErrors(info.fieldErrors);
        toast.error('Policy not saved', info.message);
      }
    } finally {
      setBusy(false);
    }
  };

  const onSave = () => (weakening.length ? setConfirm(weakening) : void save(false));

  if (q.isLoading) return <Card><LoadingBlock rows={10} /></Card>;
  if (q.isError || !p) return <Callout tone="bad">{apiError(q.error).message}</Callout>;

  return (
    <>
      <PageHeader title="Policy settings" description={<>Active policy <strong>v{p.version}</strong>{p.updated_at ? `, updated ${fmtDateTime(p.updated_at)}` : ''}. {p.active_sessions_note}</>}
        actions={editable && <>
          <Button variant="outline" icon={<RotateCcw className="h-4 w-4" />} disabled={!dirty || busy} onClick={reset}>Discard</Button>
          <Button icon={<Save className="h-4 w-4" />} disabled={!dirty || Object.keys(errors).length > 0} loading={busy} onClick={onSave} data-testid="save-policy">
            Save {dirty ? `(${Object.keys(changes).length})` : ''}
          </Button>
        </>} />
      {!editable && <Callout tone="info" className="mb-4">You can view the policy; only administrators can change it.</Callout>}
      {weakening.length > 0 && <Callout tone="warn" className="mb-4" title="Some changes weaken security">Saving will ask for explicit confirmation and is logged as a high-severity event.</Callout>}
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-6">
          {GROUPS.map((g) => {
            const items = p.specs.filter((s) => s.group === g.key);
            if (!items.length) return null;
            return (
              <Card key={g.key}>
                <CardHeader title={g.title} subtitle={g.subtitle} />
                <div className="grid gap-x-6 gap-y-5 px-5 py-5 md:grid-cols-2">
                  {items.map((s) => {
                    const changed = s.key in changes;
                    const label = <span className="flex items-center gap-1.5">{s.label}{s.unit ? ` (${s.unit})` : ''}
                      {s.sensitive && <ShieldAlert className="h-3.5 w-3.5 text-muted" aria-label="Security-sensitive" />}
                      {changed && <Badge tone="info" icon={false} className="ml-1">changed</Badge>}</span>;
                    if (s.kind === 'bool') {
                      return <div key={s.key} className="md:col-span-2">
                        <Switch checked={Boolean(draft[s.key])} disabled={!editable} onChange={(v) => setDraft((d) => ({ ...d, [s.key]: v }))}
                          label={label} description={s.description} />
                        {serverErrors[s.key] && <p className="mt-1 text-xs text-critical">{serverErrors[s.key]}</p>}
                      </div>;
                    }
                    const hint = `${s.description}${s.min !== null ? ` Range ${s.min}–${s.max}.` : ''}`;
                    return (
                      <Field key={s.key} label={label} htmlFor={`p-${s.key}`} error={errors[s.key] ?? serverErrors[s.key]} hint={hint}>
                        {s.kind === 'choice' ? (
                          <Select id={`p-${s.key}`} value={String(draft[s.key] ?? '')} disabled={!editable}
                            onChange={(e) => setDraft((d) => ({ ...d, [s.key]: e.target.value }))}>
                            {s.choices.map((c) => <option key={c} value={c}>{c}</option>)}
                          </Select>
                        ) : (
                          <Input id={`p-${s.key}`} type="number" inputMode="decimal" step={s.kind === 'int' ? 1 : 0.01} min={s.min ?? undefined}
                            max={s.max ?? undefined} value={String(draft[s.key] ?? '')} disabled={!editable} invalid={!!errors[s.key]}
                            onChange={(e) => setDraft((d) => ({ ...d, [s.key]: e.target.value }))} />
                        )}
                      </Field>
                    );
                  })}
                </div>
              </Card>
            );
          })}
        </div>
        {editable && (
          <Card className="h-fit xl:sticky xl:top-4">
            <CardHeader title="Version history" />
            <ol className="max-h-[70vh] divide-y divide-line overflow-y-auto">
              {(versions.data?.versions ?? []).map((v) => (
                <li key={v.version} className="px-5 py-3 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-ink">v{v.version}</span>
                    {v.is_sensitive_change && <Badge tone="warn">Weakening</Badge>}
                  </div>
                  <p className="mt-1 text-muted">{fmtDateTime(v.created_at)} · {v.created_by ?? 'system'}</p>
                  {v.change_summary && <p className="mt-1 break-words text-ink-2">{v.change_summary}</p>}
                </li>
              ))}
            </ol>
          </Card>
        )}
      </div>
      <Modal open={confirm !== null} onClose={() => setConfirm(null)} title="Confirm security-weakening changes"
        description="These changes reduce protection against spoofing or misidentification."
        footer={<><Button variant="outline" onClick={() => setConfirm(null)}>Cancel</Button>
          <Button variant="danger" loading={busy} disabled={!reason.trim()} onClick={() => void save(true)}>Confirm and save</Button></>}>
        <div className="space-y-4 text-sm">
          <ul className="list-disc space-y-1 pl-5 text-ink-2">
            {(confirm ?? []).map((k) => <li key={k}><strong className="text-ink">{specs.get(k)?.label ?? k}</strong>: {String(p.values[k])} → {String(changes[k])}</li>)}
          </ul>
          <Field label="Reason (recorded in the audit log)" htmlFor="policy-reason">
            <Textarea id="policy-reason" value={reason} onChange={(e) => setReason(e.target.value)} maxLength={500} />
          </Field>
        </div>
      </Modal>
    </>
  );
}
