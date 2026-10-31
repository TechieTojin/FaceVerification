import { useState, type FormEvent } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, RefreshCw, Send, Webhook } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtRelative, humanize } from '../lib/format';
import type { ApiClient } from '../lib/types';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, EmptyState, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Input, Select } from '../components/ui/form';
import { ConfirmDialog, Modal } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { CopyField } from '../components/CopyField';
import { SCOPE_TEXT } from './Developers';

const STATE_TONE: Record<string, 'good' | 'warn' | 'bad' | 'neutral'> = { ACTIVE: 'good', EXPIRING: 'warn', EXPIRED: 'neutral', REVOKED: 'bad' };
const DELIVERY_TONE: Record<string, 'good' | 'warn' | 'bad'> = { DELIVERED: 'good', PENDING: 'warn', FAILED: 'bad' };
const EVENTS = ['verification.completed', 'verification.reviewed', 'ping'];

export default function DeveloperClient() {
  const { clientId = '' } = useParams();
  const qc = useQueryClient();
  const q = useApi<ApiClient>(`/integrations/clients/${clientId}`);
  const [secret, setSecret] = useState<{ title: string; value: string } | null>(null);
  const [rotating, setRotating] = useState(false);
  const [grace, setGrace] = useState(0);
  const [revokeClient, setRevokeClient] = useState(false);
  const [hook, setHook] = useState<{ url: string; events: string[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const refresh = () => void qc.invalidateQueries({ queryKey: [`/integrations/clients/${clientId}`] });

  const run = async (key: string, fn: () => Promise<void>) => {
    setBusy(key); setError(null);
    try { await fn(); } catch (err) { setError(apiError(err).message); } finally { setBusy(null); }
  };

  const c = q.data;
  if (q.isLoading || !c) return q.isError ? <Callout tone="bad">{apiError(q.error).message}</Callout> : <LoadingBlock rows={6} />;
  const active = c.status === 'ACTIVE';

  const addHook = (e: FormEvent) => {
    e.preventDefault();
    if (hook) void run('hook', async () => {
      const r = await api.post<{ secret: string }>(`/integrations/clients/${c.id}/webhooks`, hook);
      setHook(null);
      setSecret({ title: 'Webhook signing secret', value: r.data.secret });
      refresh();
    });
  };

  return (
    <>
      <Link to="/developers" className="mb-3 inline-flex items-center gap-1 text-sm text-muted hover:text-ink"><ArrowLeft className="h-4 w-4" />API clients</Link>
      <PageHeader title={c.name} description={c.description ?? 'Integration client'}
        actions={active && <div className="flex gap-2">
          <Button variant="outline" icon={<RefreshCw className="h-4 w-4" />} onClick={() => setRotating(true)} data-testid="rotate-key">Rotate key</Button>
          <Button variant="danger" onClick={() => setRevokeClient(true)}>Revoke client</Button></div>} />
      {error && <Callout tone="bad" className="mb-4">{error}</Callout>}
      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Client" actions={<Badge tone={active ? 'good' : 'bad'}>{c.status}</Badge>} />
          <div className="p-5"><KeyValue items={[
            ['Client id (audience)', <code key="id" className="break-all text-xs">{c.id}</code>],
            ['Scopes', c.scopes.map((s) => SCOPE_TEXT[s] ?? s).join('; ')],
            ['Rate limit', `${c.rate_limit_per_minute} requests / minute`],
            ['Created', fmtDateTime(c.created_at)],
          ]} /></div>
        </Card>
        <Card>
          <CardHeader title="Credentials" subtitle="Secrets are stored only as keyed hashes." />
          <Table label="Credentials">
            <THead><tr><Th>Key</Th><Th>State</Th><Th>Last used</Th><Th className="text-right" /></tr></THead>
            <tbody>{(c.credentials ?? []).map((k) => (
              <Tr key={k.id}>
                <Td><code className="text-xs">{k.prefix}</code><span className="block text-xs text-muted">created {fmtRelative(k.created_at)}{k.expires_at ? ` · expires ${fmtRelative(k.expires_at)}` : ''}</span></Td>
                <Td><Badge tone={STATE_TONE[k.state] ?? 'neutral'}>{humanize(k.state)}</Badge></Td>
                <Td>{k.last_used_at ? fmtRelative(k.last_used_at) : 'Never'}</Td>
                <Td className="text-right">{k.state !== 'REVOKED' && k.state !== 'EXPIRED' && (
                  <Button size="sm" variant="ghost" loading={busy === k.id} onClick={() => void run(k.id, async () => {
                    await api.post(`/integrations/clients/${c.id}/credentials/${k.id}/revoke`); refresh();
                  })}>Revoke</Button>)}</Td>
              </Tr>))}</tbody>
          </Table>
        </Card>
        <Card className="xl:col-span-2">
          <CardHeader title="Webhooks" subtitle="Signed with FV-Signature (HMAC-SHA256); failed deliveries retry with back-off."
            actions={active && <Button size="sm" icon={<Webhook className="h-4 w-4" />} onClick={() => setHook({ url: '', events: ['verification.completed', 'verification.reviewed'] })}
              data-testid="add-webhook">Add endpoint</Button>} />
          {!c.webhooks?.length ? <EmptyState icon={<Webhook className="h-5 w-5" />} title="No webhook endpoints" /> : (
            <Table label="Webhooks">
              <THead><tr><Th>URL</Th><Th>Events</Th><Th>Status</Th><Th className="text-right">Actions</Th></tr></THead>
              <tbody>{c.webhooks.map((w) => (
                <Tr key={w.id}>
                  <Td className="max-w-xs break-all font-mono text-xs">{w.url}</Td>
                  <Td className="text-xs">{w.events.join(', ')}</Td>
                  <Td><Badge tone={w.status === 'ACTIVE' ? 'good' : 'neutral'}>{w.status}</Badge></Td>
                  <Td className="text-right"><div className="flex justify-end gap-1">
                    <Button size="sm" variant="ghost" icon={<Send className="h-3.5 w-3.5" />} loading={busy === `t${w.id}`} disabled={w.status !== 'ACTIVE'}
                      onClick={() => void run(`t${w.id}`, async () => {
                        const r = await api.post<{ status: string; last_status_code: number | null; last_error: string | null }>(`/integrations/webhooks/${w.id}/test`);
                        if (r.data.status === 'DELIVERED') toast.success('Test event delivered', `HTTP ${r.data.last_status_code}`);
                        else toast.error('Test delivery failed', r.data.last_error ?? 'Will be retried');
                        refresh();
                      })}>Send test</Button>
                    <Button size="sm" variant="ghost" disabled={w.status !== 'ACTIVE'} onClick={() => void run(`s${w.id}`, async () => {
                      const r = await api.post<{ secret: string }>(`/integrations/webhooks/${w.id}/rotate-secret`);
                      setSecret({ title: 'New webhook signing secret', value: r.data.secret });
                    })}>Rotate secret</Button>
                    <Button size="sm" variant="ghost" disabled={w.status !== 'ACTIVE'} onClick={() => void run(`d${w.id}`, async () => {
                      await api.post(`/integrations/webhooks/${w.id}/disable`); refresh();
                    })}>Disable</Button>
                  </div></Td>
                </Tr>))}</tbody>
            </Table>
          )}
          {!!c.deliveries?.length && (
            <div className="border-t border-line">
              <p className="px-5 pt-4 text-sm font-semibold text-ink">Recent deliveries</p>
              <Table label="Deliveries">
                <THead><tr><Th>Event</Th><Th>Status</Th><Th>Attempts</Th><Th>Response</Th><Th>Next attempt</Th><Th className="text-right" /></tr></THead>
                <tbody>{c.deliveries.map((d) => (
                  <Tr key={d.id}>
                    <Td><code className="text-xs">{d.event_type}</code><span className="block text-xs text-muted">{fmtRelative(d.created_at)}</span></Td>
                    <Td><Badge tone={DELIVERY_TONE[d.status] ?? 'neutral'}>{humanize(d.status)}</Badge></Td>
                    <Td className="tabular">{d.attempts}</Td>
                    <Td className="text-xs">{d.last_status_code ? `HTTP ${d.last_status_code}` : d.last_error ?? '—'}</Td>
                    <Td>{d.next_attempt_at ? fmtRelative(d.next_attempt_at) : '—'}</Td>
                    <Td className="text-right">{d.status !== 'DELIVERED' && (
                      <Button size="sm" variant="ghost" loading={busy === d.id} onClick={() => void run(d.id, async () => {
                        await api.post(`/integrations/deliveries/${d.id}/retry`); refresh();
                      })}>Retry now</Button>)}</Td>
                  </Tr>))}</tbody>
              </Table>
            </div>
          )}
        </Card>
      </div>

      <Modal open={rotating} onClose={() => setRotating(false)} title="Rotate API key"
        description="A new key is issued. Existing keys stop working after the grace period, so you can deploy the new one first.">
        <div className="space-y-4">
          <Field label="Grace period for existing keys" htmlFor="grace">
            <Select id="grace" value={grace} onChange={(e) => setGrace(Number(e.target.value))}>
              <option value={0}>None — old keys stop now</option><option value={1}>1 hour</option><option value={24}>24 hours</option><option value={72}>72 hours</option>
            </Select>
          </Field>
          <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setRotating(false)}>Cancel</Button>
            <Button loading={busy === 'rotate'} onClick={() => void run('rotate', async () => {
              const r = await api.post<{ api_key: string }>(`/integrations/clients/${c.id}/credentials/rotate`, { grace_hours: grace });
              setRotating(false);
              setSecret({ title: 'New API key', value: r.data.api_key });
              refresh();
            })}>Rotate</Button></div>
        </div>
      </Modal>
      <Modal open={!!hook} onClose={() => setHook(null)} title="Add webhook endpoint" description="HTTPS public endpoints only. The signing secret is shown once.">
        {hook && (
          <form onSubmit={addHook} className="space-y-4">
            <Field label="URL" htmlFor="hook-url"><Input id="hook-url" type="url" placeholder="https://example.com/webhooks/faceverification" value={hook.url}
              onChange={(e) => setHook({ ...hook, url: e.target.value })} required /></Field>
            <fieldset className="space-y-1"><legend className="text-sm font-medium text-ink">Events</legend>
              {EVENTS.map((ev) => <label key={ev} className="flex items-center gap-2 text-sm"><input type="checkbox" checked={hook.events.includes(ev)}
                onChange={(e) => setHook({ ...hook, events: e.target.checked ? [...hook.events, ev] : hook.events.filter((x) => x !== ev) })} /><code className="text-xs">{ev}</code></label>)}
            </fieldset>
            <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setHook(null)}>Cancel</Button>
              <Button type="submit" loading={busy === 'hook'} disabled={!hook.url || !hook.events.length}>Add endpoint</Button></div>
          </form>
        )}
      </Modal>
      <Modal open={!!secret} onClose={() => setSecret(null)} title={secret?.title ?? ''} description="Copy it now. It will not be shown again."
        footer={<Button onClick={() => setSecret(null)}>Done</Button>}>
        {secret && <div data-testid="one-time-secret"><CopyField value={secret.value} label={secret.title} /></div>}
      </Modal>
      <ConfirmDialog open={revokeClient} onClose={() => setRevokeClient(false)} danger confirmLabel="Revoke client" typeToConfirm={c.name}
        title={`Revoke ${c.name}?`} body="All keys stop working immediately and webhooks are disabled. This cannot be undone."
        onConfirm={() => void run('revoke', async () => { await api.post(`/integrations/clients/${c.id}/revoke`); setRevokeClient(false); refresh(); })} />
    </>
  );
}
