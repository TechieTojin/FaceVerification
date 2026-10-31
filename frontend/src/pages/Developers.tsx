import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { BookOpen, KeySquare, Plus } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtRelative } from '../lib/format';
import type { ApiClient } from '../lib/types';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Field, Input, Textarea } from '../components/ui/form';
import { Modal } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { CopyField } from '../components/CopyField';

export const SCOPE_TEXT: Record<string, string> = {
  'sessions:write': 'Create hosted verifications',
  'sessions:read': 'Read verification status and assertions',
  'assertions:introspect': 'Validate assertions server-side (once each)',
};

export default function Developers() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const q = useApi<{ clients: ApiClient[]; scopes: string[]; issuer: string }>('/integrations/clients');
  const [form, setForm] = useState<{ name: string; description: string; scopes: string[]; rate: number } | null>(null);
  const [created, setCreated] = useState<ApiClient | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    if (!form) return;
    setBusy(true); setError(null);
    try {
      const r = await api.post<ApiClient>('/integrations/clients', { name: form.name, description: form.description,
        scopes: form.scopes, rate_limit_per_minute: form.rate });
      setCreated(r.data);
      setForm(null);
      void qc.invalidateQueries({ queryKey: ['/integrations/clients'] });
    } catch (err) { setError(apiError(err).message); } finally { setBusy(false); }
  };

  return (
    <>
      <PageHeader title="API clients" description="Server-to-server integrations that start hosted verifications and receive signed results. Keys belong on your servers, never in a browser or app bundle."
        actions={<div className="flex gap-2">
          <Link to="/developers/docs" className="inline-flex h-9 items-center gap-2 rounded-lg border border-line px-4 text-sm text-ink hover:bg-surface-2"><BookOpen className="h-4 w-4" />API docs</Link>
          <Button icon={<Plus className="h-4 w-4" />} data-testid="new-client"
            onClick={() => setForm({ name: '', description: '', scopes: q.data?.scopes ?? [], rate: 60 })}>New client</Button></div>} />
      <Card>
        {q.isLoading ? <LoadingBlock rows={4} /> : !q.data?.clients.length
          ? <EmptyState icon={<KeySquare className="h-5 w-5" />} title="No API clients" body="Create a client to integrate verification into your own product." /> : (
            <Table label="API clients">
              <THead><tr><Th>Client</Th><Th>Scopes</Th><Th>Status</Th><Th>Active keys</Th><Th>Rate limit</Th><Th>Created</Th></tr></THead>
              <tbody>{q.data.clients.map((c) => (
                <Tr key={c.id} onClick={() => navigate(`/developers/${c.id}`)} className="cursor-pointer">
                  <Td className="font-medium text-ink">{c.name}{c.description && <span className="block text-xs font-normal text-muted">{c.description}</span>}</Td>
                  <Td className="text-xs">{c.scopes.join(', ')}</Td>
                  <Td><Badge tone={c.status === 'ACTIVE' ? 'good' : 'bad'}>{c.status}</Badge></Td>
                  <Td className="tabular">{c.active_credentials}</Td>
                  <Td className="tabular">{c.rate_limit_per_minute}/min</Td>
                  <Td>{fmtRelative(c.created_at)}</Td>
                </Tr>))}</tbody>
            </Table>
          )}
      </Card>
      <Modal open={!!form} onClose={() => setForm(null)} title="New API client" description="Creating credentials is a high-risk action and asks you to confirm your identity.">
        {form && (
          <form onSubmit={create} className="space-y-4">
            {error && <Callout tone="bad">{error}</Callout>}
            <Field label="Name" htmlFor="cl-name"><Input id="cl-name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required /></Field>
            <Field label="Description" htmlFor="cl-desc"><Textarea id="cl-desc" rows={2} value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></Field>
            <fieldset className="space-y-2"><legend className="text-sm font-medium text-ink">Scopes</legend>
              {(q.data?.scopes ?? []).map((s) => (
                <label key={s} className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={form.scopes.includes(s)}
                  onChange={(e) => setForm({ ...form, scopes: e.target.checked ? [...form.scopes, s] : form.scopes.filter((x) => x !== s) })} />
                  <span><code className="text-xs">{s}</code><span className="block text-xs text-muted">{SCOPE_TEXT[s]}</span></span></label>))}
            </fieldset>
            <Field label="Requests per minute" htmlFor="cl-rate"><Input id="cl-rate" type="number" min={1} max={6000} value={form.rate} onChange={(e) => setForm({ ...form, rate: Number(e.target.value) })} /></Field>
            <div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setForm(null)}>Cancel</Button>
              <Button type="submit" loading={busy} disabled={form.name.trim().length < 2 || !form.scopes.length}>Create client</Button></div>
          </form>
        )}
      </Modal>
      <Modal open={!!created} onClose={() => setCreated(null)} title="API key created" description="Copy the key now and store it in your server's secret store. It is never shown again."
        footer={<Button onClick={() => { const id = created?.id; setCreated(null); if (id) navigate(`/developers/${id}`); }}>Done</Button>}>
        {created?.api_key && <div className="space-y-2" data-testid="api-key"><CopyField value={created.api_key} label="API key" />
          <p className="text-xs text-muted">Client id (assertion audience): <code>{created.id}</code></p></div>}
      </Modal>
    </>
  );
}
