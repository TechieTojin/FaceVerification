import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileCheck2, Plus } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, humanize } from '../lib/format';
import type { DocumentCheck } from '../lib/types';
import { Button } from '../components/ui/Button';
import { Callout, Card, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Checkbox, Field, Input, Select } from '../components/ui/form';
import { Modal } from '../components/ui/Modal';
import { Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { DecisionBadge } from './DocumentDetail';

export default function Documents() {
  const navigate = useNavigate();
  const [creating, setCreating] = useState(false);
  const q = useApi<{ total: number; checks: DocumentCheck[] }>('/documents', { page_size: 50 });
  return (
    <>
      <PageHeader title="Document check"
        description="Compare a live, liveness-checked face with the portrait on an identity document. Matching a portrait does not prove the document is genuine."
        actions={<Button icon={<Plus className="h-4 w-4" />} onClick={() => setCreating(true)}>New check</Button>} />
      <Card>
        {q.isLoading ? <LoadingBlock rows={5} /> : q.isError ? <Callout tone="bad" className="m-4">{apiError(q.error).message}</Callout> :
          !q.data?.checks.length ? (
            <EmptyState icon={<FileCheck2 className="h-5 w-5" />} title="No document checks yet"
              action={<Button onClick={() => setCreating(true)}>Start a check</Button>} />
          ) : (
            <Table label="Document checks">
              <THead><tr><Th>Created</Th><Th>Subject</Th><Th>Document</Th><Th>Status</Th><Th>Face match</Th><Th>Authenticity</Th><Th>Decision</Th></tr></THead>
              <tbody>{q.data.checks.map((c) => (
                <Tr key={c.id} onClick={() => navigate(`/documents/${c.id}`)}>
                  <Td className="text-xs">{fmtDateTime(c.created_at)}</Td>
                  <Td>{c.subject_label ?? '—'}</Td>
                  <Td>{humanize(c.document_type)}</Td>
                  <Td>{humanize(c.status)}</Td>
                  <Td>{humanize(c.face_match_result)}</Td>
                  <Td>{humanize(c.authenticity_status)}</Td>
                  <Td><DecisionBadge decision={c.overall_decision} /></Td>
                </Tr>
              ))}</tbody>
            </Table>
          )}
      </Card>
      <NewCheckModal open={creating} onClose={() => setCreating(false)} onCreated={(id) => navigate(`/documents/${id}`)} />
    </>
  );
}

function NewCheckModal({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: (id: string) => void }) {
  const types = useApi<{ document_types: string[]; authenticity_provider: { name: string; available: boolean } }>('/documents/types');
  const [label, setLabel] = useState('');
  const [type, setType] = useState('ID_CARD');
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!consent) { setError('The subject must consent before their document is processed.'); return; }
    setBusy(true);
    try {
      const r = await api.post<DocumentCheck>('/documents', { subject_label: label.trim(), document_type: type, consent_given: true });
      onCreated(r.data.id);
    } catch (err) {
      setError(apiError(err).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal open={open} onClose={onClose} title="New document check"
      footer={<><Button variant="outline" onClick={onClose}>Cancel</Button><Button type="submit" form="new-doc" loading={busy}>Create check</Button></>}>
      <form id="new-doc" onSubmit={submit} className="space-y-4" noValidate>
        {error && <Callout tone="bad">{error}</Callout>}
        {types.data && !types.data.authenticity_provider.available && (
          <Callout tone="warn" title="Document authenticity will not be verified">
            No document-forensics provider is configured. The result can confirm that the live face matches the portrait, but not that the document is genuine.
          </Callout>
        )}
        <Field label="Subject reference" htmlFor="doc-label" hint="A name or case number. Avoid entering document numbers.">
          <Input id="doc-label" value={label} onChange={(e) => setLabel(e.target.value)} maxLength={200} />
        </Field>
        <Field label="Document type" htmlFor="doc-type">
          <Select id="doc-type" value={type} onChange={(e) => setType(e.target.value)}>
            {(types.data?.document_types ?? ['ID_CARD']).map((t) => <option key={t} value={t}>{humanize(t)}</option>)}
          </Select>
        </Field>
        <Checkbox checked={consent} onChange={setConsent}>
          The subject consents to their document portrait being compared with a live capture. The document image is processed in memory and not stored.
        </Checkbox>
      </form>
    </Modal>
  );
}
