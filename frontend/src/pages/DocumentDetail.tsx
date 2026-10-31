import { useCallback, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { FileImage, Trash2, Upload } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, fmtNum, humanize } from '../lib/format';
import type { DocumentCheck, SessionView } from '../lib/types';
import { toast } from '../store/toast';
import { Button } from '../components/ui/Button';
import { Badge, Callout, Card, CardHeader, KeyValue, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { ConfirmDialog } from '../components/ui/Modal';
import { LiveCapturePanel } from '../features/camera/LiveCapturePanel';
import { useLiveCapture } from '../features/camera/useLiveCapture';

const DECISION: Record<string, { label: string; tone: 'good' | 'bad' | 'warn' | 'neutral' | 'info' }> = {
  IDENTITY_VERIFIED: { label: 'Identity verified', tone: 'good' },
  FACE_MATCH_ONLY: { label: 'Face matches — document not authenticated', tone: 'warn' },
  NOT_VERIFIED: { label: 'Face does not match', tone: 'bad' },
  LIVENESS_FAILED: { label: 'Liveness failed', tone: 'bad' },
  INCOMPLETE: { label: 'Incomplete', tone: 'neutral' },
  PENDING: { label: 'Pending', tone: 'info' },
};

export function DecisionBadge({ decision }: { decision: string }) {
  const d = DECISION[decision] ?? { label: humanize(decision), tone: 'neutral' as const };
  return <Badge tone={d.tone}>{d.label}</Badge>;
}

export default function DocumentDetail() {
  const { checkId = '' } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const q = useApi<DocumentCheck>(`/documents/${checkId}`);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [del, setDel] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const start = useCallback(async () => (await api.post<SessionView>(`/documents/${checkId}/session`)).data, [checkId]);
  const live = useLiveCapture(start, () => void qc.invalidateQueries({ queryKey: [`/documents/${checkId}`] }));
  const d = q.data;

  const upload = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    const form = new FormData();
    form.append('file', file);
    try {
      await api.post(`/documents/${checkId}/image`, form, { timeout: 90_000 });
      toast.success('Portrait detected', 'Continue with the live capture.');
      void q.refetch();
    } catch (err) {
      setUploadError(apiError(err).message);
      void q.refetch();
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const remove = async () => {
    try {
      await api.delete(`/documents/${checkId}`);
      toast.success('Check deleted and portrait template purged');
      navigate('/documents');
    } catch (err) {
      toast.error('Delete failed', apiError(err).message);
    }
  };

  if (q.isLoading) return <Card><LoadingBlock rows={6} /></Card>;
  if (q.isError || !d) return <Callout tone="bad" title="Document check unavailable">{apiError(q.error).message}</Callout>;

  return (
    <>
      <PageHeader title={d.subject_label || 'Document check'} description={`${humanize(d.document_type)} · created ${fmtDateTime(d.created_at)}`}
        actions={<>
          <DecisionBadge decision={d.overall_decision} />
          {d.status !== 'DELETED' && <Button variant="outline" icon={<Trash2 className="h-4 w-4" />} onClick={() => setDel(true)}>Delete</Button>}
        </>} />

      <div className="mb-6 grid gap-4 md:grid-cols-3" data-testid="document-results">
        <ResultCard title="Face ↔ portrait match" value={humanize(d.face_match_result)}
          tone={d.face_match_result === 'MATCH' ? 'good' : d.face_match_result === 'NO_MATCH' ? 'bad' : 'neutral'}
          hint={d.face_match_distance !== null ? `Distance ${fmtNum(d.face_match_distance)} (threshold ${fmtNum(d.threshold, 2)})` : 'Not evaluated yet'} />
        <ResultCard title="Document authenticity" value={humanize(d.authenticity_status)}
          tone={d.authenticity_status === 'PASSED' ? 'good' : d.authenticity_status === 'FAILED' ? 'bad' : 'warn'}
          hint={d.authenticity_detail ?? undefined} />
        <ResultCard title="Overall decision" value={(DECISION[d.overall_decision] ?? { label: humanize(d.overall_decision) }).label}
          tone={DECISION[d.overall_decision]?.tone ?? 'neutral'}
          hint="Identity is verified only when the face matches AND the document is authenticated." />
      </div>

      {d.status === 'AWAITING_DOCUMENT' && (
        <Card className="max-w-2xl">
          <CardHeader title="1. Upload the document" subtitle="Photograph the document flat, well lit, without glare." />
          <div className="space-y-4 px-5 py-5">
            {(uploadError || d.document_quality_issues.length > 0) && (
              <Callout tone="bad" title="The last upload could not be used">
                {uploadError ?? d.document_quality_issues.map(humanize).join(', ')}
              </Callout>
            )}
            <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" id="doc-file"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void upload(f); }} />
            <div className="flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-line p-8 text-center">
              <FileImage className="h-8 w-8 text-muted" />
              <p className="text-sm text-ink-2">JPEG, PNG or WebP up to 8 MB. The image is analysed in memory and discarded.</p>
              <Button icon={<Upload className="h-4 w-4" />} loading={uploading} onClick={() => fileRef.current?.click()}>Choose image</Button>
            </div>
          </div>
        </Card>
      )}

      {d.status === 'AWAITING_LIVE_CAPTURE' && (
        <>
          <Callout tone="info" className="mb-4" title="2. Live capture">
            Portrait found (document quality {fmtNum(d.document_quality_score, 2)}). Now capture the person live; they will be asked to complete a short liveness challenge.
          </Callout>
          <LiveCapturePanel live={live} startLabel="Start live capture" canStart
            onAgain={() => { live.reset(); void q.refetch(); }} />
        </>
      )}

      {(d.status === 'COMPLETED' || d.status === 'EXPIRED' || d.status === 'DELETED') && (
        <Card className="max-w-3xl">
          <CardHeader title="Record" />
          <div className="px-5 py-4">
            <KeyValue items={[
              ['Status', humanize(d.status)], ['Completed', fmtDateTime(d.completed_at)],
              ['Liveness', d.liveness_passed === null ? '—' : d.liveness_passed ? 'Passed' : 'Failed'],
              ['Document quality', fmtNum(d.document_quality_score, 2)],
              ['Portrait template', d.portrait_template_stored ? 'Stored (encrypted)' : `Purged ${fmtDateTime(d.purged_at)}`],
              ['Authenticity provider', d.authenticity_provider ?? '—'],
            ]} />
            {d.session_id && <Button className="mt-4" variant="outline" size="sm" onClick={() => navigate(`/history/${d.session_id}`)}>Live capture session</Button>}
          </div>
        </Card>
      )}

      <ConfirmDialog open={del} onClose={() => setDel(false)} onConfirm={() => void remove()} danger title="Delete this check?"
        confirmLabel="Delete" body={<p>The encrypted portrait template (if any) is purged immediately. The audit record is kept.</p>} />
    </>
  );
}

function ResultCard({ title, value, tone, hint }: { title: string; value: string; tone: 'good' | 'bad' | 'warn' | 'neutral' | 'info'; hint?: string }) {
  return (
    <Card className="p-4">
      <p className="text-xs font-medium text-muted">{title}</p>
      <div className="mt-2"><Badge tone={tone}>{value}</Badge></div>
      {hint && <p className="mt-2 text-xs text-muted">{hint}</p>}
    </Card>
  );
}
