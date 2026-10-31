import { useState } from 'react';
import { apiError } from '../lib/api';
import { useApi } from '../lib/query';
import { fmtDateTime, humanize } from '../lib/format';
import type { AuditEntry, SecurityEventRow } from '../lib/types';
import { Badge, Callout, Card, EmptyState, LoadingBlock, PageHeader } from '../components/ui/primitives';
import { Select } from '../components/ui/form';
import { Pagination, Table, Td, Th, THead, Tr } from '../components/ui/Table';
import { Tabs } from '../components/ui/Tabs';

const SEV = { HIGH: 'bad', CRITICAL: 'bad', WARNING: 'warn', INFO: 'info' } as const;

export default function SecurityLog() {
  const [tab, setTab] = useState<'events' | 'audit'>('events');
  return (
    <>
      <PageHeader title="Security log" description="Security events and the administrative audit trail. Neither ever contains images, templates or secrets." />
      <div className="mb-4"><Tabs label="Log type" value={tab} onChange={setTab} items={[{ value: 'events', label: 'Security events' }, { value: 'audit', label: 'Audit trail' }]} /></div>
      {tab === 'events' ? <Events /> : <Audit />}
    </>
  );
}

function Events() {
  const [severity, setSeverity] = useState('');
  const [type, setType] = useState('');
  const [page, setPage] = useState(1);
  const q = useApi<{ total: number; event_types: string[]; events: SecurityEventRow[] }>('/security/events',
    { severity, event_type: type, page, page_size: 50 }, { placeholderData: (p) => p });
  return (
    <Card>
      <div className="flex flex-wrap gap-3 border-b border-line p-4">
        <Select aria-label="Severity" value={severity} onChange={(e) => { setSeverity(e.target.value); setPage(1); }} className="w-40">
          <option value="">All severities</option><option value="HIGH">High</option><option value="WARNING">Warning</option><option value="INFO">Info</option>
        </Select>
        <Select aria-label="Event type" value={type} onChange={(e) => { setType(e.target.value); setPage(1); }} className="w-64">
          <option value="">All event types</option>{(q.data?.event_types ?? []).map((t) => <option key={t} value={t}>{humanize(t)}</option>)}
        </Select>
      </div>
      {q.isLoading ? <LoadingBlock rows={8} /> : q.isError ? <Callout tone="bad" className="m-4">{apiError(q.error).message}</Callout> :
        !q.data?.events.length ? <EmptyState title="No events" /> : (
          <>
            <Table label="Security events">
              <THead><tr><Th>Time</Th><Th>Event</Th><Th>Severity</Th><Th>Description</Th><Th>User</Th><Th>IP</Th></tr></THead>
              <tbody>{q.data.events.map((e) => (
                <Tr key={e.id}>
                  <Td className="whitespace-nowrap text-xs">{fmtDateTime(e.timestamp)}</Td>
                  <Td className="text-xs font-medium text-ink">{humanize(e.event_type)}</Td>
                  <Td><Badge tone={SEV[e.severity as keyof typeof SEV] ?? 'neutral'}>{humanize(e.severity)}</Badge></Td>
                  <Td className="max-w-md truncate text-xs" title={e.description ?? ''}>{e.description}</Td>
                  <Td className="text-xs">{e.user_email ?? '—'}</Td>
                  <Td className="font-mono text-xs">{e.ip_address ?? '—'}</Td>
                </Tr>
              ))}</tbody>
            </Table>
            <Pagination page={page} pageSize={50} total={q.data.total} onPage={setPage} />
          </>
        )}
    </Card>
  );
}

function Audit() {
  const [action, setAction] = useState('');
  const [page, setPage] = useState(1);
  const q = useApi<{ total: number; actions: string[]; entries: AuditEntry[] }>('/security/audit', { action, page, page_size: 50 },
    { placeholderData: (p) => p });
  return (
    <Card>
      <div className="flex flex-wrap gap-3 border-b border-line p-4">
        <Select aria-label="Action" value={action} onChange={(e) => { setAction(e.target.value); setPage(1); }} className="w-64">
          <option value="">All actions</option>{(q.data?.actions ?? []).map((a) => <option key={a} value={a}>{humanize(a)}</option>)}
        </Select>
      </div>
      {q.isLoading ? <LoadingBlock rows={8} /> : !q.data?.entries.length ? <EmptyState title="No audit entries" /> : (
        <>
          <Table label="Audit trail">
            <THead><tr><Th>Time</Th><Th>Action</Th><Th>Actor</Th><Th>Target</Th><Th>Details</Th></tr></THead>
            <tbody>{q.data.entries.map((e) => (
              <Tr key={e.id}>
                <Td className="whitespace-nowrap text-xs">{fmtDateTime(e.timestamp)}</Td>
                <Td className="text-xs font-medium text-ink">{humanize(e.action)}</Td>
                <Td className="text-xs">{e.actor ?? 'system'}</Td>
                <Td className="font-mono text-xs">{e.target_type ? `${e.target_type}:${(e.target_id ?? '').slice(0, 8)}` : '—'}</Td>
                <Td className="max-w-md truncate font-mono text-[11px]" title={e.details ? JSON.stringify(e.details) : ''}>
                  {e.details ? JSON.stringify(e.details) : '—'}
                </Td>
              </Tr>
            ))}</tbody>
          </Table>
          <Pagination page={page} pageSize={50} total={q.data.total} onPage={setPage} />
        </>
      )}
    </Card>
  );
}
