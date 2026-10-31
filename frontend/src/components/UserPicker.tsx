import { useEffect, useState } from 'react';
import { Search, UserRound } from 'lucide-react';
import { useApi } from '../lib/query';
import type { UserRow } from '../lib/types';
import { cn } from '../lib/cn';
import { Input } from './ui/form';
import { Badge, Spinner } from './ui/primitives';
import { humanize } from '../lib/format';

export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}

export function UserPicker({ value, onChange, enrolledOnly = false, disabled, label = 'Subject' }: {
  value: UserRow | null; onChange: (u: UserRow) => void; enrolledOnly?: boolean; disabled?: boolean; label?: string;
}) {
  const [search, setSearch] = useState('');
  const q = useDebounced(search);
  const res = useApi<{ users: UserRow[] }>('/users/directory', { search: q, enrolled_only: enrolledOnly, limit: 30 });
  const users = res.data?.users ?? [];
  return (
    <div className="space-y-2">
      <label htmlFor="user-search" className="block text-xs font-medium text-ink-2">{label}</label>
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden />
        <Input id="user-search" className="pl-9" placeholder="Search name or email" value={search} disabled={disabled}
          onChange={(e) => setSearch(e.target.value)} />
      </div>
      <div className="max-h-64 overflow-y-auto rounded-lg border border-line" role="listbox" aria-label={`${label} list`}>
        {res.isLoading ? (
          <div className="flex justify-center p-4 text-muted"><Spinner /></div>
        ) : users.length === 0 ? (
          <p className="p-4 text-center text-sm text-muted">{enrolledOnly ? 'No enrolled users match.' : 'No users match.'}</p>
        ) : users.map((u) => (
          <button key={u.id} type="button" role="option" aria-selected={value?.id === u.id} disabled={disabled}
            onClick={() => onChange(u)}
            className={cn('flex w-full items-center gap-3 border-b border-line px-3 py-2.5 text-left last:border-0 disabled:opacity-60',
              value?.id === u.id ? 'bg-primary-soft/70' : 'hover:bg-surface-2')}>
            <UserRound className="h-4 w-4 shrink-0 text-muted" aria-hidden />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm text-ink">{u.full_name}</span>
              <span className="block truncate text-xs text-muted">{u.email}</span>
            </span>
            <Badge tone={u.enrolment_usable ? 'good' : 'neutral'} icon={false}>
              {u.enrolment_usable ? 'Enrolled' : humanize(u.enrolment_status)}
            </Badge>
          </button>
        ))}
      </div>
    </div>
  );
}
