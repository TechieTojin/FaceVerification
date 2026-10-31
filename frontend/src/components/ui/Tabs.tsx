import { cn } from '../../lib/cn';

export function Tabs<T extends string>({ value, onChange, items, label }: {
  value: T; onChange: (v: T) => void; items: { value: T; label: string; count?: number }[]; label: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="inline-flex flex-wrap gap-1 rounded-lg bg-surface-2 p-1">
      {items.map((it) => (
        <button key={it.value} role="tab" type="button" aria-selected={value === it.value} onClick={() => onChange(it.value)}
          className={cn('rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
            value === it.value ? 'bg-surface text-ink shadow-sm' : 'text-ink-2 hover:text-ink')}>
          {it.label}
          {it.count !== undefined && <span className="ml-1.5 tabular text-muted">{it.count}</span>}
        </button>
      ))}
    </div>
  );
}
