import type { HTMLAttributes, ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, Info, XCircle } from 'lucide-react';
import { cn } from '../../lib/cn';
import type { Tone } from '../../lib/format';

export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  // min-w-0 lets wide content (tables) scroll inside the card instead of widening grid columns.
  return <div className={cn('min-w-0 rounded-xl bg-surface shadow-card', className)} {...rest} />;
}

export function CardHeader({ title, subtitle, actions, className }: {
  title: ReactNode; subtitle?: ReactNode; actions?: ReactNode; className?: string;
}) {
  return (
    <div className={cn('flex flex-wrap items-start justify-between gap-3 border-b border-line px-5 py-4', className)}>
      <div className="min-w-0">
        <h2 className="text-sm font-semibold text-ink">{title}</h2>
        {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

const TONE_BADGE: Record<Tone, string> = {
  good: 'bg-good/12 text-good-text ring-good/30',
  bad: 'bg-critical/10 text-critical ring-critical/30',
  warn: 'bg-warning/15 text-ink ring-warning/40',
  info: 'bg-primary-soft text-primary-strong ring-primary/25',
  neutral: 'bg-surface-2 text-ink-2 ring-line',
};

const TONE_ICON: Record<Tone, ReactNode> = {
  good: <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />,
  bad: <XCircle className="h-3.5 w-3.5" aria-hidden />,
  warn: <AlertTriangle className="h-3.5 w-3.5" aria-hidden />,
  info: <Info className="h-3.5 w-3.5" aria-hidden />,
  neutral: null,
};

/** Status badge: colour is always paired with an icon and a text label. */
export function Badge({ tone = 'neutral', children, icon = true, className }: {
  tone?: Tone; children: ReactNode; icon?: boolean; className?: string;
}) {
  return (
    <span className={cn('inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset',
      TONE_BADGE[tone], className)}>
      {icon && TONE_ICON[tone]}
      {children}
    </span>
  );
}

export function StatusDot({ tone, label, pulse = false }: { tone: Tone; label: string; pulse?: boolean }) {
  const color = { good: 'bg-good', bad: 'bg-critical', warn: 'bg-warning', info: 'bg-primary', neutral: 'bg-muted' }[tone];
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-ink-2">
      <span className={cn('h-2 w-2 rounded-full', color, pulse && 'animate-pulse2')} aria-hidden />
      {label}
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cn('h-5 w-5 animate-spin', className)} viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-md bg-surface-2', className)} aria-hidden />;
}

export function LoadingBlock({ label = 'Loading…', rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div role="status" aria-label={label} className="space-y-3 p-5">
      {Array.from({ length: rows }, (_, i) => <Skeleton key={i} className="h-5 w-full" />)}
      <span className="sr-only">{label}</span>
    </div>
  );
}

export function EmptyState({ icon, title, body, action, className }: {
  icon?: ReactNode; title: string; body?: ReactNode; action?: ReactNode; className?: string;
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center px-6 py-12 text-center', className)}>
      {icon && <div className="mb-3 rounded-full bg-surface-2 p-3 text-muted">{icon}</div>}
      <p className="text-sm font-medium text-ink">{title}</p>
      {body && <p className="mt-1 max-w-sm text-sm text-muted">{body}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

const CALLOUT: Record<Tone, string> = {
  good: 'border-good/40 bg-good/8',
  bad: 'border-critical/40 bg-critical/8',
  warn: 'border-warning/60 bg-warning/10',
  info: 'border-primary/30 bg-primary-soft/50',
  neutral: 'border-line bg-surface-2',
};

export function Callout({ tone = 'info', title, children, className, action }: {
  tone?: Tone; title?: ReactNode; children?: ReactNode; className?: string; action?: ReactNode;
}) {
  const role = tone === 'bad' ? 'alert' : 'status';
  return (
    <div role={role} className={cn('flex gap-3 rounded-lg border px-4 py-3 text-sm', CALLOUT[tone], className)}>
      <span className={cn('mt-0.5 shrink-0', tone === 'bad' ? 'text-critical' : tone === 'good' ? 'text-good-text' : tone === 'warn' ? 'text-ink' : 'text-primary')}>
        {TONE_ICON[tone] ?? <Info className="h-3.5 w-3.5" />}
      </span>
      <div className="min-w-0 flex-1">
        {title && <p className="font-medium text-ink">{title}</p>}
        {children && <div className={cn('text-ink-2', title && 'mt-0.5')}>{children}</div>}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function StatCard({ label, value, hint, icon, tone }: {
  label: string; value: ReactNode; hint?: ReactNode; icon?: ReactNode; tone?: Tone;
}) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted">{label}</span>
        {icon && <span className={cn('text-muted', tone === 'bad' && 'text-critical', tone === 'good' && 'text-good-text')}>{icon}</span>}
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-tight text-ink">{value}</div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </Card>
  );
}

export function KeyValue({ items, className }: { items: [ReactNode, ReactNode][]; className?: string }) {
  return (
    <dl className={cn('grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2', className)}>
      {items.map(([k, v], i) => (
        <div key={i} className="min-w-0">
          <dt className="text-xs text-muted">{k}</dt>
          <dd className="mt-0.5 break-words text-sm text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
