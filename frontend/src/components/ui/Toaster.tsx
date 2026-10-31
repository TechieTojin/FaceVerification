import { CheckCircle2, Info, X, XCircle } from 'lucide-react';
import { useToasts } from '../../store/toast';
import { cn } from '../../lib/cn';

export function Toaster() {
  const { toasts, dismiss } = useToasts();
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[min(380px,calc(100vw-2rem))] flex-col gap-2"
      aria-live="polite" role="region" aria-label="Notifications">
      {toasts.map((t) => (
        <div key={t.id} className="pointer-events-auto flex animate-fade-in items-start gap-3 rounded-xl bg-surface p-3.5 shadow-pop">
          <span className={cn('mt-0.5', t.tone === 'success' ? 'text-good-text' : t.tone === 'error' ? 'text-critical' : 'text-primary')}>
            {t.tone === 'success' ? <CheckCircle2 className="h-4 w-4" /> : t.tone === 'error' ? <XCircle className="h-4 w-4" /> : <Info className="h-4 w-4" />}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-ink">{t.title}</p>
            {t.body && <p className="mt-0.5 text-xs text-muted">{t.body}</p>}
          </div>
          <button type="button" className="text-muted hover:text-ink" onClick={() => dismiss(t.id)} aria-label="Dismiss notification">
            <X className="h-4 w-4" />
          </button>
        </div>
      ))}
    </div>
  );
}
