import { forwardRef, useId, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes,
  type TextareaHTMLAttributes } from 'react';
import { cn } from '../../lib/cn';

const CONTROL = 'block w-full rounded-lg border border-line bg-surface px-3 text-sm text-ink placeholder:text-muted ' +
  'transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/25 disabled:opacity-60';

export function Field({ label, hint, error, children, htmlFor, className }: {
  label: ReactNode; hint?: ReactNode; error?: string; children: ReactNode; htmlFor?: string; className?: string;
}) {
  return (
    <div className={cn('space-y-1.5', className)}>
      <label htmlFor={htmlFor} className="block text-xs font-medium text-ink-2">{label}</label>
      {children}
      {error ? <p className="text-xs text-critical" role="alert">{error}</p> : hint ? <p className="text-xs text-muted">{hint}</p> : null}
    </div>
  );
}

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }>(
  function Input({ className, invalid, ...rest }, ref) {
    return <input ref={ref} aria-invalid={invalid || undefined}
      className={cn(CONTROL, 'h-9', invalid && 'border-critical focus:border-critical focus:ring-critical/25', className)} {...rest} />;
  },
);

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  function Select({ className, children, ...rest }, ref) {
    return <select ref={ref} className={cn(CONTROL, 'h-9 pr-8', className)} {...rest}>{children}</select>;
  },
);

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  function Textarea({ className, ...rest }, ref) {
    return <textarea ref={ref} className={cn(CONTROL, 'min-h-[80px] py-2', className)} {...rest} />;
  },
);

export function Switch({ checked, onChange, label, description, disabled }: {
  checked: boolean; onChange: (v: boolean) => void; label: ReactNode; description?: ReactNode; disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <label htmlFor={id} className="text-sm font-medium text-ink">{label}</label>
        {description && <p className="mt-0.5 text-xs text-muted">{description}</p>}
      </div>
      <button id={id} type="button" role="switch" aria-checked={checked} disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn('relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors disabled:opacity-50',
          checked ? 'bg-primary' : 'bg-line')}>
        <span className={cn('inline-block h-5 w-5 rounded-full bg-white shadow transition-transform',
          checked ? 'translate-x-5' : 'translate-x-0.5')} />
      </button>
    </div>
  );
}

export function Checkbox({ checked, onChange, children, id }: {
  checked: boolean; onChange: (v: boolean) => void; children: ReactNode; id?: string;
}) {
  const auto = useId();
  const cid = id ?? auto;
  return (
    <label htmlFor={cid} className="flex cursor-pointer items-start gap-2.5 text-sm text-ink">
      <input id={cid} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 rounded border-line accent-[rgb(var(--primary))]" />
      <span>{children}</span>
    </label>
  );
}
