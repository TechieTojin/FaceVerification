import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { Button } from './Button';
import { Field, Input, Textarea } from './form';

export function Modal({ open, onClose, title, description, children, footer, size = 'md' }: {
  open: boolean; onClose: () => void; title: ReactNode; description?: ReactNode; children?: ReactNode;
  footer?: ReactNode; size?: 'sm' | 'md' | 'lg' | 'xl';
}) {
  const titleId = useId();
  const panel = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeRef.current();
      if (e.key === 'Tab' && panel.current) {
        const f = panel.current.querySelectorAll<HTMLElement>('button,input,select,textarea,a[href],[tabindex]:not([tabindex="-1"])');
        if (!f.length) return;
        const first = f[0];
        const last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };
    document.addEventListener('keydown', onKey);
    const t = window.setTimeout(() => {
      const target = panel.current?.querySelector<HTMLElement>('[data-autofocus],input,textarea,select,button');
      target?.focus();
    }, 0);
    return () => {
      document.removeEventListener('keydown', onKey);
      window.clearTimeout(t);
      prev?.focus?.();
    };
  }, [open]);

  if (!open) return null;
  const width = { sm: 'max-w-md', md: 'max-w-lg', lg: 'max-w-2xl', xl: 'max-w-5xl' }[size];
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-end justify-center p-0 sm:items-center sm:p-4">
      <div className="absolute inset-0 bg-black/45" onClick={onClose} aria-hidden />
      <div ref={panel} role="dialog" aria-modal="true" aria-labelledby={titleId}
        className={`relative w-full ${width} animate-fade-in rounded-t-2xl bg-surface shadow-pop sm:rounded-2xl`}>
        <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div>
            <h2 id={titleId} className="text-base font-semibold text-ink">{title}</h2>
            {description && <p className="mt-1 text-sm text-muted">{description}</p>}
          </div>
          <button type="button" onClick={onClose} className="rounded-md p-1 text-muted hover:bg-surface-2 hover:text-ink" aria-label="Close dialog">
            <X className="h-4 w-4" />
          </button>
        </div>
        {children && <div className="max-h-[70vh] overflow-y-auto px-5 py-4">{children}</div>}
        {footer && <div className="flex flex-wrap justify-end gap-2 border-t border-line px-5 py-3">{footer}</div>}
      </div>
    </div>,
    document.body,
  );
}

/** Confirmation dialog; optionally requires a note or typing a confirmation word. */
export function ConfirmDialog({ open, onClose, onConfirm, title, body, confirmLabel = 'Confirm', danger = false,
  loading = false, noteLabel, noteRequired = false, typeToConfirm }: {
  open: boolean; onClose: () => void; onConfirm: (note: string) => void; title: string; body?: ReactNode;
  confirmLabel?: string; danger?: boolean; loading?: boolean; noteLabel?: string; noteRequired?: boolean;
  typeToConfirm?: string;
}) {
  const [note, setNote] = useState('');
  const [typed, setTyped] = useState('');
  useEffect(() => {
    if (open) { setNote(''); setTyped(''); }
  }, [open]);
  const blocked = (noteRequired && !note.trim()) || (typeToConfirm !== undefined && typed !== typeToConfirm);
  return (
    <Modal open={open} onClose={onClose} title={title} size="sm"
      footer={<>
        <Button variant="outline" onClick={onClose}>Cancel</Button>
        <Button variant={danger ? 'danger' : 'primary'} loading={loading} disabled={blocked} onClick={() => onConfirm(note.trim())}>
          {confirmLabel}
        </Button>
      </>}>
      <div className="space-y-4 text-sm text-ink-2">
        {body}
        {noteLabel && (
          <Field label={noteLabel + (noteRequired ? '' : ' (optional)')} htmlFor="confirm-note">
            <Textarea id="confirm-note" value={note} onChange={(e) => setNote(e.target.value)} maxLength={1000} />
          </Field>
        )}
        {typeToConfirm !== undefined && (
          <Field label={<>Type <span className="font-mono font-semibold text-ink">{typeToConfirm}</span> to confirm</>} htmlFor="confirm-type">
            <Input id="confirm-type" value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" />
          </Field>
        )}
      </div>
    </Modal>
  );
}
