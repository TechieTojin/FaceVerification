import type { HTMLAttributes, ReactNode, TdHTMLAttributes, ThHTMLAttributes } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '../../lib/cn';
import { Button } from './Button';

export function Table({ children, className, label }: { children: ReactNode; className?: string; label: string }) {
  return (
    <div className={cn('w-full overflow-x-auto', className)}>
      <table className="w-full min-w-[640px] border-collapse text-sm" aria-label={label}>{children}</table>
    </div>
  );
}

export function THead({ children }: { children: ReactNode }) {
  return <thead className="border-b border-line bg-surface-2/60 text-left text-xs font-medium text-muted">{children}</thead>;
}

export function Th({ className, ...rest }: ThHTMLAttributes<HTMLTableCellElement>) {
  return <th scope="col" className={cn('whitespace-nowrap px-4 py-2.5 font-medium', className)} {...rest} />;
}

export function Tr({ className, onClick, ...rest }: HTMLAttributes<HTMLTableRowElement>) {
  return <tr className={cn('border-b border-line last:border-0', onClick && 'cursor-pointer hover:bg-surface-2/60', className)}
    onClick={onClick} {...rest} />;
}

export function Td({ className, ...rest }: TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={cn('px-4 py-3 align-middle text-ink-2', className)} {...rest} />;
}

export function Pagination({ page, pageSize, total, onPage }: {
  page: number; pageSize: number; total: number; onPage: (p: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(total, page * pageSize);
  return (
    <nav className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-3 text-xs text-muted" aria-label="Pagination">
      <span className="tabular">{from}–{to} of {total}</span>
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => onPage(page - 1)} disabled={page <= 1} aria-label="Previous page"
          icon={<ChevronLeft className="h-4 w-4" />} />
        <span className="tabular">Page {page} of {pages}</span>
        <Button variant="outline" size="sm" onClick={() => onPage(page + 1)} disabled={page >= pages} aria-label="Next page"
          icon={<ChevronRight className="h-4 w-4" />} />
      </div>
    </nav>
  );
}
