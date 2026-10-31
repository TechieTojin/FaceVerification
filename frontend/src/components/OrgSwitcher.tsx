import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Building2, Check, ChevronsUpDown, Settings2 } from 'lucide-react';
import { api, apiError } from '../lib/api';
import { cn } from '../lib/cn';
import type { Me } from '../lib/types';
import { useAuth } from '../store/auth';
import { toast } from '../store/toast';

/** Switches the active organisation. The server re-validates membership and re-issues the session token. */
export function OrgSwitcher() {
  const user = useAuth((s) => s.user);
  const setUser = useAuth((s) => s.setUser);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);
  if (!user?.organization) return null;

  const choose = async (id: string) => {
    setOpen(false);
    if (id === user.organization?.id) return;
    try {
      const r = await api.post<{ user: Me }>('/auth/switch-org', { organization_id: id });
      qc.clear();
      setUser(r.data.user);
      navigate('/', { replace: true });
      toast.success(`Switched to ${r.data.user.organization?.name}`);
    } catch (err) {
      toast.error('Could not switch organisation', apiError(err).message);
    }
  };

  return (
    <div className="relative min-w-0" ref={ref}>
      <button className="flex min-w-0 max-w-[140px] items-center gap-2 rounded-lg border border-line px-2.5 py-1.5 text-sm hover:bg-surface-2 sm:max-w-[220px]"
        aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)} data-testid="org-switcher">
        <Building2 className="h-4 w-4 shrink-0 text-muted" />
        <span className="truncate font-medium text-ink">{user.organization.name}</span>
        <ChevronsUpDown className="h-3.5 w-3.5 shrink-0 text-muted" />
      </button>
      {open && (
        <div className="absolute left-0 z-30 mt-1 w-64 animate-fade-in rounded-xl bg-surface p-1 shadow-pop" role="listbox"
          aria-label="Organisations">
          <p className="px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted">Your organisations</p>
          {user.organizations.map((o) => (
            <button key={o.id} role="option" aria-selected={o.id === user.organization?.id} onClick={() => void choose(o.id)}
              className={cn('flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm hover:bg-surface-2',
                o.id === user.organization?.id ? 'text-ink' : 'text-ink-2')}>
              <span className="min-w-0 flex-1 truncate">{o.name}</span>
              <span className="text-[11px] text-muted">{o.role}</span>
              {o.id === user.organization?.id && <Check className="h-4 w-4 text-primary" />}
            </button>
          ))}
          <button className="mt-1 flex w-full items-center gap-2 rounded-lg border-t border-line px-3 py-2 text-sm text-ink-2 hover:bg-surface-2"
            onClick={() => { setOpen(false); navigate('/organization'); }}>
            <Settings2 className="h-4 w-4" /> Organisation settings
          </button>
        </div>
      )}
    </div>
  );
}
