import { create } from 'zustand';
import type { Me, Permission } from '../lib/types';

type Status = 'unknown' | 'authenticated' | 'anonymous';

interface AuthState {
  user: Me | null;
  status: Status;
  setUser: (u: Me) => void;
  clear: () => void;
}

/** Holds only the non-sensitive profile; credentials live in HttpOnly cookies. */
export const useAuth = create<AuthState>()((set) => ({
  user: null,
  status: 'unknown',
  setUser: (user) => set({ user, status: 'authenticated' }),
  clear: () => set({ user: null, status: 'anonymous' }),
}));

export function useCan(): (p: Permission) => boolean {
  const user = useAuth((s) => s.user);
  return (p: Permission) => !!user?.permissions.includes(p);
}
