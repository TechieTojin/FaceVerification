/**
 * Auth store — v2.1
 *
 * JWT is stored in an HttpOnly cookie set by the backend, NOT in localStorage.
 * This store only holds the user profile (non-sensitive) and auth status.
 *
 * On page reload, App.tsx calls /auth/me with withCredentials=true.
 * The browser automatically sends the fv_access cookie; if valid, user is
 * restored. If not, the frontend redirects to /login.
 */
import { create } from 'zustand';
import type { UserInfo } from '../types';

interface AuthState {
  user: UserInfo | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  setUser: (user: UserInfo) => void;
  setLoading: (loading: boolean) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,   // true until the initial /me check completes

  setUser: (user) => set({ user, isAuthenticated: true, isLoading: false }),
  setLoading: (isLoading) => set({ isLoading }),
  logout: () => set({ user: null, isAuthenticated: false, isLoading: false }),
}));
