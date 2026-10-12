import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { UserInfo } from '../types';

interface AuthState {
  token: string | null;
  user: UserInfo | null;
  isAuthenticated: boolean;
  setToken: (token: string) => void;
  setUser: (user: UserInfo) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      isAuthenticated: false,
      setToken: (token) => set({ token, isAuthenticated: true }),
      setUser: (user) => set({ user }),
      logout: () => set({ token: null, user: null, isAuthenticated: false }),
    }),
    {
      name: 'fv-auth',
      partialize: (state) => ({ token: state.token, user: state.user, isAuthenticated: state.isAuthenticated }),
    }
  )
);
