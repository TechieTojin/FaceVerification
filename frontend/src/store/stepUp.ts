import { create } from 'zustand';

export type StepUpLevel = 'standard' | 'face_and_factor';

interface StepUpState {
  open: boolean;
  level: StepUpLevel;
  resolve: ((ok: boolean) => void) | null;
  request: (level: StepUpLevel) => Promise<boolean>;
  finish: (ok: boolean) => void;
}

/** One pending step-up at a time; concurrent requests share it. */
export const useStepUp = create<StepUpState>()((set, get) => ({
  open: false,
  level: 'standard',
  resolve: null,
  request: (level) => {
    const current = get();
    if (current.open && current.resolve) {
      return new Promise<boolean>((res) => {
        const prev = current.resolve!;
        set({ resolve: (ok) => { prev(ok); res(ok); } });
      });
    }
    return new Promise<boolean>((resolve) => set({ open: true, level, resolve }));
  },
  finish: (ok) => {
    get().resolve?.(ok);
    set({ open: false, resolve: null, level: 'standard' });
  },
}));
