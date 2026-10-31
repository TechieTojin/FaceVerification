import { QueryClient, useQuery, type UseQueryOptions } from '@tanstack/react-query';
import { api } from './api';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 15_000,
      refetchOnWindowFocus: false,
      retry: (count, err) => {
        const status = (err as { response?: { status?: number } })?.response?.status;
        if (status && status >= 400 && status < 500) return false;
        return count < 2;
      },
    },
  },
});

export type Params = Record<string, string | number | boolean | undefined | null>;

function clean(p?: Params): Record<string, string | number | boolean> | undefined {
  if (!p) return undefined;
  const out: Record<string, string | number | boolean> = {};
  for (const [k, v] of Object.entries(p)) if (v !== undefined && v !== null && v !== '') out[k] = v;
  return out;
}

export function useApi<T>(path: string | null, params?: Params,
  opts?: Omit<UseQueryOptions<T>, 'queryKey' | 'queryFn'>) {
  return useQuery<T>({
    queryKey: [path, clean(params)],
    queryFn: async ({ signal }) => (await api.get<T>(path as string, { params: clean(params), signal })).data,
    enabled: path !== null && (opts?.enabled ?? true),
    ...opts,
  });
}
