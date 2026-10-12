import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../client';
import type { DashboardSummary, VerificationSession, FrameResult, EnrolmentStatus } from '../../types';

// Auth
export const useLogin = () => {
  const mutation = useMutation({
    mutationFn: (data: { email: string; password: string }) =>
      api.post('/auth/login', data).then(r => r.data),
  });
  return mutation;
};

export const useMe = () => {
  return useQuery({
    queryKey: ['me'],
    queryFn: () => api.get('/auth/me').then(r => r.data),
  });
};

// Users
export const useUsers = (search?: string) => {
  return useQuery({
    queryKey: ['users', search],
    queryFn: () => api.get('/users', { params: { search } }).then(r => r.data),
  });
};

export const useCreateUser = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: any) => api.post('/users', data).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['users'] }),
  });
};

export const useUpdateUser = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...data }: any) => api.put(`/users/${id}`, data).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['users'] }),
  });
};

export const useDisableUser = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.post(`/users/${id}/disable`).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['users'] }),
  });
};

// Dashboard
export const useDashboard = () => {
  return useQuery<DashboardSummary>({
    queryKey: ['dashboard'],
    queryFn: () => api.get('/dashboard/summary').then(r => r.data),
    refetchInterval: 30000,
  });
};

// Analytics
export const useAnalytics = () => {
  return useQuery({
    queryKey: ['analytics'],
    queryFn: () => api.get('/analytics').then(r => r.data),
  });
};

// Enrolment
export const useEnrolmentStatus = (userId: string | null) => {
  return useQuery<EnrolmentStatus>({
    queryKey: ['enrolment', userId],
    queryFn: () => api.get(`/enrolments/${userId}`).then(r => r.data),
    enabled: !!userId,
  });
};

export const useRecordConsent = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, ...data }: any) =>
      api.post(`/enrolments/${userId}/consent`, data).then(r => r.data),
    onSuccess: (_d: any, { userId }: any) => qc.invalidateQueries({ queryKey: ['enrolment', userId] }),
  });
};

export const useEnrolUser = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, file }: { userId: string; file: File }) => {
      const form = new FormData();
      form.append('file', file);
      return api.post(`/enrolments/${userId}/enrol`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }).then(r => r.data);
    },
    onSuccess: (_d: any, { userId }: any) => qc.invalidateQueries({ queryKey: ['enrolment', userId] }),
  });
};

export const useRevokeEnrolment = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (userId: string) => api.delete(`/enrolments/${userId}`).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['enrolment'] }),
  });
};

// Verification
export const useCreateSession = () => {
  return useMutation<VerificationSession, Error, { target_user_id: string }>({
    mutationFn: (data) => api.post('/verification/sessions', data).then(r => r.data),
  });
};

export const useSubmitFrame = () => {
  return useMutation<FrameResult, Error, { sessionId: string; file: Blob }>({
    mutationFn: ({ sessionId, file }) => {
      const form = new FormData();
      form.append('file', file, 'frame.jpg');
      return api.post(`/verification/sessions/${sessionId}/submit-frame`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }).then(r => r.data);
    },
  });
};

// History
export const useVerificationEvents = (page = 1, userId?: string, outcome?: string) => {
  return useQuery({
    queryKey: ['events', page, userId, outcome],
    queryFn: () => api.get('/verification/events', { params: { page, page_size: 20, user_id: userId, outcome } }).then(r => r.data),
  });
};

// Security events
export const useSecurityEvents = () => {
  return useQuery({
    queryKey: ['security-events'],
    queryFn: () => api.get('/security/events').then(r => r.data),
  });
};

// Settings
export const useSettings = () => {
  return useQuery({
    queryKey: ['settings'],
    queryFn: () => api.get('/settings').then(r => r.data),
  });
};

export const useUpdateSettings = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Record<string, string>) => api.put('/settings', data).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['settings'] }),
  });
};
