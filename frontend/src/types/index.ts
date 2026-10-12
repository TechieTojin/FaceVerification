export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'ADMIN' | 'OPERATOR' | 'VIEWER';
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_login_at?: string;
  has_enrolment: boolean;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserInfo {
  id: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export type OutcomeType = 'VERIFIED' | 'REJECTED' | 'NO_FACE' | 'MULTIPLE_FACES' | 'SPOOF_DETECTED' | 'ERROR' | 'TIMEOUT' | 'LOCKOUT';

export interface VerificationSession {
  session_id: string;
  target_user_id: string;
  expires_at: string;
  is_active: boolean;
  frame_count: number;
}

export interface FrameResult {
  session_id: string;
  outcome: OutcomeType;
  action: string;
  explanation: string;
  distance?: number;
  threshold?: number;
  liveness_score?: number;
  face_count: number;
  inference_ms: number;
  attempts_remaining: number;
  is_liveness_real?: boolean;
}

export interface DashboardSummary {
  total_users: number;
  active_enrolments: number;
  total_verifications: number;
  successful_verifications: number;
  failed_verifications: number;
  spoof_attempts: number;
  success_rate: number;
  avg_latency_ms: number;
  recent_events: any[];
  backend_healthy: boolean;
  models_ready: boolean;
}

export interface VerificationEvent {
  id: string;
  session_id: string;
  user_id?: string;
  timestamp: string;
  outcome: OutcomeType;
  distance?: number;
  threshold?: number;
  liveness_score?: number;
  face_count: number;
  inference_ms?: number;
  explanation?: string;
}

export interface SecurityEvent {
  id: string;
  event_type: string;
  description: string;
  severity: string;
  timestamp: string;
  user_id?: string;
  ip_address?: string;
  metadata?: any;
}

export interface EnrolmentStatus {
  user_id: string;
  is_enrolled: boolean;
  consent_given: boolean;
  quality_score: number;
  enrolled_at?: string;
  model_name?: string;
}
