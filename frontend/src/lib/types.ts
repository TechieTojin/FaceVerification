export type Role = 'ADMIN' | 'OPERATOR' | 'VIEWER';

export type Permission =
  | 'dashboard:view' | 'users:directory' | 'users:read' | 'users:manage'
  | 'enrolment:capture' | 'enrolment:approve' | 'enrolment:revoke' | 'enrolment:read'
  | 'verification:run' | 'history:read_own' | 'history:read_all' | 'history:scores' | 'history:export'
  | 'security:read' | 'alerts:manage' | 'analytics:read' | 'evaluation:run'
  | 'policy:read' | 'policy:manage' | 'system:health' | 'document:run'
  | 'org:read' | 'org:manage' | 'review:read' | 'review:decide' | 'cameras:read' | 'cameras:manage'
  | 'devices:read' | 'devices:manage' | 'integrations:manage';

export interface OrgRef { id: string; name: string; slug: string; role?: Role }

export interface Me {
  id: string;
  email: string;
  full_name: string;
  role: Role | null;
  is_platform_admin: boolean;
  is_active: boolean;
  created_at: string;
  last_login_at: string | null;
  password_changed_at: string | null;
  permissions: Permission[];
  organization: OrgRef | null;
  organizations: OrgRef[];
  amr: string[];
}

export type EnrolmentStatusCode = 'NOT_ENROLLED' | 'ACTIVE' | 'REVOKED' | 'WITHDRAWN' | 'REENROLMENT_REQUIRED';

export interface UserRow {
  id: string;
  email: string;
  full_name: string;
  role: Role;
  is_active: boolean;
  enrolment_status: EnrolmentStatusCode;
  enrolment_usable: boolean;
  reenrolment_requested: boolean;
  created_at?: string;
  updated_at?: string;
  last_login_at?: string | null;
  disabled_at?: string | null;
  consent_given?: boolean;
  /** v4: the account belongs only to this organisation, so account-wide actions are allowed */
  managed_here?: boolean;
  membership_status?: string;
}

export interface Paged {
  total: number;
  page: number;
  page_size: number;
}

export interface UserList extends Paged { users: UserRow[] }

export interface EnrolmentView {
  status: EnrolmentStatusCode;
  is_enrolled: boolean;
  usable: boolean;
  consent_given: boolean;
  consent_timestamp?: string | null;
  enrolled_at?: string | null;
  template_version?: number;
  model_name?: string | null;
  sample_count?: number;
  quality_score?: number;
  consistency_score?: number | null;
  reenrolment_requested: boolean;
  revoked_at?: string | null;
}

export interface UserDetail extends UserRow {
  enrolment: EnrolmentView;
  verification_count: number;
  pending_requests: number;
}

export interface EnrolmentDetail extends EnrolmentView {
  user_id: string;
  user_name: string;
  user_email: string;
  user_role: Role;
  consent_history: { granted: boolean; recorded_at: string; version: string | null; purpose: string; method: string | null }[];
}

export interface EnrolmentRequest {
  id: string;
  user_id: string;
  user_name: string | null;
  user_email: string | null;
  kind: 'INITIAL' | 'REENROL';
  status: 'PENDING' | 'APPROVED' | 'REJECTED' | 'AUTO_APPROVED' | 'CANCELLED';
  approval_reasons: string[];
  sample_count: number;
  quality_score: number;
  consistency_score: number | null;
  model_name: string;
  requested_by: string | null;
  requested_by_user_id: string | null;
  created_at: string;
  reviewed_by: string | null;
  reviewed_at: string | null;
  review_note: string | null;
  session_id: string | null;
  duplicate_user_ids?: string[];
}

export type StepType = 'LOOK_STRAIGHT' | 'TURN_LEFT' | 'TURN_RIGHT' | 'BLINK';

export type FrameOutcome =
  | 'IN_PROGRESS' | 'RETRY_REQUIRED' | 'NO_FACE' | 'MULTIPLE_FACES' | 'QUALITY_FAILED'
  | 'VERIFIED' | 'NOT_MATCHED' | 'LIVENESS_FAILED' | 'SESSION_EXPIRED' | 'SESSION_CANCELLED'
  | 'ATTEMPTS_EXHAUSTED' | 'ENROLMENT_CAPTURED' | 'ERROR';

export interface CaptureConfig {
  interval_ms: number;
  max_width: number;
  jpeg_quality: number;
  max_frame_bytes: number;
  min_face_width_ratio: number;
  max_face_width_ratio: number;
  yaw_threshold_degrees: number;
  blink_threshold: number;
}

export interface ChallengeView {
  round: number;
  index: number;
  total: number;
  type: StepType;
  instruction: string;
  steps: StepType[];
  progress: number;
}

export interface FrameView {
  seq: number;
  outcome: FrameOutcome;
  reason: string | null;
  message: string;
  guidance: string[];
  terminal: boolean;
  new_round: boolean;
  face_count: number;
  face_box: [number, number, number, number] | null;
  pose: { yaw: number; pitch: number } | null;
  quality: { score: number; issues: string[]; components: Record<string, number> } | null;
  processing_ms: number;
  timings_ms: Record<string, number>;
  distance?: number | null;
  passive?: { label: 'REAL' | 'SPOOF'; confidence: number } | null;
  request_id?: string;
  request_status?: EnrolmentRequest['status'];
  activated?: boolean;
  approval_reasons?: string[];
}

export interface SessionView {
  session_id: string;
  purpose: 'VERIFY' | 'ENROL' | 'DOCUMENT';
  target_user_id: string | null;
  document_check_id: string | null;
  status: 'ACTIVE' | 'CLOSED';
  final_outcome: FrameOutcome | null;
  reason_code: string | null;
  created_at: string;
  expires_at: string;
  completed_at: string | null;
  frame_seq: number;
  frames_processed: number;
  frames_remaining: number;
  attempts_used: number;
  attempts_remaining: number;
  policy_version: number | null;
  capture: CaptureConfig;
  challenge: ChallengeView | null;
  frame: FrameView | null;
  frame_token?: string;
  final_distance?: number | null;
  threshold?: number | null;
}

export interface HistoryRow {
  session_id: string;
  purpose: SessionView['purpose'];
  created_at: string;
  completed_at: string | null;
  status: 'ACTIVE' | 'CLOSED';
  outcome: FrameOutcome | null;
  reason_code: string | null;
  target_user_id: string | null;
  target_name: string | null;
  target_email: string | null;
  operator_user_id: string | null;
  operator_name: string | null;
  operator_email: string | null;
  frames: number;
  attempts_used: number;
  rounds: number;
  processing_ms: number;
  policy_version: number | null;
  duration_s: number | null;
  liveness_outcome: 'PASSED' | 'FAILED' | null;
  final_distance?: number | null;
  threshold?: number | null;
}

export interface HistoryDetail extends HistoryRow {
  expires_at: string;
  challenges: { round: number; index: number; type: StepType; status: string; started_at: string | null;
    completed_at: string | null; evidence: Record<string, unknown> | null }[];
  events: { timestamp: string; seq: number | null; outcome: FrameOutcome; reason_code: string | null;
    step_type: StepType | null; face_count: number; quality_score: number | null; processing_ms: number | null;
    explanation: string | null; policy_version: string | null; distance?: number | null; threshold?: number | null;
    passive_label?: string | null; passive_confidence?: number | null; yaw?: number | null; pitch?: number | null }[];
  policy_snapshot?: Record<string, unknown>;
}

export interface Percentiles { p50: number | null; p95: number | null; p99: number | null; avg: number | null; samples: number }

export interface HealthCheck { ok: boolean; [k: string]: unknown }

export interface DashboardSummary {
  generated_at: string;
  active_sessions: Record<string, number>;
  active_sessions_total: number;
  window_hours: number;
  completed_verifications: number;
  verified: number;
  not_matched: number;
  liveness_failures: number;
  other_outcomes: Record<string, number>;
  open_alerts: Record<string, number>;
  open_alerts_total: number;
  suspicious_events_24h: number;
  latency_last_hour_ms: Percentiles;
  throughput_last_hour: { sessions: number; frames: number; frames_per_minute: number };
  health: { ready: boolean; database: { ok: boolean; latency_ms: number; error: string | null };
    models: { ok: boolean; state: string }; migrations_up_to_date: boolean };
  recent_sessions: { session_id: string; purpose: string; outcome: FrameOutcome | null; reason_code: string | null;
    completed_at: string | null; processing_ms: number; target_name: string | null; operator_name: string | null }[];
}

export interface ModelStatus { name: string; task: string; status: string; load_ms?: number; error?: string; embedding_dim?: number; sha256?: string; file_name?: string }

export interface AnalyticsOverview {
  days: number;
  totals: { sessions: number; closed: number; decided: number; frames: number };
  session_completion_rate: number | null;
  verified_rate: number | null;
  liveness_rejection_rate: number | null;
  quality_failure_rate: number | null;
  frame_error_rate: number | null;
  outcome_distribution: Record<string, number>;
  frame_outcome_distribution: Record<string, number>;
  latency_ms: Percentiles;
  daily: { date: string; sessions: number; verified: number; not_matched: number; liveness_failed: number;
    frame_errors: number; latency_p50: number | null; latency_p95: number | null; frames: number }[];
  throughput_24h: { hour: string; sessions: number }[];
  notes: string[];
  models: ModelStatus[];
  runtime_latency_ms: Percentiles;
}

export interface Alert {
  id: string;
  rule: string;
  title: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  status: 'OPEN' | 'ACKNOWLEDGED' | 'RESOLVED';
  subject_type: string | null;
  subject_id: string | null;
  subject_label: string | null;
  description: string | null;
  occurrence_count: number;
  first_seen_at: string;
  last_seen_at: string;
  acknowledged_by: string | null;
  acknowledged_at: string | null;
  resolved_by: string | null;
  resolved_at: string | null;
  resolution_note: string | null;
}

export interface SecurityEventRow {
  id: string;
  event_type: string;
  description: string | null;
  severity: string;
  timestamp: string;
  user_id: string | null;
  user_email: string | null;
  ip_address: string | null;
  metadata: Record<string, unknown> | null;
}

export interface AuditEntry {
  id: string;
  timestamp: string;
  action: string;
  actor: string | null;
  target_type: string | null;
  target_id: string | null;
  ip_address?: string | null;
  details: Record<string, unknown> | null;
}

export interface PolicySpec {
  key: string;
  kind: 'float' | 'int' | 'bool' | 'choice';
  default: number | boolean | string;
  group: string;
  label: string;
  description: string;
  min: number | null;
  max: number | null;
  choices: string[];
  unit: string;
  sensitive: boolean;
  weakening: 'increase' | 'decrease' | 'disable' | null;
}

export type PolicyValue = number | boolean | string;

export interface PolicyResponse {
  version: number;
  updated_at: string | null;
  values: Record<string, PolicyValue>;
  specs: PolicySpec[];
  active_sessions_note: string;
}

export interface PolicyVersionRow { version: number; created_at: string; created_by: string | null; change_summary: string | null; is_sensitive_change: boolean }

export interface DocumentCheck {
  id: string;
  status: 'AWAITING_DOCUMENT' | 'AWAITING_LIVE_CAPTURE' | 'COMPLETED' | 'FAILED' | 'EXPIRED' | 'DELETED';
  subject_label: string | null;
  target_user_id: string | null;
  document_type: string;
  created_at: string;
  completed_at: string | null;
  consent_given: boolean;
  document_quality_score: number | null;
  document_quality_issues: string[];
  portrait_detected: boolean;
  portrait_template_stored: boolean;
  face_match_result: 'MATCH' | 'NO_MATCH' | 'NOT_EVALUATED';
  face_match_distance: number | null;
  threshold: number | null;
  liveness_passed: boolean | null;
  authenticity_status: 'PASSED' | 'FAILED' | 'NOT_PERFORMED' | 'ERROR';
  authenticity_provider: string | null;
  authenticity_detail: string | null;
  overall_decision: string;
  session_id: string | null;
  retention_expires_at: string;
  purged_at: string | null;
}

export interface EvaluationResults {
  genuine_pairs: number;
  impostor_pairs: number;
  eer: number;
  eer_threshold: number;
  at_policy_threshold: { threshold: number; far: number; frr: number };
  operating_points: { threshold: number; far: number; frr: number }[];
  genuine_distance: { mean: number; p95: number; max: number };
  impostor_distance: { mean: number; p5: number; min: number };
  curve: { threshold: number; far: number; frr: number }[];
  warnings: string[];
}

export interface EvaluationRun {
  id: string;
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  model_name: string;
  dataset_label: string | null;
  created_at: string;
  completed_at: string | null;
  identity_count: number;
  image_count: number;
  skipped_images: number;
  genuine_pairs: number;
  impostor_pairs: number;
  error: string | null;
  results: EvaluationResults | null;
}

export interface SystemHealth {
  ready: boolean;
  checks: Record<string, HealthCheck>;
  checked_at: string;
  version: string;
  environment: string;
  uptime_seconds: number;
  python: string;
  platform: string;
  database_backend: string;
  inference: {
    provider: string | null; warmup_state: string; ready: boolean; workers: number; capacity: number;
    in_flight: number; completed: number; rejected: number; failed: number; timeouts: number;
    latency_ms: Percentiles; models: ModelStatus[];
  };
  realtime: { sse_subscribers: number };
  maintenance: { last_run_at: string | null; ok: boolean | null; result: Record<string, number> | null; error: string | null };
}

export interface SignInSession { family_id: string; last_rotated_at: string; expires_at: string; user_agent: string | null; ip_address: string | null; current: boolean }

// ── v4 ──────────────────────────────────────────────────────────────────────

export type MfaMethod = 'totp' | 'webauthn' | 'recovery_code' | 'email_otp';

export type LoginResponse =
  | { status: 'ok'; user: Me; recovery_codes?: string[] }
  | { status: 'mfa_required'; pending_token: string; methods: MfaMethod[]; enrol_required: boolean }
  | { status: 'face_required'; pending_token: string; user_name: string };

export interface PasskeyRow { id: string; name: string; created_at: string; last_used_at: string | null;
  device_type: string | null; backed_up: boolean; transports: string[] }
export interface FactorRow { id: string; kind: 'TOTP' | 'EMAIL_OTP'; label: string; created_at: string;
  confirmed_at: string | null; last_used_at: string | null }
export interface SecurityOverview {
  passkeys: PasskeyRow[]; factors: FactorRow[]; recovery_codes_remaining: number; email_otp_available: boolean;
  mfa_required_by_policy: boolean; second_factor_enabled: boolean; step_up_valid: boolean;
  webauthn: { rp_id: string; origins: string[] };
}

export interface Organization extends OrgRef { status: 'ACTIVE' | 'SUSPENDED'; created_at: string; members: number;
  active_members: number; sessions: number; devices: number }
export interface Invitation { id: string; email: string; role: Role; status: string; created_at: string;
  expires_at: string; accepted_at: string | null; link?: string; email_status?: string }

export interface NotificationRow { id: string; type: string; severity: string; title: string; body: string | null;
  link: string | null; created_at: string; read: boolean; email_status: string }
export interface NotificationList extends Paged { unread: number; notifications: NotificationRow[] }
export interface NotificationPref { label: string; in_app: boolean; email: boolean; mandatory: boolean }

export type PairingStatus = 'WAITING' | 'AWAITING_CONFIRMATION' | 'READY' | 'IN_PROGRESS' | 'COMPLETED' | 'EXPIRED' | 'CANCELLED';
export interface Pairing { id: string; kind: 'QR' | 'HOSTED'; status: PairingStatus; created_at: string; expires_at: string;
  claimed_at: string | null; confirmed_at: string | null; completed_at: string | null; target_user_id: string;
  target_name: string | null; camera_id: string | null; session_id: string | null; sessions_started: number;
  result_outcome: string | null; cancelled_reason: string | null; device: string | null; qr_url?: string;
  session?: { status: string; final_outcome: string | null; reason_code: string | null; frames: number; review_status: string | null } }
export interface PhoneState { status: PairingStatus; kind: 'QR' | 'HOSTED'; organization: string | null; subject: string;
  expires_at: string; result_outcome: string | null; session_id: string | null; attempts_used: number;
  cancelled_reason: string | null; session_status?: string; session_outcome?: string | null; review_status?: string | null }
export interface LinkInfo { kind: 'QR' | 'HOSTED'; organization: string | null; requested_by: string | null; subject: string;
  expires_at: string; needs_operator_confirmation: boolean }

export interface Camera { id: string; label: string; location: string | null; hardware_label: string | null;
  capabilities: Record<string, unknown> | null; status: 'ACTIVE' | 'DISABLED';
  connection: 'ONLINE' | 'OFFLINE' | 'NEVER_SEEN' | 'DISABLED'; device_id: string | null; last_heartbeat_at: string | null;
  last_heartbeat_by: string | null; last_verification_at: string | null; created_at: string; disabled_at: string | null;
  activity?: ActivityRow[]; recent_sessions?: { session_id: string; created_at: string; outcome: string | null;
    purpose: string; channel: string }[] }
export interface ActivityRow { at: string; action: string; actor: string | null; details: Record<string, unknown> | null }

export interface Device { id: string; name: string; kind: 'KIOSK' | 'BROWSER_TERMINAL'; status: 'PENDING' | 'ACTIVE' | 'REVOKED';
  connection: string; location: string | null; software_version: string | null; platform: string | null;
  key_fingerprint: string | null; key_protection: string | null; hardware_attested: boolean; last_seen_at: string | null;
  created_at: string; activated_at: string | null; revoked_at: string | null; revoke_reason: string | null;
  offline_enabled: boolean; offline_validity_hours: number; health: Record<string, unknown> | null;
  provisioning_expires_at: string | null; provisioning_code?: string;
  cameras?: Camera[]; offline_subjects?: { user_id: string; full_name: string; added_at: string }[];
  latest_bundle?: { id: string; issued_at: string; valid_until: string; subject_count: number; status: string } | null;
  offline_events?: number; activity?: ActivityRow[] }
export interface OfflineEventRow { event_id: string; seq: number; type: string; occurred_at: string; received_at: string;
  subject: string | null; outcome: string | null; reason_code: string | null; integrity: string; bundle_id: string | null }

export interface Review { id: string; status: string; risk_level: 'LOW' | 'MEDIUM' | 'HIGH'; reasons: string[];
  reason_text: string[]; technical_outcome: string | null; spoof_confirmed: boolean; can_approve: boolean;
  session_id: string | null; document_check_id: string | null; subject_user_id: string | null; subject_name: string | null;
  assigned_to_user_id: string | null; assigned_to: string | null; decided_by: string | null; decided_at: string | null;
  decision_note: string | null; created_at: string; expires_at: string; version: number; channel: string | null;
  operator_user_id: string | null; operator_name: string | null;
  history?: { action: string; note: string | null; at: string; actor: string | null }[];
  session?: { purpose: string; final_outcome: string | null; reason_code: string | null; final_distance: number | null;
    threshold: number | null; created_at: string; completed_at: string | null; frames: number; attempts_used: number;
    policy_version: number | null } }

export interface ApiClient { id: string; name: string; description: string | null; scopes: string[]; status: string;
  rate_limit_per_minute: number; created_at: string; revoked_at: string | null; active_credentials: number;
  api_key?: string;
  credentials?: { id: string; key_id: string; prefix: string; created_at: string; expires_at: string | null;
    revoked_at: string | null; last_used_at: string | null; state: string }[];
  webhooks?: { id: string; url: string; events: string[]; status: string; created_at: string; secret_rotated_at: string | null }[];
  deliveries?: { id: string; endpoint_id: string; event_id: string; event_type: string; status: string; attempts: number;
    last_status_code: number | null; last_error: string | null; created_at: string; delivered_at: string | null;
    next_attempt_at: string | null }[] }

export interface KioskStatus { device_id: string; name: string; organization: string; server_url: string; online: boolean;
  revoked: boolean; mode: 'ONLINE' | 'OFFLINE'; software_version: string; key_protection: string; hardware_backed: boolean;
  inference: string; inference_location: string; last_sync_at: string | null; last_sync_error: string | null;
  events: { events: number; pending: number };
  bundle: { provisioned: boolean; valid: boolean; reason: string | null; bundle_id?: string; valid_until?: string;
    issued_at?: string; subject_count?: number } }
