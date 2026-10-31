"""v4.0 platform: organisations (multi-tenancy), passkeys/MFA, cross-device pairing, devices,
cameras, offline kiosks, manual review, notifications and the integration API.

Data handling:
  * A "Default organization" is created and every existing tenant-scoped row
    (sessions, events, enrolments, consent, requests, documents, evaluations,
    policies, alerts, security events, audit entries) is assigned to it.
    Nothing is deleted.
  * Every user gets a membership in the default organisation with their v3
    role (users.role is then dropped). v3 administrators become platform
    administrators, so the upgraded deployment keeps full administrative control.
  * Capture sessions are marked as operator-owned (owner_kind=USER).
  * biometric_enrolments becomes unique per (organisation, user) instead of per user.
  * policy_versions becomes versioned per organisation.

Downgrade restores the v3 schema. It refuses to run when more than one
organisation exists, because rows of other organisations cannot be represented
in v3 without merging tenants. Restore the pre-migration backup instead.

Revision ID: 0003_v4_platform
Revises: 0002_v3_platform
"""
from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision = "0003_v4_platform"
down_revision = "0002_v3_platform"
branch_labels = None
depends_on = None

# (table, nullable) — tables that gain organization_id
TENANT_TABLES = (
    ("verification_sessions", False),
    ("verification_events", False),
    ("biometric_enrolments", False),
    ("consent_records", False),
    ("enrolment_requests", False),
    ("document_checks", False),
    ("evaluation_runs", False),
    ("policy_versions", False),
    ("security_events", True),
    ("security_alerts", True),
    ("audit_logs", True),
)


# <generated-tables>
def _create_v4_tables() -> None:
    op.create_table('signing_keys',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('alg', sa.String(length=10), nullable=False),
    sa.Column('public_pem', sa.Text(), nullable=False),
    sa.Column('private_encrypted', sa.LargeBinary(), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('retired_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('mfa_factors',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('label', sa.String(length=100), nullable=False),
    sa.Column('secret_encrypted', sa.LargeBinary(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('confirmed_at', sa.DateTime(), nullable=True),
    sa.Column('last_used_step', sa.Integer(), nullable=True),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_mfa_factors_user_id'), 'mfa_factors', ['user_id'], unique=False)
    op.create_table('notification_preferences',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('type', sa.String(length=40), nullable=False),
    sa.Column('in_app', sa.Boolean(), nullable=False),
    sa.Column('email', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'type', name='uq_notification_pref')
    )
    op.create_index(op.f('ix_notification_preferences_user_id'), 'notification_preferences', ['user_id'], unique=False)
    op.create_table('organizations',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('slug', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('created_by_user_id', sa.String(length=36), nullable=True),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_organizations_slug'), 'organizations', ['slug'], unique=True)
    op.create_table('recovery_codes',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('batch_id', sa.String(length=36), nullable=False),
    sa.Column('code_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('used_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('code_hash')
    )
    op.create_index(op.f('ix_recovery_codes_batch_id'), 'recovery_codes', ['batch_id'], unique=False)
    op.create_index(op.f('ix_recovery_codes_user_id'), 'recovery_codes', ['user_id'], unique=False)
    op.create_table('webauthn_credentials',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('credential_id', sa.String(length=512), nullable=False),
    sa.Column('public_key', sa.LargeBinary(), nullable=False),
    sa.Column('sign_count', sa.Integer(), nullable=False),
    sa.Column('transports', sa.String(length=200), nullable=True),
    sa.Column('aaguid', sa.String(length=36), nullable=True),
    sa.Column('device_type', sa.String(length=24), nullable=True),
    sa.Column('backed_up', sa.Boolean(), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('credential_id')
    )
    op.create_index(op.f('ix_webauthn_credentials_user_id'), 'webauthn_credentials', ['user_id'], unique=False)
    op.create_table('auth_challenges',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('kind', sa.String(length=24), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=True),
    sa.Column('organization_id', sa.String(length=36), nullable=True),
    sa.Column('challenge', sa.String(length=200), nullable=True),
    sa.Column('token_hash', sa.String(length=64), nullable=True),
    sa.Column('data_json', sa.Text(), nullable=True),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('consumed_at', sa.DateTime(), nullable=True),
    sa.Column('ip_address', sa.String(length=45), nullable=True),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('token_hash')
    )
    op.create_index(op.f('ix_auth_challenges_expires_at'), 'auth_challenges', ['expires_at'], unique=False)
    op.create_index(op.f('ix_auth_challenges_kind'), 'auth_challenges', ['kind'], unique=False)
    op.create_index(op.f('ix_auth_challenges_user_id'), 'auth_challenges', ['user_id'], unique=False)
    op.create_table('devices',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('kind', sa.String(length=20), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('location', sa.String(length=200), nullable=True),
    sa.Column('public_key_pem', sa.Text(), nullable=True),
    sa.Column('key_fingerprint', sa.String(length=64), nullable=True),
    sa.Column('key_protection', sa.String(length=32), nullable=True),
    sa.Column('provisioning_code_hash', sa.String(length=64), nullable=True),
    sa.Column('provisioning_expires_at', sa.DateTime(), nullable=True),
    sa.Column('software_version', sa.String(length=40), nullable=True),
    sa.Column('platform', sa.String(length=80), nullable=True),
    sa.Column('health_json', sa.Text(), nullable=True),
    sa.Column('last_seen_at', sa.DateTime(), nullable=True),
    sa.Column('last_ip', sa.String(length=45), nullable=True),
    sa.Column('offline_enabled', sa.Boolean(), nullable=False),
    sa.Column('offline_validity_hours', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('created_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('activated_at', sa.DateTime(), nullable=True),
    sa.Column('key_rotated_at', sa.DateTime(), nullable=True),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.Column('revoked_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('revoke_reason', sa.String(length=200), nullable=True),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.ForeignKeyConstraint(['revoked_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('provisioning_code_hash')
    )
    op.create_index(op.f('ix_devices_key_fingerprint'), 'devices', ['key_fingerprint'], unique=False)
    op.create_index(op.f('ix_devices_organization_id'), 'devices', ['organization_id'], unique=False)
    op.create_index(op.f('ix_devices_status'), 'devices', ['status'], unique=False)
    op.create_table('integration_clients',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('scopes', sa.String(length=300), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('rate_limit_per_minute', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('created_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_integration_clients_organization_id'), 'integration_clients', ['organization_id'], unique=False)
    op.create_table('memberships',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('role', sa.Enum('ADMIN', 'OPERATOR', 'VIEWER', name='roleenum', native_enum=False, length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('invited_by_user_id', sa.String(length=36), nullable=True),
    sa.ForeignKeyConstraint(['invited_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('organization_id', 'user_id', name='uq_membership_org_user')
    )
    op.create_index(op.f('ix_memberships_organization_id'), 'memberships', ['organization_id'], unique=False)
    op.create_index(op.f('ix_memberships_user_id'), 'memberships', ['user_id'], unique=False)
    op.create_table('notifications',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=True),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('type', sa.String(length=40), nullable=False),
    sa.Column('severity', sa.String(length=8), nullable=False),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('body', sa.Text(), nullable=True),
    sa.Column('link', sa.String(length=300), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('read_at', sa.DateTime(), nullable=True),
    sa.Column('email_status', sa.String(length=16), nullable=False),
    sa.Column('email_attempts', sa.Integer(), nullable=False),
    sa.Column('email_next_attempt_at', sa.DateTime(), nullable=True),
    sa.Column('email_last_error', sa.String(length=300), nullable=True),
    sa.Column('email_sent_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_notifications_created_at'), 'notifications', ['created_at'], unique=False)
    op.create_index(op.f('ix_notifications_email_next_attempt_at'), 'notifications', ['email_next_attempt_at'], unique=False)
    op.create_index(op.f('ix_notifications_organization_id'), 'notifications', ['organization_id'], unique=False)
    op.create_index(op.f('ix_notifications_type'), 'notifications', ['type'], unique=False)
    op.create_index(op.f('ix_notifications_user_id'), 'notifications', ['user_id'], unique=False)
    op.create_table('organization_invitations',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('role', sa.Enum('ADMIN', 'OPERATOR', 'VIEWER', name='roleenum', native_enum=False, length=16), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('created_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('accepted_at', sa.DateTime(), nullable=True),
    sa.Column('accepted_user_id', sa.String(length=36), nullable=True),
    sa.ForeignKeyConstraint(['accepted_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('token_hash')
    )
    op.create_index(op.f('ix_organization_invitations_email'), 'organization_invitations', ['email'], unique=False)
    op.create_index(op.f('ix_organization_invitations_organization_id'), 'organization_invitations', ['organization_id'], unique=False)
    op.create_index(op.f('ix_organization_invitations_status'), 'organization_invitations', ['status'], unique=False)
    op.create_table('api_credentials',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('client_id', sa.String(length=36), nullable=False),
    sa.Column('key_id', sa.String(length=24), nullable=False),
    sa.Column('secret_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('created_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('expires_at', sa.DateTime(), nullable=True),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['client_id'], ['integration_clients.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('key_id')
    )
    op.create_index(op.f('ix_api_credentials_client_id'), 'api_credentials', ['client_id'], unique=False)
    op.create_table('cameras',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('device_id', sa.String(length=36), nullable=True),
    sa.Column('label', sa.String(length=120), nullable=False),
    sa.Column('location', sa.String(length=200), nullable=True),
    sa.Column('hardware_label', sa.String(length=200), nullable=True),
    sa.Column('capabilities_json', sa.Text(), nullable=True),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('created_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('last_heartbeat_at', sa.DateTime(), nullable=True),
    sa.Column('last_heartbeat_by', sa.String(length=64), nullable=True),
    sa.Column('last_verification_at', sa.DateTime(), nullable=True),
    sa.Column('disabled_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['device_id'], ['devices.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cameras_device_id'), 'cameras', ['device_id'], unique=False)
    op.create_index(op.f('ix_cameras_organization_id'), 'cameras', ['organization_id'], unique=False)
    op.create_index(op.f('ix_cameras_status'), 'cameras', ['status'], unique=False)
    op.create_table('device_nonces',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('device_id', sa.String(length=36), nullable=False),
    sa.Column('nonce', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['device_id'], ['devices.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('device_id', 'nonce', name='uq_device_nonce')
    )
    op.create_index(op.f('ix_device_nonces_created_at'), 'device_nonces', ['created_at'], unique=False)
    op.create_table('idempotency_records',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('client_id', sa.String(length=36), nullable=False),
    sa.Column('key', sa.String(length=128), nullable=False),
    sa.Column('request_hash', sa.String(length=64), nullable=False),
    sa.Column('status_code', sa.Integer(), nullable=False),
    sa.Column('response_json', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['client_id'], ['integration_clients.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('client_id', 'key', name='uq_idempotency_client_key')
    )
    op.create_index(op.f('ix_idempotency_records_created_at'), 'idempotency_records', ['created_at'], unique=False)
    op.create_table('offline_bundles',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('device_id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('issued_at', sa.DateTime(), nullable=False),
    sa.Column('valid_until', sa.DateTime(), nullable=False),
    sa.Column('subject_count', sa.Integer(), nullable=False),
    sa.Column('policy_version', sa.Integer(), nullable=True),
    sa.Column('payload_sha256', sa.String(length=64), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.ForeignKeyConstraint(['device_id'], ['devices.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_offline_bundles_device_id'), 'offline_bundles', ['device_id'], unique=False)
    op.create_index(op.f('ix_offline_bundles_organization_id'), 'offline_bundles', ['organization_id'], unique=False)
    op.create_table('offline_events',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('device_id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('event_id', sa.String(length=36), nullable=False),
    sa.Column('seq', sa.Integer(), nullable=False),
    sa.Column('event_type', sa.String(length=32), nullable=False),
    sa.Column('occurred_at', sa.DateTime(), nullable=False),
    sa.Column('received_at', sa.DateTime(), nullable=False),
    sa.Column('bundle_id', sa.String(length=36), nullable=True),
    sa.Column('subject_user_id', sa.String(length=36), nullable=True),
    sa.Column('outcome', sa.String(length=32), nullable=True),
    sa.Column('reason_code', sa.String(length=48), nullable=True),
    sa.Column('payload_json', sa.Text(), nullable=True),
    sa.Column('prev_hash', sa.String(length=64), nullable=True),
    sa.Column('chain_hash', sa.String(length=64), nullable=False),
    sa.Column('integrity', sa.String(length=12), nullable=False),
    sa.ForeignKeyConstraint(['device_id'], ['devices.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.ForeignKeyConstraint(['subject_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('device_id', 'event_id', name='uq_offline_event')
    )
    op.create_index(op.f('ix_offline_events_device_id'), 'offline_events', ['device_id'], unique=False)
    op.create_index(op.f('ix_offline_events_organization_id'), 'offline_events', ['organization_id'], unique=False)
    op.create_table('offline_subjects',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('device_id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('added_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['added_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['device_id'], ['devices.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('device_id', 'user_id', name='uq_offline_subject')
    )
    op.create_index(op.f('ix_offline_subjects_device_id'), 'offline_subjects', ['device_id'], unique=False)
    op.create_table('webhook_endpoints',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('client_id', sa.String(length=36), nullable=False),
    sa.Column('url', sa.String(length=500), nullable=False),
    sa.Column('secret_encrypted', sa.LargeBinary(), nullable=False),
    sa.Column('events', sa.String(length=300), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('secret_rotated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['client_id'], ['integration_clients.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_webhook_endpoints_client_id'), 'webhook_endpoints', ['client_id'], unique=False)
    op.create_index(op.f('ix_webhook_endpoints_organization_id'), 'webhook_endpoints', ['organization_id'], unique=False)
    op.create_table('capture_pairings',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('kind', sa.String(length=8), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('secret_hash', sa.String(length=64), nullable=True),
    sa.Column('confirm_code_hash', sa.String(length=64), nullable=True),
    sa.Column('confirm_attempts', sa.Integer(), nullable=False),
    sa.Column('target_user_id', sa.String(length=36), nullable=False),
    sa.Column('created_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('integration_client_id', sa.String(length=36), nullable=True),
    sa.Column('external_ref', sa.String(length=128), nullable=True),
    sa.Column('camera_id', sa.String(length=36), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('claimed_at', sa.DateTime(), nullable=True),
    sa.Column('claim_ip', sa.String(length=45), nullable=True),
    sa.Column('claim_user_agent', sa.String(length=200), nullable=True),
    sa.Column('confirmed_at', sa.DateTime(), nullable=True),
    sa.Column('session_id', sa.String(length=36), nullable=True),
    sa.Column('sessions_started', sa.Integer(), nullable=False),
    sa.Column('result_outcome', sa.String(length=32), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('cancelled_reason', sa.String(length=48), nullable=True),
    sa.Column('metadata_json', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ),
    sa.ForeignKeyConstraint(['integration_client_id'], ['integration_clients.id'], ),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.ForeignKeyConstraint(['target_user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('secret_hash'),
    sa.UniqueConstraint('token_hash')
    )
    op.create_index(op.f('ix_capture_pairings_created_by_user_id'), 'capture_pairings', ['created_by_user_id'], unique=False)
    op.create_index(op.f('ix_capture_pairings_expires_at'), 'capture_pairings', ['expires_at'], unique=False)
    op.create_index(op.f('ix_capture_pairings_integration_client_id'), 'capture_pairings', ['integration_client_id'], unique=False)
    op.create_index(op.f('ix_capture_pairings_organization_id'), 'capture_pairings', ['organization_id'], unique=False)
    op.create_index(op.f('ix_capture_pairings_session_id'), 'capture_pairings', ['session_id'], unique=False)
    op.create_index(op.f('ix_capture_pairings_status'), 'capture_pairings', ['status'], unique=False)
    op.create_index(op.f('ix_capture_pairings_target_user_id'), 'capture_pairings', ['target_user_id'], unique=False)
    op.create_table('webhook_deliveries',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('endpoint_id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('event_id', sa.String(length=36), nullable=False),
    sa.Column('event_type', sa.String(length=48), nullable=False),
    sa.Column('payload_json', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('next_attempt_at', sa.DateTime(), nullable=True),
    sa.Column('last_status_code', sa.Integer(), nullable=True),
    sa.Column('last_error', sa.String(length=300), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('delivered_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['endpoint_id'], ['webhook_endpoints.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('endpoint_id', 'event_id', name='uq_webhook_delivery_event')
    )
    op.create_index(op.f('ix_webhook_deliveries_endpoint_id'), 'webhook_deliveries', ['endpoint_id'], unique=False)
    op.create_index(op.f('ix_webhook_deliveries_next_attempt_at'), 'webhook_deliveries', ['next_attempt_at'], unique=False)
    op.create_index(op.f('ix_webhook_deliveries_organization_id'), 'webhook_deliveries', ['organization_id'], unique=False)
    op.create_index(op.f('ix_webhook_deliveries_status'), 'webhook_deliveries', ['status'], unique=False)
    op.create_table('verification_assertions',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('client_id', sa.String(length=36), nullable=True),
    sa.Column('session_id', sa.String(length=36), nullable=False),
    sa.Column('subject', sa.String(length=128), nullable=False),
    sa.Column('outcome', sa.String(length=32), nullable=False),
    sa.Column('kid', sa.String(length=36), nullable=False),
    sa.Column('issued_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('consumed_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['client_id'], ['integration_clients.id'], ),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['verification_sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_verification_assertions_client_id'), 'verification_assertions', ['client_id'], unique=False)
    op.create_index(op.f('ix_verification_assertions_organization_id'), 'verification_assertions', ['organization_id'], unique=False)
    op.create_index(op.f('ix_verification_assertions_session_id'), 'verification_assertions', ['session_id'], unique=False)
    op.create_table('verification_reviews',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('organization_id', sa.String(length=36), nullable=False),
    sa.Column('session_id', sa.String(length=36), nullable=True),
    sa.Column('document_check_id', sa.String(length=36), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('risk_level', sa.String(length=8), nullable=False),
    sa.Column('reasons_json', sa.Text(), nullable=False),
    sa.Column('technical_outcome', sa.String(length=32), nullable=True),
    sa.Column('spoof_confirmed', sa.Boolean(), nullable=False),
    sa.Column('subject_user_id', sa.String(length=36), nullable=True),
    sa.Column('assigned_to_user_id', sa.String(length=36), nullable=True),
    sa.Column('assigned_at', sa.DateTime(), nullable=True),
    sa.Column('decided_by_user_id', sa.String(length=36), nullable=True),
    sa.Column('decided_at', sa.DateTime(), nullable=True),
    sa.Column('decision_note', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['assigned_to_user_id'], ['users.id'], ),
    sa.ForeignKeyConstraint(['decided_by_user_id'], ['users.id'], ),
    sa.ForeignKeyConstraint(['document_check_id'], ['document_checks.id'], ),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['verification_sessions.id'], ),
    sa.ForeignKeyConstraint(['subject_user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_id')
    )
    op.create_index(op.f('ix_verification_reviews_assigned_to_user_id'), 'verification_reviews', ['assigned_to_user_id'], unique=False)
    op.create_index(op.f('ix_verification_reviews_created_at'), 'verification_reviews', ['created_at'], unique=False)
    op.create_index(op.f('ix_verification_reviews_organization_id'), 'verification_reviews', ['organization_id'], unique=False)
    op.create_index(op.f('ix_verification_reviews_status'), 'verification_reviews', ['status'], unique=False)
    op.create_index(op.f('ix_verification_reviews_subject_user_id'), 'verification_reviews', ['subject_user_id'], unique=False)
    op.create_table('review_events',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('review_id', sa.String(length=36), nullable=False),
    sa.Column('actor_user_id', sa.String(length=36), nullable=True),
    sa.Column('action', sa.String(length=24), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['actor_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['review_id'], ['verification_reviews.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_review_events_review_id'), 'review_events', ['review_id'], unique=False)


V4_TABLES = ['api_credentials', 'auth_challenges', 'cameras', 'capture_pairings', 'device_nonces', 'devices', 'idempotency_records', 'integration_clients', 'memberships', 'mfa_factors', 'notification_preferences', 'notifications', 'offline_bundles', 'offline_events', 'offline_subjects', 'organization_invitations', 'organizations', 'recovery_codes', 'review_events', 'signing_keys', 'verification_assertions', 'verification_reviews', 'webauthn_credentials', 'webhook_deliveries', 'webhook_endpoints']
# </generated-tables>


def _add_org_column(table: str, nullable: bool, org_id: str) -> None:
    with op.batch_alter_table(table) as b:
        b.add_column(sa.Column("organization_id", sa.String(36), nullable=True))
    op.execute(sa.text(f"UPDATE {table} SET organization_id = :o").bindparams(o=org_id))
    with op.batch_alter_table(table) as b:
        if not nullable:
            b.alter_column("organization_id", existing_type=sa.String(36), nullable=False)
        b.create_foreign_key(f"fk_{table}_organization_id", "organizations", ["organization_id"], ["id"])
        b.create_index(f"ix_{table}_organization_id", ["organization_id"])


def upgrade() -> None:
    bind = op.get_bind()
    _create_v4_tables()

    now = datetime.utcnow()
    org_id = str(uuid.uuid4())
    bind.execute(sa.text("INSERT INTO organizations (id, slug, name, status, created_at) "
                         "VALUES (:id, 'default', 'Default organization', 'ACTIVE', :now)"),
                 {"id": org_id, "now": now})

    users = bind.execute(sa.text("SELECT id, role FROM users")).fetchall()
    for uid, role in users:
        bind.execute(sa.text("INSERT INTO memberships (id, organization_id, user_id, role, status, created_at, updated_at) "
                             "VALUES (:id, :org, :uid, :role, 'ACTIVE', :now, :now)"),
                     {"id": str(uuid.uuid4()), "org": org_id, "uid": uid, "role": role or "OPERATOR", "now": now})

    with op.batch_alter_table("users") as b:
        b.add_column(sa.Column("is_platform_admin", sa.Boolean(), nullable=False, server_default=sa.false()))
        b.add_column(sa.Column("last_org_id", sa.String(36), nullable=True))
    bind.execute(sa.text("UPDATE users SET is_platform_admin = :t, last_org_id = :org WHERE role = 'ADMIN'"),
                 {"t": True, "org": org_id})
    bind.execute(sa.text("UPDATE users SET last_org_id = :org WHERE last_org_id IS NULL"), {"org": org_id})
    with op.batch_alter_table("users") as b:
        b.alter_column("is_platform_admin", server_default=None, existing_type=sa.Boolean(), existing_nullable=False)
        b.drop_column("role")

    for table, nullable in TENANT_TABLES:
        _add_org_column(table, nullable, org_id)

    # Capture-session ownership and cross-device columns.
    with op.batch_alter_table("verification_sessions") as b:
        b.add_column(sa.Column("owner_kind", sa.String(16), nullable=True))
        b.add_column(sa.Column("owner_ref", sa.String(36), nullable=True))
        b.add_column(sa.Column("channel", sa.String(16), nullable=True))
        b.add_column(sa.Column("camera_id", sa.String(36), nullable=True))
        b.add_column(sa.Column("pairing_id", sa.String(36), nullable=True))
        b.add_column(sa.Column("review_status", sa.String(20), nullable=True))
        b.add_column(sa.Column("risk_json", sa.Text(), nullable=True))
    bind.execute(sa.text("UPDATE verification_sessions SET owner_kind = 'USER', channel = 'DESKTOP', "
                         "owner_ref = COALESCE(created_by_user_id, '')"))
    with op.batch_alter_table("verification_sessions") as b:
        b.alter_column("owner_kind", existing_type=sa.String(16), nullable=False)
        b.alter_column("owner_ref", existing_type=sa.String(36), nullable=False)
        b.alter_column("channel", existing_type=sa.String(16), nullable=False)
        b.create_foreign_key("fk_verification_sessions_camera_id", "cameras", ["camera_id"], ["id"], ondelete="SET NULL")
        b.create_index("ix_verification_sessions_owner_ref", ["owner_ref"])
        b.create_index("ix_verification_sessions_camera_id", ["camera_id"])
        b.create_index("ix_verification_sessions_pairing_id", ["pairing_id"])
        b.create_index("ix_verification_sessions_review_status", ["review_status"])

    # Enrolments: unique per (organisation, user).
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("biometric_enrolments",
                                  naming_convention={"uq": "uq_%(table_name)s_%(column_0_name)s"}) as b:
            b.drop_constraint("uq_biometric_enrolments_user_id", type_="unique")
    else:
        op.drop_constraint("biometric_enrolments_user_id_key", "biometric_enrolments", type_="unique")
    with op.batch_alter_table("biometric_enrolments") as b:
        b.create_unique_constraint("uq_enrolment_org_user", ["organization_id", "user_id"])
        b.create_index("ix_biometric_enrolments_user_id", ["user_id"])

    # Policies: versioned per organisation.
    op.drop_index("ix_policy_versions_version", table_name="policy_versions")
    with op.batch_alter_table("policy_versions") as b:
        b.create_index("ix_policy_versions_version", ["version"])
        b.create_unique_constraint("uq_policy_org_version", ["organization_id", "version"])

    with op.batch_alter_table("refresh_tokens") as b:
        b.add_column(sa.Column("organization_id", sa.String(36), nullable=True))
        b.add_column(sa.Column("amr", sa.String(64), nullable=True))
        b.create_foreign_key("fk_refresh_tokens_organization_id", "organizations", ["organization_id"], ["id"],
                             ondelete="CASCADE")
    bind.execute(sa.text("UPDATE refresh_tokens SET organization_id = :org, amr = 'pwd'"), {"org": org_id})


def downgrade() -> None:
    bind = op.get_bind()
    orgs = bind.execute(sa.text("SELECT id FROM organizations")).fetchall()
    if len(orgs) > 1:
        raise RuntimeError("Downgrade refused: more than one organisation exists. Restore the pre-v4 backup instead.")

    with op.batch_alter_table("refresh_tokens") as b:
        b.drop_constraint("fk_refresh_tokens_organization_id", type_="foreignkey")
        b.drop_column("amr")
        b.drop_column("organization_id")

    with op.batch_alter_table("policy_versions") as b:
        b.drop_constraint("uq_policy_org_version", type_="unique")
        b.drop_index("ix_policy_versions_version")
    op.create_index("ix_policy_versions_version", "policy_versions", ["version"], unique=True)

    with op.batch_alter_table("biometric_enrolments") as b:
        b.drop_constraint("uq_enrolment_org_user", type_="unique")
        b.drop_index("ix_biometric_enrolments_user_id")
        b.create_unique_constraint("uq_biometric_enrolments_user_id", ["user_id"])

    with op.batch_alter_table("verification_sessions") as b:
        for ix in ("owner_ref", "camera_id", "pairing_id", "review_status"):
            b.drop_index(f"ix_verification_sessions_{ix}")
        b.drop_constraint("fk_verification_sessions_camera_id", type_="foreignkey")
        for col in ("owner_kind", "owner_ref", "channel", "camera_id", "pairing_id", "review_status", "risk_json"):
            b.drop_column(col)

    for table, _ in TENANT_TABLES:
        with op.batch_alter_table(table) as b:
            b.drop_index(f"ix_{table}_organization_id")
            b.drop_constraint(f"fk_{table}_organization_id", type_="foreignkey")
            b.drop_column("organization_id")

    with op.batch_alter_table("users") as b:
        b.add_column(sa.Column("role", sa.String(16), nullable=True))
    rows = bind.execute(sa.text("SELECT user_id, role FROM memberships")).fetchall()
    for uid, role in rows:
        bind.execute(sa.text("UPDATE users SET role = :r WHERE id = :u"), {"r": role, "u": uid})
    bind.execute(sa.text("UPDATE users SET role = 'ADMIN' WHERE is_platform_admin = :t AND role IS NULL"), {"t": True})
    bind.execute(sa.text("UPDATE users SET role = 'VIEWER' WHERE role IS NULL"))
    with op.batch_alter_table("users") as b:
        b.alter_column("role", existing_type=sa.String(16), nullable=False)
        b.drop_column("is_platform_admin")
        b.drop_column("last_org_id")

    # Reflection also loads tables referenced by foreign keys (users, verification_sessions …);
    # only the v4 tables themselves may be dropped.
    meta = sa.MetaData()
    meta.reflect(bind=bind, only=[t for t in V4_TABLES if sa.inspect(bind).has_table(t)])
    for table in reversed(meta.sorted_tables):
        if table.name in V4_TABLES:
            op.drop_table(table.name)
