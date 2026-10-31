"""v2.1 baseline schema (as previously created by Base.metadata.create_all).

Existing v2.1 databases are *stamped* at this revision (not re-created), so
every database — fresh or existing — reaches v3 through the same 0002 path.

Revision ID: 0001_v21_baseline
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_v21_baseline"
down_revision = None
branch_labels = None
depends_on = None

_ROLE = sa.Enum("ADMIN", "OPERATOR", "VIEWER", name="roleenum", native_enum=False, length=8)
_OUTCOME_VALUES = ("VERIFIED", "REJECTED", "NO_FACE", "MULTIPLE_FACES", "SPOOF_DETECTED", "ERROR", "TIMEOUT", "LOCKOUT")


def _outcome():
    return sa.Enum(*_OUTCOME_VALUES, name="outcomeenum", native_enum=False, length=14)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", _ROLE, nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "roles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(50), nullable=False, unique=True),
        sa.Column("description", sa.String(255), nullable=True),
    )
    op.create_table(
        "user_roles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("role_id", sa.String(36), sa.ForeignKey("roles.id"), nullable=False),
    )
    op.create_table(
        "biometric_enrolments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, unique=True),
        sa.Column("embedding_encrypted", sa.LargeBinary(), nullable=True),
        sa.Column("consent_given", sa.Boolean(), nullable=False),
        sa.Column("consent_timestamp", sa.DateTime(), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("enrolled_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("model_name", sa.String(50), nullable=False),
    )
    op.create_table(
        "consent_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("consent_given", sa.Boolean(), nullable=False),
        sa.Column("consent_text", sa.Text(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=True),
    )
    op.create_table(
        "verification_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("target_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_by_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("final_outcome", _outcome(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=False),
    )
    op.create_table(
        "verification_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), sa.ForeignKey("verification_sessions.id"), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("outcome", _outcome(), nullable=False),
        sa.Column("distance", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("liveness_score", sa.Float(), nullable=True),
        sa.Column("face_count", sa.Integer(), nullable=False),
        sa.Column("inference_ms", sa.Float(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("policy_version", sa.String(50), nullable=True),
        sa.Column("is_liveness_real", sa.Boolean(), nullable=True),
    )
    op.create_table(
        "security_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
    )
    op.create_table(
        "system_settings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("key", sa.String(100), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("updated_by", sa.String(36), nullable=True),
    )
    op.create_index("ix_system_settings_key", "system_settings", ["key"], unique=True)
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("replaced_by", sa.String(64), nullable=True),
    )
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=True)
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_table(
        "login_attempts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("identifier", sa.String(255), nullable=False),
        sa.Column("identifier_type", sa.String(10), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("attempted_at", sa.DateTime(), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=True),
    )
    op.create_index("ix_login_attempts_identifier", "login_attempts", ["identifier"])


def downgrade() -> None:
    for t in ("login_attempts", "refresh_tokens", "system_settings", "security_events", "verification_events",
              "verification_sessions", "consent_records", "biometric_enrolments", "user_roles", "roles", "users"):
        op.drop_table(t)
