"""v3.0 platform schema and data migration.

Data handling:
  * Legacy (v2.1) templates were produced by an RGB-swapped, Haar-detected
    pipeline and are not comparable with v3 embeddings. They are kept
    (encrypted, untouched) but marked REENROLMENT_REQUIRED, so they can never
    be used for matching until the user is re-enrolled.
  * Legacy outcome codes are mapped to the v3 vocabulary.
  * Legacy sessions still marked active are closed as SESSION_EXPIRED.
  * Legacy system_settings were never applied by v2.1. They are recorded in the
    initial policy version's change summary for administrator review and are
    NOT silently applied (e.g. a stored enable_liveness=false would otherwise
    start disabling anti-spoofing).
  * roles / user_roles (unused in v2.1) are dropped only if empty.

Revision ID: 0002_v3_platform
Revises: 0001_v21_baseline
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from alembic import op
import sqlalchemy as sa

revision = "0002_v3_platform"
down_revision = "0001_v21_baseline"
branch_labels = None
depends_on = None

OUTCOME_MAP = {
    "REJECTED": "NOT_MATCHED",
    "SPOOF_DETECTED": "LIVENESS_FAILED",
    "TIMEOUT": "SESSION_EXPIRED",
    "LOCKOUT": "ATTEMPTS_EXHAUSTED",
}


def _now() -> datetime:
    return datetime.utcnow()


def upgrade() -> None:
    bind = op.get_bind()

    # ── new standalone tables ────────────────────────────────────────────
    op.create_table(
        "policy_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("values_json", sa.Text(), nullable=False),
        sa.Column("change_summary", sa.Text(), nullable=True),
        sa.Column("is_sensitive_change", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_policy_versions_version", "policy_versions", ["version"], unique=True)

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("actor_user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("target_type", sa.String(32), nullable=True),
        sa.Column("target_id", sa.String(64), nullable=True),
        sa.Column("details_json", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
    )
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"])
    op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_target_id", "audit_logs", ["target_id"])

    op.create_table(
        "document_checks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_by_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("target_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("subject_label", sa.String(200), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("consent_given", sa.Boolean(), nullable=False),
        sa.Column("consent_at", sa.DateTime(), nullable=True),
        sa.Column("document_type", sa.String(24), nullable=False),
        sa.Column("document_quality_score", sa.Float(), nullable=True),
        sa.Column("document_quality_issues", sa.Text(), nullable=True),
        sa.Column("portrait_detected", sa.Boolean(), nullable=False),
        sa.Column("portrait_template_encrypted", sa.LargeBinary(), nullable=True),
        sa.Column("face_match_result", sa.String(16), nullable=False),
        sa.Column("face_match_distance", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("liveness_passed", sa.Boolean(), nullable=True),
        sa.Column("authenticity_status", sa.String(16), nullable=False),
        sa.Column("authenticity_provider", sa.String(32), nullable=True),
        sa.Column("authenticity_detail", sa.Text(), nullable=True),
        sa.Column("overall_decision", sa.String(40), nullable=False),
        sa.Column("session_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("retention_expires_at", sa.DateTime(), nullable=False),
        sa.Column("purged_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_document_checks_created_by_user_id", "document_checks", ["created_by_user_id"])
    op.create_index("ix_document_checks_status", "document_checks", ["status"])
    op.create_index("ix_document_checks_created_at", "document_checks", ["created_at"])

    op.create_table(
        "model_metadata",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False, unique=True),
        sa.Column("task", sa.String(24), nullable=False),
        sa.Column("file_name", sa.String(128), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=True),
        sa.Column("approved", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("load_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_by_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("model_name", sa.String(50), nullable=False),
        sa.Column("dataset_label", sa.String(200), nullable=True),
        sa.Column("identity_count", sa.Integer(), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False),
        sa.Column("skipped_images", sa.Integer(), nullable=False),
        sa.Column("genuine_pairs", sa.Integer(), nullable=False),
        sa.Column("impostor_pairs", sa.Integer(), nullable=False),
        sa.Column("results_json", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_table(
        "enrolment_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("requested_by_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("session_id", sa.String(36), nullable=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("approval_reasons", sa.Text(), nullable=True),
        sa.Column("candidate_template_encrypted", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.String(50), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("consistency_score", sa.Float(), nullable=True),
        sa.Column("duplicate_user_ids", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("reviewed_by_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
    )
    op.create_index("ix_enrolment_requests_user_id", "enrolment_requests", ["user_id"])
    op.create_index("ix_enrolment_requests_status", "enrolment_requests", ["status"])

    op.create_table(
        "security_alerts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("rule", sa.String(48), nullable=False),
        sa.Column("severity", sa.String(12), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("subject_type", sa.String(16), nullable=True),
        sa.Column("subject_id", sa.String(64), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("occurrence_count", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("acknowledged_by_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(), nullable=True),
        sa.Column("resolved_by_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
    )
    op.create_index("ix_security_alerts_rule", "security_alerts", ["rule"])
    op.create_index("ix_security_alerts_status", "security_alerts", ["status"])
    op.create_index("ix_security_alerts_subject_id", "security_alerts", ["subject_id"])
    op.create_index("ix_security_alerts_last_seen_at", "security_alerts", ["last_seen_at"])

    op.create_table(
        "frame_digests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_frame_digests_digest", "frame_digests", ["digest"])
    op.create_index("ix_frame_digests_created_at", "frame_digests", ["created_at"])

    # ── users ─────────────────────────────────────────────────────────────
    with op.batch_alter_table("users") as b:
        b.add_column(sa.Column("password_changed_at", sa.DateTime(), nullable=True))
        b.add_column(sa.Column("disabled_at", sa.DateTime(), nullable=True))
        b.alter_column("role", type_=sa.Enum("ADMIN", "OPERATOR", "VIEWER", name="roleenum", native_enum=False,
                                             length=16), existing_nullable=False)
    rows = bind.execute(sa.text("SELECT id, email FROM users")).fetchall()
    taken = {r.email.strip().lower() for r in rows if r.email == r.email.strip().lower()}
    for r in rows:
        norm = r.email.strip().lower()
        if norm != r.email and norm not in taken:
            bind.execute(sa.text("UPDATE users SET email = :e WHERE id = :i"), {"e": norm, "i": r.id})
            taken.add(norm)

    # ── biometric_enrolments ──────────────────────────────────────────────
    with op.batch_alter_table("biometric_enrolments") as b:
        b.add_column(sa.Column("status", sa.String(24), nullable=False, server_default="NOT_ENROLLED"))
        b.add_column(sa.Column("template_version", sa.Integer(), nullable=False, server_default="2"))
        b.add_column(sa.Column("embedding_dim", sa.Integer(), nullable=True))
        b.add_column(sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"))
        b.add_column(sa.Column("consistency_score", sa.Float(), nullable=True))
        b.add_column(sa.Column("enrolled_by_user_id", sa.String(36), sa.ForeignKey("users.id", name="fk_enr_enrolled_by"), nullable=True))
        b.add_column(sa.Column("approved_by_user_id", sa.String(36), sa.ForeignKey("users.id", name="fk_enr_approved_by"), nullable=True))
        b.add_column(sa.Column("revoked_at", sa.DateTime(), nullable=True))
        b.add_column(sa.Column("revoked_by_user_id", sa.String(36), sa.ForeignKey("users.id", name="fk_enr_revoked_by"), nullable=True))
        b.add_column(sa.Column("reenrolment_requested", sa.Boolean(), nullable=False, server_default=sa.false()))
        b.add_column(sa.Column("updated_at", sa.DateTime(), nullable=True))
    bind.execute(sa.text(
        "UPDATE biometric_enrolments SET "
        "status = CASE "
        "  WHEN embedding_encrypted IS NOT NULL THEN 'REENROLMENT_REQUIRED' "
        "  WHEN is_active = :f AND enrolled_at IS NULL AND consent_given = :t THEN 'REVOKED' "
        "  ELSE 'NOT_ENROLLED' END, "
        "template_version = CASE WHEN embedding_encrypted IS NOT NULL THEN 1 ELSE 2 END, "
        "embedding_dim = CASE WHEN embedding_encrypted IS NOT NULL THEN 512 ELSE NULL END, "
        "sample_count = CASE WHEN embedding_encrypted IS NOT NULL THEN 1 ELSE 0 END, "
        "updated_at = :now"), {"f": False, "t": True, "now": _now()})
    with op.batch_alter_table("biometric_enrolments") as b:
        b.alter_column("updated_at", existing_type=sa.DateTime(), nullable=False)
        b.drop_column("is_active")

    # ── consent_records ───────────────────────────────────────────────────
    with op.batch_alter_table("consent_records") as b:
        b.add_column(sa.Column("consent_version", sa.String(20), nullable=True))
        b.add_column(sa.Column("purpose", sa.String(40), nullable=False, server_default="BIOMETRIC_ENROLMENT"))
        b.add_column(sa.Column("method", sa.String(30), nullable=True))
        b.add_column(sa.Column("actor_user_id", sa.String(36), sa.ForeignKey("users.id", name="fk_consent_actor"), nullable=True))
        b.create_index("ix_consent_records_user_id", ["user_id"])
    bind.execute(sa.text("UPDATE consent_records SET method = 'LEGACY_OPERATOR_ASSERTED' WHERE method IS NULL"))

    # ── verification_sessions ─────────────────────────────────────────────
    with op.batch_alter_table("verification_sessions") as b:
        b.add_column(sa.Column("purpose", sa.String(16), nullable=False, server_default="VERIFY"))
        b.add_column(sa.Column("document_check_id", sa.String(36),
                               sa.ForeignKey("document_checks.id", name="fk_session_document"), nullable=True))
        b.add_column(sa.Column("reason_code", sa.String(48), nullable=True))
        b.add_column(sa.Column("attempts_used", sa.Integer(), nullable=False, server_default="0"))
        b.add_column(sa.Column("frame_seq", sa.Integer(), nullable=False, server_default="0"))
        b.add_column(sa.Column("frame_token_hash", sa.String(64), nullable=True))
        b.add_column(sa.Column("in_flight", sa.Boolean(), nullable=False, server_default=sa.false()))
        b.add_column(sa.Column("in_flight_since", sa.DateTime(), nullable=True))
        b.add_column(sa.Column("last_activity_at", sa.DateTime(), nullable=True))
        b.add_column(sa.Column("policy_version", sa.Integer(), nullable=True))
        b.add_column(sa.Column("policy_snapshot", sa.Text(), nullable=True))
        b.add_column(sa.Column("challenge_round", sa.Integer(), nullable=False, server_default="0"))
        b.add_column(sa.Column("challenge_index", sa.Integer(), nullable=False, server_default="0"))
        b.add_column(sa.Column("step_started_at", sa.DateTime(), nullable=True))
        b.add_column(sa.Column("state_json", sa.Text(), nullable=True))
        b.add_column(sa.Column("samples_encrypted", sa.LargeBinary(), nullable=True))
        b.add_column(sa.Column("final_distance", sa.Float(), nullable=True))
        b.add_column(sa.Column("threshold", sa.Float(), nullable=True))
        b.add_column(sa.Column("total_processing_ms", sa.Float(), nullable=False, server_default="0"))
        b.alter_column("target_user_id", existing_type=sa.String(36), nullable=True)
        b.alter_column("final_outcome", type_=sa.String(32), existing_nullable=True)
    for old, new in OUTCOME_MAP.items():
        bind.execute(sa.text("UPDATE verification_sessions SET final_outcome = :n WHERE final_outcome = :o"),
                     {"n": new, "o": old})
    bind.execute(sa.text(
        "UPDATE verification_sessions SET is_active = :f, final_outcome = 'SESSION_EXPIRED', "
        "reason_code = 'MIGRATED_V21', completed_at = COALESCE(completed_at, :now) WHERE is_active = :t"),
        {"f": False, "t": True, "now": _now()})
    with op.batch_alter_table("verification_sessions") as b:
        for col in ("purpose", "target_user_id", "created_by_user_id", "created_at", "is_active", "final_outcome"):
            b.create_index(f"ix_verification_sessions_{col}", [col])

    # ── verification_events ───────────────────────────────────────────────
    with op.batch_alter_table("verification_events") as b:
        b.add_column(sa.Column("reason_code", sa.String(48), nullable=True))
        b.add_column(sa.Column("frame_seq", sa.Integer(), nullable=True))
        b.add_column(sa.Column("step_type", sa.String(24), nullable=True))
        b.add_column(sa.Column("passive_label", sa.String(8), nullable=True))
        b.add_column(sa.Column("passive_confidence", sa.Float(), nullable=True))
        b.add_column(sa.Column("quality_score", sa.Float(), nullable=True))
        b.add_column(sa.Column("yaw", sa.Float(), nullable=True))
        b.add_column(sa.Column("pitch", sa.Float(), nullable=True))
        b.alter_column("outcome", type_=sa.String(32), existing_nullable=False)
    # v2.1 stored the predicted-class confidence as "liveness_score"; keep it under its correct meaning.
    bind.execute(sa.text(
        "UPDATE verification_events SET "
        "passive_label = CASE WHEN is_liveness_real = :t THEN 'REAL' WHEN is_liveness_real = :f THEN 'SPOOF' ELSE NULL END, "
        "passive_confidence = liveness_score"), {"t": True, "f": False})
    for old, new in OUTCOME_MAP.items():
        bind.execute(sa.text("UPDATE verification_events SET outcome = :n WHERE outcome = :o"), {"n": new, "o": old})
    with op.batch_alter_table("verification_events") as b:
        b.drop_column("liveness_score")
        b.drop_column("is_liveness_real")
        for col in ("session_id", "user_id", "timestamp", "outcome"):
            b.create_index(f"ix_verification_events_{col}", [col])

    op.create_table(
        "liveness_challenges",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), sa.ForeignKey("verification_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("step_type", sa.String(24), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("evidence_json", sa.Text(), nullable=True),
    )
    op.create_index("ix_liveness_challenges_session_id", "liveness_challenges", ["session_id"])

    # ── security_events / auth tables ─────────────────────────────────────
    with op.batch_alter_table("security_events") as b:
        for col in ("user_id", "event_type", "timestamp"):
            b.create_index(f"ix_security_events_{col}", [col])
    with op.batch_alter_table("refresh_tokens") as b:
        b.add_column(sa.Column("family_id", sa.String(36), nullable=True))
        b.add_column(sa.Column("user_agent", sa.String(200), nullable=True))
        b.add_column(sa.Column("ip_address", sa.String(45), nullable=True))
        b.create_index("ix_refresh_tokens_family_id", ["family_id"])
    for (tid,) in bind.execute(sa.text("SELECT id FROM refresh_tokens WHERE family_id IS NULL")).fetchall():
        bind.execute(sa.text("UPDATE refresh_tokens SET family_id = :f WHERE id = :i"), {"f": str(uuid.uuid4()), "i": tid})
    with op.batch_alter_table("login_attempts") as b:
        b.create_index("ix_login_attempts_attempted_at", ["attempted_at"])

    # ── legacy settings → initial policy version (recorded, not applied) ──
    from backend.services.policy_service import DEFAULTS
    legacy = {r.key: r.value for r in bind.execute(sa.text("SELECT key, value FROM system_settings")).fetchall()}
    summary = "Initial v3 policy (defaults)."
    if legacy:
        summary += (" Legacy v2.1 settings were stored but never enforced by v2.1; they were NOT applied. "
                    "Review and re-apply deliberately if intended: " + json.dumps(legacy, sort_keys=True))
    bind.execute(sa.text(
        "INSERT INTO policy_versions (id, version, values_json, change_summary, is_sensitive_change, created_at) "
        "VALUES (:id, 1, :v, :s, :f, :now)"),
        {"id": str(uuid.uuid4()), "v": json.dumps(DEFAULTS, sort_keys=True), "s": summary, "f": False, "now": _now()})
    op.drop_index("ix_system_settings_key", table_name="system_settings")
    op.drop_table("system_settings")

    # ── unused v2.1 tables ────────────────────────────────────────────────
    n_user_roles = bind.execute(sa.text("SELECT COUNT(*) FROM user_roles")).scalar() or 0
    n_roles = bind.execute(sa.text("SELECT COUNT(*) FROM roles")).scalar() or 0
    if n_user_roles == 0:
        op.drop_table("user_roles")
        if n_roles == 0:
            op.drop_table("roles")


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrade from v3 is not supported. Restore the automatic pre-migration backup in ./backups instead.")
