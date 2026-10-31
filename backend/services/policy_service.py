"""Versioned verification policy.

* Policies are per organisation. The latest `PolicyVersion` row of an
  organisation is its active policy for *new* sessions.
* Each capture session stores the policy version and a JSON snapshot at
  creation; frames for that session are always evaluated against the snapshot,
  so changing policy never alters an in-progress session's rules.
* Reading the latest row on every session creation keeps all API workers
  consistent without a cache-invalidation protocol.
* A change that moves a sensitive field in its weakening direction must be
  explicitly confirmed and is recorded as a HIGH-severity security event.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models.settings import PolicyVersion

log = logging.getLogger(__name__)

APPROVED_RECOGNITION_MODELS = ("ArcFace",)


@dataclass(frozen=True)
class FieldSpec:
    key: str
    kind: Literal["float", "int", "bool", "choice"]
    default: Any
    group: str
    label: str
    description: str
    min: Optional[float] = None
    max: Optional[float] = None
    choices: tuple[str, ...] = ()
    unit: str = ""
    # Direction that weakens security; changes that way need confirmation.
    weakening: Optional[Literal["increase", "decrease", "disable"]] = None


SPECS: tuple[FieldSpec, ...] = (
    # ── Matching ────────────────────────────────────────────────────────────
    FieldSpec("match_threshold", "float", 0.68, "matching", "Match threshold (cosine distance)",
              "A live face matches when its cosine distance to the reference is at or below this value. "
              "0.68 is DeepFace's published ArcFace default; it is not calibrated for this deployment. "
              "The ceiling prevents loosening beyond the published value.",
              min=0.30, max=0.68, weakening="increase"),
    FieldSpec("require_bracket_match", "bool", True, "matching", "Match at start and end of challenge",
              "Require the identity match on both the opening and closing straight-ahead frames, so the "
              "same person must be present throughout the liveness challenge.", weakening="disable"),
    FieldSpec("max_attempts", "int", 3, "matching", "Maximum attempts per session",
              "Failed liveness rounds (and mismatches when retry is enabled) allowed before the session ends.",
              min=1, max=10, weakening="increase"),
    FieldSpec("retry_on_mismatch", "bool", False, "matching", "Allow retry after a mismatch",
              "If disabled, the first NOT_MATCHED decision ends the session.", weakening="increase"),
    FieldSpec("recognition_model", "choice", "ArcFace", "matching", "Recognition model",
              "Only models with installed, approved weights are selectable.", choices=APPROVED_RECOGNITION_MODELS),
    # ── Session ────────────────────────────────────────────────────────────
    FieldSpec("session_timeout_seconds", "int", 120, "session", "Session timeout", "Hard lifetime of a capture session.",
              min=30, max=600, unit="s", weakening="increase"),
    FieldSpec("inactivity_timeout_seconds", "int", 20, "session", "Inactivity timeout",
              "A session with no frames for this long is expired.", min=5, max=120, unit="s"),
    FieldSpec("max_frames_per_session", "int", 150, "session", "Frame processing limit",
              "Maximum frames the server will process for one session.", min=20, max=400, weakening="increase"),
    FieldSpec("max_frame_bytes", "int", 600_000, "session", "Maximum frame size", "Uploads above this size are rejected.",
              min=50_000, max=2_000_000, unit="bytes"),
    FieldSpec("operator_session_rate_limit", "int", 60, "session", "Sessions per operator per 10 min",
              "Hard limit on capture sessions an operator may create in a rolling 10-minute window.",
              min=5, max=1000, weakening="increase"),
    # ── Liveness ───────────────────────────────────────────────────────────
    FieldSpec("passive_liveness_required", "bool", True, "liveness", "Passive anti-spoofing",
              "Run MiniFASNet texture anti-spoofing on the straight-ahead frames.", weakening="disable"),
    FieldSpec("active_liveness_required", "bool", True, "liveness", "Active challenge",
              "Require the randomised blink / head-turn challenge.", weakening="disable"),
    FieldSpec("challenge_steps", "int", 2, "liveness", "Random challenge actions",
              "Number of randomised actions between the opening and closing straight-ahead steps.",
              min=1, max=4, weakening="decrease"),
    FieldSpec("challenge_step_timeout_seconds", "int", 10, "liveness", "Challenge step timeout",
              "Time allowed to complete each action.", min=4, max=30, unit="s"),
    FieldSpec("yaw_threshold_degrees", "float", 18.0, "liveness", "Head-turn angle",
              "Minimum head rotation for a turn action.", min=10.0, max=35.0, unit="°", weakening="decrease"),
    FieldSpec("blink_threshold", "float", 0.5, "liveness", "Blink closure score",
              "Eye-closure blendshape score both eyes must reach for a blink.", min=0.3, max=0.9,
              weakening="decrease"),
    # ── Quality ────────────────────────────────────────────────────────────
    FieldSpec("verify_min_quality", "float", 0.40, "quality", "Minimum verification quality",
              "Composite face-crop quality required for identity frames.", min=0.20, max=0.95, weakening="decrease"),
    FieldSpec("enrol_min_quality", "float", 0.55, "quality", "Minimum enrolment quality",
              "Composite face-crop quality required for enrolment samples.", min=0.30, max=0.95, weakening="decrease"),
    FieldSpec("min_face_width_ratio", "float", 0.20, "quality", "Minimum face width",
              "Face width as a fraction of frame width; smaller faces get 'move closer'.", min=0.10, max=0.50),
    FieldSpec("max_face_width_ratio", "float", 0.75, "quality", "Maximum face width",
              "Larger faces get 'move back'.", min=0.40, max=0.95),
    FieldSpec("enrol_samples", "int", 3, "quality", "Enrolment samples",
              "Number of quality-controlled straight-ahead samples averaged into the template.",
              min=3, max=7, weakening="decrease"),
    FieldSpec("enrol_consistency_max_distance", "float", 0.40, "quality", "Enrolment sample consistency",
              "Maximum cosine distance between any two enrolment samples.", min=0.20, max=0.60,
              weakening="increase"),
    FieldSpec("duplicate_check_enabled", "bool", True, "quality", "Duplicate enrolment check",
              "Compare new templates with other active templates; matches require admin approval.",
              weakening="disable"),
    FieldSpec("require_four_eyes", "bool", False, "quality", "Four-eyes approval",
              "Approver of a sensitive enrolment change must differ from the requester.", weakening="disable"),
    # ── Capture (sent to clients) ──────────────────────────────────────────
    FieldSpec("capture_interval_ms", "int", 350, "capture", "Capture interval",
              "Minimum interval between submitted frames.", min=150, max=2000, unit="ms"),
    FieldSpec("capture_max_width", "int", 640, "capture", "Capture width", "Frames are downscaled to this width before upload.",
              min=320, max=1280, unit="px"),
    FieldSpec("capture_jpeg_quality", "float", 0.85, "capture", "JPEG quality", "Upload encoding quality.", min=0.6, max=0.95),
    # ── Alerts ─────────────────────────────────────────────────────────────
    FieldSpec("alert_failed_verifications_threshold", "int", 5, "alerts", "Failed verifications alert",
              "Raise an alert when one identity fails this many sessions within an hour.", min=2, max=50),
    FieldSpec("alert_liveness_failures_threshold", "int", 3, "alerts", "Liveness failures alert",
              "Raise an alert when one identity fails liveness this many times within an hour.", min=2, max=50),
    FieldSpec("alert_session_creation_threshold", "int", 30, "alerts", "Session burst alert",
              "Raise an alert when one operator creates this many sessions within 10 minutes.", min=5, max=500),
    FieldSpec("alert_verification_frequency_threshold", "int", 10, "alerts", "Verification frequency alert",
              "Raise an alert when one identity is verified this many times within an hour.", min=3, max=200),
    # ── Retention ──────────────────────────────────────────────────────────
    FieldSpec("retention_events_days", "int", 365, "retention", "Frame event retention",
              "Per-frame verification events older than this are deleted.", min=7, max=3650, unit="days"),
    FieldSpec("retention_sessions_days", "int", 365, "retention", "Session retention",
              "Closed sessions and their challenge records older than this are deleted.", min=7, max=3650, unit="days"),
    FieldSpec("retention_documents_hours", "int", 24, "retention", "Document portrait retention",
              "Encrypted document-portrait templates are purged after this period.", min=1, max=168, unit="h"),
    FieldSpec("retention_login_attempts_days", "int", 30, "retention", "Login attempt retention",
              "", min=1, max=365, unit="days"),
    FieldSpec("retention_frame_digest_hours", "int", 48, "retention", "Replay digest retention",
              "How long frame digests are kept for cross-session replay detection.", min=1, max=168, unit="h"),
    FieldSpec("retention_notifications_days", "int", 90, "retention", "Notification retention",
              "Read notifications older than this are deleted.", min=7, max=3650, unit="days"),
    # ── Manual review ──────────────────────────────────────────────────────
    FieldSpec("review_enabled", "bool", True, "review", "Manual review of uncertain results",
              "Borderline matches and low-confidence liveness results are routed to a reviewer queue. "
              "Disabling this means such results are only ever decided automatically.", weakening="disable"),
    FieldSpec("review_distance_margin", "float", 0.03, "review", "Borderline match band",
              "A decision whose distance is within this margin of the threshold is flagged for review.",
              min=0.0, max=0.15, weakening="decrease"),
    FieldSpec("review_passive_confidence_floor", "float", 0.80, "review", "Liveness confidence floor",
              "A pass whose passive anti-spoofing confidence is below this value is flagged for review.",
              min=0.5, max=0.99, weakening="decrease"),
    FieldSpec("review_expiry_hours", "int", 48, "review", "Review expiry",
              "Undecided reviews expire after this period.", min=1, max=720, unit="h"),
    # ── Authentication ─────────────────────────────────────────────────────
    FieldSpec("mfa_required_for_admins", "bool", False, "authentication", "Require MFA for administrators",
              "Administrators must complete a second factor (passkey, authenticator app or recovery code) "
              "after their password before a session is issued.", weakening="disable"),
    FieldSpec("passkey_login_requires_face", "bool", False, "authentication", "Face check after passkey sign-in",
              "After a successful passkey sign-in, enrolled members must also pass a live face verification "
              "before a session is issued. The face check never signs anyone in on its own.",
              weakening="disable"),
    FieldSpec("high_risk_requires_face", "bool", False, "authentication", "Face + factor for high-risk actions",
              "High-risk administrative actions (integration credentials, kiosk provisioning, creating or "
              "suspending organisations) require a live face verification of the administrator plus an independent factor.",
              weakening="disable"),
    FieldSpec("step_up_max_age_seconds", "int", 600, "authentication", "Step-up validity",
              "How long a step-up confirmation authorises sensitive administrative actions.",
              min=60, max=3600, unit="s", weakening="increase"),
    # ── Cross-device ───────────────────────────────────────────────────────
    FieldSpec("pairing_ttl_seconds", "int", 180, "cross_device", "QR pairing lifetime",
              "A QR code must be scanned within this period.", min=60, max=600, unit="s", weakening="increase"),
    FieldSpec("pairing_max_sessions", "int", 3, "cross_device", "Captures per pairing",
              "How many capture sessions one confirmed pairing may start (retries).", min=1, max=5,
              weakening="increase"),
    # ── Offline kiosks ─────────────────────────────────────────────────────
    FieldSpec("offline_validity_hours", "int", 24, "offline", "Offline validity period",
              "Templates provisioned to a kiosk expire after this many hours; the kiosk must reconnect to "
              "renew them. Revocations reach a disconnected kiosk only when it reconnects.",
              min=1, max=72, unit="h", weakening="increase"),
    FieldSpec("offline_max_subjects", "int", 200, "offline", "Subjects per kiosk",
              "Upper bound on identities provisioned to one kiosk.", min=1, max=5000, weakening="increase"),
)

SPEC_BY_KEY = {s.key: s for s in SPECS}
DEFAULTS: dict[str, Any] = {s.key: s.default for s in SPECS}


class PolicyValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


class SensitiveChangeNotConfirmed(Exception):
    def __init__(self, changes: list[str]):
        super().__init__("Sensitive changes require confirmation")
        self.changes = changes


class Policy:
    """Immutable view over a policy snapshot."""

    __slots__ = ("version", "_values")

    def __init__(self, version: int, values: dict[str, Any]):
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in values.items() if k in SPEC_BY_KEY})
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "_values", merged)

    def __getattr__(self, item: str) -> Any:
        try:
            return self._values[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, *_: Any) -> None:
        raise AttributeError("Policy is immutable")

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def to_json(self) -> str:
        return json.dumps({"version": self.version, "values": self._values}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "Policy":
        data = json.loads(raw)
        return cls(int(data["version"]), data["values"])

    def capture_config(self) -> dict[str, Any]:
        return {
            "interval_ms": self.capture_interval_ms,
            "max_width": self.capture_max_width,
            "jpeg_quality": self.capture_jpeg_quality,
            "max_frame_bytes": self.max_frame_bytes,
            "min_face_width_ratio": self.min_face_width_ratio,
            "max_face_width_ratio": self.max_face_width_ratio,
            "yaw_threshold_degrees": self.yaw_threshold_degrees,
            "blink_threshold": self.blink_threshold,
        }


def coerce_value(spec: FieldSpec, raw: Any) -> Any:
    """Validate a single value against its spec. Raises ValueError with a message."""
    if spec.kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in ("true", "false", "1", "0"):
            return raw.lower() in ("true", "1")
        raise ValueError("must be true or false")
    if spec.kind == "choice":
        if raw not in spec.choices:
            raise ValueError(f"must be one of {', '.join(spec.choices)}")
        return raw
    if isinstance(raw, bool):
        raise ValueError("must be a number")
    try:
        val: float | int = int(raw) if spec.kind == "int" else float(raw)
    except (TypeError, ValueError):
        raise ValueError("must be an integer" if spec.kind == "int" else "must be a number") from None
    if spec.kind == "int" and isinstance(raw, float) and not float(raw).is_integer():
        raise ValueError("must be an integer")
    if val != val:  # NaN
        raise ValueError("must be a number")
    if spec.min is not None and val < spec.min:
        raise ValueError(f"must be at least {spec.min}")
    if spec.max is not None and val > spec.max:
        raise ValueError(f"must be at most {spec.max}")
    return val


def _cross_field_errors(values: dict[str, Any]) -> dict[str, str]:
    errs: dict[str, str] = {}
    if values["min_face_width_ratio"] >= values["max_face_width_ratio"]:
        errs["min_face_width_ratio"] = "must be smaller than max_face_width_ratio"
    if values["enrol_min_quality"] < values["verify_min_quality"]:
        errs["enrol_min_quality"] = "must be at least verify_min_quality"
    if not values["passive_liveness_required"] and not values["active_liveness_required"]:
        errs["active_liveness_required"] = "at least one liveness mechanism must remain enabled"
    return errs


def _is_weakening(spec: FieldSpec, old: Any, new: Any) -> bool:
    if spec.weakening is None or old == new:
        return False
    if spec.weakening == "disable":
        return bool(old) and not bool(new)
    if spec.kind == "bool":  # bool with increase semantic → enabling weakens
        return (not old) and bool(new)
    if spec.weakening == "increase":
        return new > old
    return new < old


def _latest(db: Session, org_id: str) -> Optional[PolicyVersion]:
    return (db.query(PolicyVersion).filter(PolicyVersion.organization_id == org_id)
            .order_by(PolicyVersion.version.desc()).first())


def get_active_policy(db: Session, org_id: str) -> Policy:
    row = _latest(db, org_id)
    if row is None:
        row = ensure_policy_seeded(db, org_id)
    return Policy(row.version, json.loads(row.values_json))


def ensure_policy_seeded(db: Session, org_id: str) -> PolicyVersion:
    row = _latest(db, org_id)
    if row is not None:
        return row
    row = PolicyVersion(organization_id=org_id, version=1, values_json=json.dumps(DEFAULTS, sort_keys=True),
                        change_summary="Initial default policy", is_sensitive_change=False)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        row = _latest(db, org_id)
    return row


def plan_update(current: Policy, changes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]], list[str]]:
    """Validate changes. Returns (new_values, diff, weakening_keys). Raises PolicyValidationError."""
    errors: dict[str, str] = {}
    new_values = current.as_dict()
    for key, raw in changes.items():
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            errors[key] = "unknown setting"
            continue
        try:
            new_values[key] = coerce_value(spec, raw)
        except ValueError as exc:
            errors[key] = str(exc)
    if errors:
        raise PolicyValidationError(errors)
    errors = _cross_field_errors(new_values)
    if errors:
        raise PolicyValidationError(errors)
    old = current.as_dict()
    diff = {k: (old[k], new_values[k]) for k in new_values if old[k] != new_values[k]}
    weakening = [k for k, (o, n) in diff.items() if _is_weakening(SPEC_BY_KEY[k], o, n)]
    return new_values, diff, weakening


def update_policy(db: Session, org_id: str, changes: dict[str, Any], actor_id: str, confirm: bool,
                  reason: str = "") -> tuple[Policy, dict[str, tuple[Any, Any]], list[str]]:
    """Persist a new policy version. Caller handles audit/security logging."""
    current = get_active_policy(db, org_id)
    new_values, diff, weakening = plan_update(current, changes)
    if not diff:
        return current, {}, []
    if weakening and not confirm:
        raise SensitiveChangeNotConfirmed(weakening)
    summary = ", ".join(f"{k}: {o} → {n}" for k, (o, n) in diff.items())
    if reason:
        summary = f"{summary} — {reason[:300]}"
    row = PolicyVersion(organization_id=org_id, version=current.version + 1,
                        values_json=json.dumps(new_values, sort_keys=True),
                        change_summary=summary, is_sensitive_change=bool(weakening),
                        created_by_user_id=actor_id)
    db.add(row)
    db.commit()
    return Policy(row.version, new_values), diff, weakening


def describe_specs() -> list[dict[str, Any]]:
    return [
        {
            "key": s.key, "kind": s.kind, "default": s.default, "group": s.group, "label": s.label,
            "description": s.description, "min": s.min, "max": s.max, "choices": list(s.choices),
            "unit": s.unit, "sensitive": s.weakening is not None, "weakening": s.weakening,
        }
        for s in SPECS
    ]
