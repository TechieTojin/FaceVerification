"""SQLAlchemy ORM models (import all so Base.metadata is complete)."""
from backend.models.user import RoleEnum, User
from backend.models.organization import Invitation, Membership, Organization
from backend.models.biometric import BiometricEnrolment, ConsentRecord, EnrolmentRequest
from backend.models.documents import DocumentCheck, EvaluationRun, ModelMetadata
from backend.models.devices import Camera, Device, DeviceNonce, OfflineBundle, OfflineEvent, OfflineSubject
from backend.models.verification import (
    FrameDigest, LivenessChallenge, SecurityAlert, SecurityEvent,
    VerificationEvent, VerificationSession,
)
from backend.models.settings import AuditLog, PolicyVersion
from backend.models.auth import LoginAttempt, RefreshToken
from backend.models.mfa import AuthChallenge, MfaFactor, RecoveryCode, WebAuthnCredential
from backend.models.pairing import CapturePairing
from backend.models.reviews import Notification, NotificationPreference, ReviewEvent, VerificationReview
from backend.models.integrations import (
    ApiCredential, IdempotencyRecord, IntegrationClient, SigningKey, VerificationAssertion,
    WebhookDelivery, WebhookEndpoint,
)

__all__ = [
    "RoleEnum", "User", "Invitation", "Membership", "Organization",
    "BiometricEnrolment", "ConsentRecord", "EnrolmentRequest",
    "DocumentCheck", "EvaluationRun", "ModelMetadata",
    "Camera", "Device", "DeviceNonce", "OfflineBundle", "OfflineEvent", "OfflineSubject",
    "FrameDigest", "LivenessChallenge", "SecurityAlert", "SecurityEvent",
    "VerificationEvent", "VerificationSession",
    "AuditLog", "PolicyVersion",
    "LoginAttempt", "RefreshToken",
    "AuthChallenge", "MfaFactor", "RecoveryCode", "WebAuthnCredential",
    "CapturePairing",
    "Notification", "NotificationPreference", "ReviewEvent", "VerificationReview",
    "ApiCredential", "IdempotencyRecord", "IntegrationClient", "SigningKey", "VerificationAssertion",
    "WebhookDelivery", "WebhookEndpoint",
]
