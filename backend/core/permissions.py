"""Role-based access control.

This module is the single source of truth for what each role may do. Roles
are per organisation (membership role). The frontend receives the caller's
permission list for the active organisation from `/auth/me` and only uses it
for presentation; every endpoint enforces permissions server-side.

Organisation *creation* is not a role permission: it requires a platform
administrator (`users.is_platform_admin`).
"""
from __future__ import annotations

from enum import Enum

from backend.models.user import RoleEnum


class Permission(str, Enum):
    DASHBOARD_VIEW = "dashboard:view"
    USERS_DIRECTORY = "users:directory"          # minimal list for picking subjects
    USERS_READ = "users:read"                    # full user records
    USERS_MANAGE = "users:manage"                # create/edit/disable/roles
    ENROL_CAPTURE = "enrolment:capture"          # run an enrolment capture
    ENROL_APPROVE = "enrolment:approve"          # approve/reject sensitive changes
    ENROL_REVOKE = "enrolment:revoke"
    ENROL_READ = "enrolment:read"
    VERIFY_RUN = "verification:run"
    HISTORY_READ_OWN = "history:read_own"
    HISTORY_READ_ALL = "history:read_all"
    HISTORY_SCORES = "history:scores"            # distances / liveness confidences
    HISTORY_EXPORT = "history:export"
    SECURITY_READ = "security:read"
    ALERTS_MANAGE = "alerts:manage"
    ANALYTICS_READ = "analytics:read"
    EVALUATION_RUN = "evaluation:run"
    POLICY_READ = "policy:read"
    POLICY_MANAGE = "policy:manage"
    SYSTEM_HEALTH = "system:health"
    DOCUMENT_RUN = "document:run"
    ORG_READ = "org:read"                        # organisation profile and member list
    ORG_MANAGE = "org:manage"                    # members, invitations, organisation settings
    REVIEW_READ = "review:read"
    REVIEW_DECIDE = "review:decide"
    CAMERAS_READ = "cameras:read"
    CAMERAS_MANAGE = "cameras:manage"
    DEVICES_READ = "devices:read"
    DEVICES_MANAGE = "devices:manage"
    INTEGRATIONS_MANAGE = "integrations:manage"


_ALL = frozenset(Permission)

ROLE_PERMISSIONS: dict[RoleEnum, frozenset[Permission]] = {
    RoleEnum.ADMIN: _ALL,
    RoleEnum.OPERATOR: frozenset({
        Permission.DASHBOARD_VIEW,
        Permission.USERS_DIRECTORY,
        Permission.ENROL_CAPTURE,
        Permission.ENROL_READ,
        Permission.VERIFY_RUN,
        Permission.HISTORY_READ_OWN,
        Permission.HISTORY_SCORES,
        Permission.HISTORY_EXPORT,
        Permission.POLICY_READ,
        Permission.DOCUMENT_RUN,
        Permission.ORG_READ,
        Permission.CAMERAS_READ,
        Permission.DEVICES_READ,
    }),
    RoleEnum.VIEWER: frozenset({
        Permission.DASHBOARD_VIEW,
        Permission.HISTORY_READ_ALL,
        Permission.ANALYTICS_READ,
        Permission.ORG_READ,
        Permission.REVIEW_READ,
    }),
}


def permissions_for(role: RoleEnum) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(role, frozenset())


def has_permission(role: RoleEnum | None, perm: Permission) -> bool:
    return role is not None and perm in permissions_for(role)
