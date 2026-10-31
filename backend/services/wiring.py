"""Explicit wiring of cross-service hooks (avoids import cycles between feature services).

* `capture_service.TERMINAL_HOOKS` — run when a capture session ends.
* `maintenance_service.TASKS`      — periodic background tasks.
Called once at application start-up (idempotent).
"""
from __future__ import annotations

from backend.services import (
    capture_service, device_service, integration_service, maintenance_service, mfa_service, notification_service, pairing_service,
    review_service,
)


def _add(registry: list, fn) -> None:
    if fn not in registry:
        registry.append(fn)


def register_all() -> None:
    _add(maintenance_service.TASKS, notification_service.maintenance_task)
    _add(maintenance_service.TASKS, mfa_service.expire_challenges)
    _add(maintenance_service.TASKS, pairing_service.expire_task)
    _add(maintenance_service.TASKS, review_service.expire_task)
    _add(maintenance_service.TASKS, integration_service.maintenance_task)
    _add(maintenance_service.TASKS, device_service.maintenance_task)
    # Order matters: the review decision is recorded before the pairing/integration see the result.
    _add(capture_service.TERMINAL_HOOKS, review_service.on_session_closed)
    _add(capture_service.TERMINAL_HOOKS, pairing_service.on_session_closed)
    _add(capture_service.TERMINAL_HOOKS, integration_service.on_session_closed)
    _add(capture_service.FRAME_HOOKS, pairing_service.publish_progress)
