"""Seed an isolated database for Playwright end-to-end tests.

Refuses to run unless APP_ENV=test. Creates synthetic users and synthetic
templates only (FakeProvider identity vectors) — no real biometric data.

    APP_ENV=test INFERENCE_PROVIDER=fake DATABASE_URL=sqlite:///./.e2e/e2e.db python -m backend.tests.e2e_seed
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    if os.environ.get("APP_ENV") != "test":
        print("e2e_seed refuses to run unless APP_ENV=test", file=sys.stderr)
        return 2
    from backend.config import settings
    from backend.database import SessionLocal
    from backend.migrations_util import upgrade_to_head
    from backend.models.user import RoleEnum, User
    from backend.services.auth_service import create_user, ensure_default_org, seed_admin
    from backend.services.policy_service import ensure_policy_seeded
    from backend.tests.helpers import enrol_direct

    upgrade_to_head(backup=False)
    db = SessionLocal()
    try:
        seed_admin(db)
        org = ensure_default_org(db)
        ensure_policy_seeded(db, org.id)
        password = os.environ.get("E2E_USER_PASSWORD", "E2e-User-Pass-1")
        people = [
            ("operator@e2e.local", "Olivia Operator", RoleEnum.OPERATOR, None),
            ("viewer@e2e.local", "Victor Viewer", RoleEnum.VIEWER, None),
            ("alice@e2e.local", "Alice Enrolled", RoleEnum.OPERATOR, 1),
            ("bob@e2e.local", "Bob Enrolled", RoleEnum.OPERATOR, 2),
            ("dave@e2e.local", "Dave Replace", RoleEnum.OPERATOR, 3),
            ("carol@e2e.local", "Carol New", RoleEnum.OPERATOR, None),
            ("erin@e2e.local", "Erin New", RoleEnum.OPERATOR, None),
            ("fiona@e2e.local", "Fiona Phone", RoleEnum.VIEWER, 5),
            ("gina@e2e.local", "Gina Kiosk", RoleEnum.VIEWER, 8),
            ("hank@e2e.local", "Hank Hosted", RoleEnum.VIEWER, 4),
            ("rita@e2e.local", "Rita Reviewer", RoleEnum.ADMIN, None),
            ("pat@e2e.local", "Pat Passkey", RoleEnum.OPERATOR, None),
            ("tom@e2e.local", "Tom Totp", RoleEnum.OPERATOR, None),
        ]
        for email, name, role, identity in people:
            u = db.query(User).filter(User.email == email).first() or create_user(db, email, name, password, role,
                                                                                   org_id=org.id)
            if identity is not None:
                enrol_direct(u.id, identity=identity)
        print(f"Seeded {len(people)} users into {settings.database_url}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
