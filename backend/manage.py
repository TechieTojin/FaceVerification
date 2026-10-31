"""Operational commands.

    python -m backend.manage migrate      # back up (SQLite) and upgrade schema to head
    python -m backend.manage backup       # consistent SQLite backup into BACKUP_DIR
    python -m backend.manage check        # print schema revision and readiness
"""
from __future__ import annotations

import json
import sys

from backend.config import settings
from backend.migrations_util import backup_sqlite, current_revision, head_revision, upgrade_to_head


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "migrate":
        print(json.dumps(upgrade_to_head(), indent=2, default=str))
        return 0
    if cmd == "backup":
        path = backup_sqlite(settings.database_url, "manual")
        print(path or "Not a file-based SQLite database; use your database's native backup tooling (e.g. pg_dump).")
        return 0
    if cmd == "check":
        rev, legacy = current_revision()
        print(json.dumps({"database": settings.database_url.split("@")[-1], "revision": rev, "head": head_revision(),
                          "legacy_v21_schema": legacy, "up_to_date": rev == head_revision()}, indent=2))
        return 0 if rev == head_revision() else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
