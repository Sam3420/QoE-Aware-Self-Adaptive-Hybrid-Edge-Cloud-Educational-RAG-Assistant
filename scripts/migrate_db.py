"""Apply pending schema migrations to the app database (DATABASE_URL, default data/app.db).

Usage:
  .venv\\Scripts\\python scripts\\migrate_db.py            # apply pending migrations
  .venv\\Scripts\\python scripts\\migrate_db.py --status   # list applied and pending migrations
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.db.migrations import MIGRATIONS, applied, run_migrations  # noqa: E402
from app.db.session import engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", action="store_true", help="show migrations without applying them")
    args = parser.parse_args()
    print(f"Database: {get_settings().resolved_database_url}")
    if not args.status:
        ran = run_migrations(engine)
        print(f"Applied now: {', '.join(ran) or 'nothing (up to date)'}")
    done = set(applied(engine))
    for migration in MIGRATIONS:
        print(f"  [{'x' if migration.id in done else ' '}] {migration.id}: {migration.description}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
