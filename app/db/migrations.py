"""Versioned, idempotent schema migrations for the app database (no Alembic in this project).

`create_all` adds missing tables but never changes existing ones, so schema changes are listed here in order. Each
migration runs once per database; `schema_migrations` records which ones were applied and when. Run them with
`python scripts/migrate_db.py`; the app also runs them on start (`create_tables` and the agent's `ensure_ready`).
"""
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import Column, DateTime, MetaData, String, Table, inspect, select
from sqlalchemy.engine import Connection, Engine

from app.db.base import utc_now

_meta = MetaData()
schema_migrations = Table(
    "schema_migrations", _meta,
    Column("id", String(64), primary_key=True),
    Column("description", String(200), nullable=False),
    Column("applied_at", DateTime, nullable=False),
)


@dataclass(frozen=True)
class Migration:
    id: str
    description: str
    apply: Callable[[Connection], None]


def _create_concept_tracker(conn: Connection) -> None:
    from app.db.models.concept_tracker import ConceptEvent, LearnerTopicStrength

    for table in (LearnerTopicStrength.__table__, ConceptEvent.__table__):
        table.create(bind=conn, checkfirst=True)  # also creates the table's indexes and unique constraints


MIGRATIONS: tuple[Migration, ...] = (
    Migration("0001_concept_tracker", "learner_topic_strength + concept_events (concept mastery tracker)",
              _create_concept_tracker),
)


def applied(engine: Engine) -> list[str]:
    """Ids of the migrations already applied to this database."""
    if not inspect(engine).has_table("schema_migrations"):
        return []
    with engine.connect() as conn:
        return [row[0] for row in conn.execute(select(schema_migrations.c.id))]


def run_migrations(engine: Engine) -> list[str]:
    """Apply every pending migration in order, each in its own transaction; return the ids applied now."""
    _meta.create_all(bind=engine)
    done = set(applied(engine))
    ran = []
    for migration in MIGRATIONS:
        if migration.id in done:
            continue
        with engine.begin() as conn:
            migration.apply(conn)
            conn.execute(schema_migrations.insert().values(id=migration.id, description=migration.description,
                                                           applied_at=utc_now()))
        ran.append(migration.id)
    return ran
