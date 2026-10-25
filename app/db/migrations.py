"""Versioned, idempotent schema migrations for the app database (no Alembic in this project).

`create_all` adds missing tables but never changes existing ones, so schema changes are listed here in order. Each
migration runs once per database; `schema_migrations` records which ones were applied and when. Run them with
`python scripts/migrate_db.py`; the app also runs them on start (`create_tables` and the agent's `ensure_ready`).
"""
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import Column, DateTime, MetaData, String, Table, inspect, select, text
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


def _add_tracker_coverage_columns(conn: Connection) -> None:
    """covered_video, asked_count, last_source on learner_topic_strength (skipped when 0001 already created them)."""
    existing = {c["name"] for c in inspect(conn).get_columns("learner_topic_strength")}
    for name, ddl in (("covered_video", "BOOLEAN NOT NULL DEFAULT 0"), ("asked_count", "INTEGER NOT NULL DEFAULT 0"),
                      ("last_source", "VARCHAR(32)")):
        if name not in existing:
            conn.execute(text(f"ALTER TABLE learner_topic_strength ADD COLUMN {name} {ddl}"))


def _create_formative_pool(conn: Connection) -> None:
    from app.db.models.concept_tracker import FormativeItem

    FormativeItem.__table__.create(bind=conn, checkfirst=True)


def _create_parallel_test_items(conn: Connection) -> None:
    from app.db.models.parallel_test_item import ParallelTestItem

    ParallelTestItem.__table__.create(bind=conn, checkfirst=True)


MIGRATIONS: tuple[Migration, ...] = (
    Migration("0001_concept_tracker", "learner_topic_strength + concept_events (concept mastery tracker)",
              _create_concept_tracker),
    Migration("0002_tracker_coverage", "learner_topic_strength.covered_video, asked_count, last_source",
              _add_tracker_coverage_columns),
    Migration("0003_formative_pool", "formative_items (formative-check questions per concept, outside the bank)",
              _create_formative_pool),
    Migration("0004_parallel_test_items", "parallel_test_items (post-test questions paired with pre-test items)",
              _create_parallel_test_items),
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
