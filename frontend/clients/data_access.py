"""Reads and writes against the app database (data/app.db) through the existing models and repositories.

Writes go through LearningTraceRepository / AdaptationRepository; the frontend adds only read
queries for listing what those repositories already store.
"""
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings as get_app_settings
from app.db.models import (
    Assessment,
    Experience,
    Interaction,
    LearnerProfile,
    LearningSession,
    Policy,
    RuntimeConfiguration,
    SessionFeedback,
    TopicQuestionBank,
)
from app.db.repositories import AdaptationRepository, LearningTraceRepository
from app.domain.enums import CompetencyLevel, InteractionStatus, PreferredLanguage
from app.domain.models import LearnerPersonalizationUpdate, LearningPreferences

COUNTED_TABLES = {
    "learners": LearnerProfile,
    "sessions": LearningSession,
    "interactions": Interaction,
    "assessments": Assessment,
    "question banks": TopicQuestionBank,
    "feedback": SessionFeedback,
    "experiences": Experience,
    "policies": Policy,
}


@contextmanager
def db_session() -> Iterator[Session]:
    """Open an app DB session; commit on success, roll back and re-raise on error."""
    from app.db.session import SessionLocal  # imported lazily: creates the app engine

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@dataclass(frozen=True)
class DatabaseStatus:
    ok: bool
    url: str
    counts: dict[str, int]
    error: str | None = None


def database_status() -> DatabaseStatus:
    """Run real queries: SELECT 1 plus a row count per table."""
    url = get_app_settings().resolved_database_url
    try:
        with db_session() as db:
            db.execute(text("SELECT 1"))
            counts = {name: db.scalar(select(func.count()).select_from(model)) or 0
                      for name, model in COUNTED_TABLES.items()}
        return DatabaseStatus(True, url, counts)
    except SQLAlchemyError as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else ""
        return DatabaseStatus(False, url, {}, f"{type(exc).__name__}: {first_line}")


# ---- Learners and personalization ------------------------------------------------------------

def _learner_dict(learner: LearnerProfile) -> dict[str, Any]:
    return {
        "id": learner.id,
        "display_name": learner.display_name,
        "preferred_language": learner.preferred_language,
        "competency_level": learner.competency_level,
        "interests": list(learner.interests or []),
        "learning_preferences": dict(learner.learning_preferences or {}),
        "created_at": learner.created_at,
    }


def list_learners() -> list[dict[str, Any]]:
    with db_session() as db:
        rows = db.scalars(select(LearnerProfile).order_by(LearnerProfile.created_at.desc()))
        return [_learner_dict(r) for r in rows]


def get_learner(learner_id: str) -> dict[str, Any] | None:
    with db_session() as db:
        learner = LearningTraceRepository(db).get_learner(learner_id)
        return _learner_dict(learner) if learner else None


def update_personalization(learner_id: str, *, language: str | None, competency: str | None,
                           interests: list[str], explanation_style: str | None,
                           detail_level: str | None) -> dict[str, Any] | None:
    """Persist the learner's Phase 3 personalization fields with the repository's own update."""
    data = LearnerPersonalizationUpdate(
        preferred_language=PreferredLanguage(language) if language else None,
        competency_level=CompetencyLevel(competency) if competency else None,
        interests=interests,
        learning_preferences=LearningPreferences(explanation_style=explanation_style or None,
                                                 detail_level=detail_level or None),
    )
    with db_session() as db:
        learner = LearningTraceRepository(db).update_learner_personalization(learner_id, data)
        return _learner_dict(learner) if learner else None


# ---- Sessions, interactions, assessments -----------------------------------------------------

def learner_sessions(learner_id: str) -> list[dict[str, Any]]:
    with db_session() as db:
        sessions = db.scalars(select(LearningSession).where(LearningSession.learner_id == learner_id)
                              .order_by(LearningSession.started_at.desc()))
        out = []
        for s in sessions:
            n_interactions = db.scalar(select(func.count()).select_from(Interaction)
                                       .where(Interaction.session_id == s.id)) or 0
            out.append({"id": s.id, "started_at": s.started_at, "status": s.status,
                        "api_interactions": n_interactions})
        return out


def session_exists(session_id: str, learner_id: str) -> bool:
    with db_session() as db:
        s = LearningTraceRepository(db).get_learning_session(session_id)
        return s is not None and s.learner_id == learner_id


def session_interactions(session_id: str) -> list[dict[str, Any]]:
    """Interactions the FastAPI assistant persisted for a session, oldest first, with their runtime config."""
    with db_session() as db:
        rows = db.scalars(select(Interaction).where(Interaction.session_id == session_id)
                          .order_by(Interaction.occurred_at))
        return [{
            "id": r.id, "occurred_at": r.occurred_at, "user_input": r.user_input,
            "assistant_output": r.assistant_output, "response_latency_ms": r.response_latency_ms,
            "model_provider": r.model_provider, "model_name": r.model_name, "status": r.status,
            "error_message": r.error_message, "runtime_configuration_id": r.runtime_configuration_id,
        } for r in rows]


def runtime_configuration(config_id: str) -> dict[str, Any] | None:
    with db_session() as db:
        rc = db.get(RuntimeConfiguration, config_id)
        if rc is None:
            return None
        return {"id": rc.id, "name": rc.name, "version": rc.version, "description": rc.description,
                "configuration_data": rc.configuration_data}


def interaction_stats() -> dict[str, Any]:
    """Aggregate the persisted API interactions (completed vs failed, mean latency of completed)."""
    with db_session() as db:
        total = db.scalar(select(func.count()).select_from(Interaction)) or 0
        completed = db.scalar(select(func.count()).select_from(Interaction)
                              .where(Interaction.status == InteractionStatus.COMPLETED.value)) or 0
        avg = db.scalar(select(func.avg(Interaction.response_latency_ms))
                        .where(Interaction.status == InteractionStatus.COMPLETED.value))
        return {"total": total, "completed": completed, "failed": total - completed,
                "avg_latency_ms": float(avg) if avg is not None else None}


def learner_assessments(learner_id: str) -> list[dict[str, Any]]:
    with db_session() as db:
        rows = db.scalars(select(Assessment).where(Assessment.student_id == learner_id)
                          .order_by(Assessment.created_at.desc()))
        return [{"created_at": a.created_at, "topic_id": a.topic_id, "type": a.type, "score": a.score,
                 "competency": a.competency, "session_id": a.session_id} for a in rows]


def session_feedback(session_id: str) -> list[dict[str, Any]]:
    with db_session() as db:
        return AdaptationRepository(db).list_session_feedback(session_id)


def experience_for_session(session_id: str) -> dict[str, Any] | None:
    with db_session() as db:
        return AdaptationRepository(db).get_experience_for_session(session_id)


def question_bank(topic_id: str) -> dict[str, Any] | None:
    """The topic's stored MCQ bank metadata (answers are dropped before returning)."""
    with db_session() as db:
        bank = AdaptationRepository(db).get_question_bank(topic_id)
    if bank is None:
        return None
    return {k: v for k, v in bank.items() if k != "items"}


# ---- Concept tracker (learner_topic_strength) ------------------------------------------------------------------

def concept_mastery(learner_id: str, topic_id: str) -> dict[str, Any]:
    """The learner's standing on every concept of a topic: the vocabulary's concepts (untested ones at the 0.5 prior)
    plus tracked concepts the vocabulary no longer lists. Returns {vocabulary_status, rows, error}."""
    from app.db.repositories import ConceptTrackerRepository
    from knowledge_prep.concepts import load_vocabulary

    vocabulary = load_vocabulary(topic_id)
    try:
        with db_session() as db:
            tracked = {r["concept_id"]: r for r in ConceptTrackerRepository(db).get_mastery(learner_id, topic_id)}
    except SQLAlchemyError as exc:  # tracker tables not migrated yet
        return {"vocabulary_status": vocabulary.status if vocabulary else None, "rows": [],
                "error": f"Concept tracker unavailable ({type(exc).__name__}); run scripts/migrate_db.py."}
    rows = []
    for c in (vocabulary.concepts if vocabulary else []):
        r = tracked.pop(c.id, None)
        rows.append({"concept_id": c.id, "concept": c.name, "description": c.description,
                     "mastery": r["mastery"] if r else 0.5, "attempts": r["attempts"] if r else 0,
                     "correct": r["correct"] if r else 0, "covered": r["covered_count"] if r else 0,
                     "last_attempt_at": r["last_attempt_at"] if r else None, "in_vocabulary": True})
    for concept_id, r in tracked.items():  # e.g. a concept renamed after it was tracked
        rows.append({"concept_id": concept_id, "concept": concept_id, "description": "Not in the current vocabulary.",
                     "mastery": r["mastery"], "attempts": r["attempts"], "correct": r["correct"],
                     "covered": r["covered_count"], "last_attempt_at": r["last_attempt_at"], "in_vocabulary": False})
    return {"vocabulary_status": vocabulary.status if vocabulary else None, "rows": rows, "error": None}


def session_concept_results(learner_id: str, topic_id: str, session_id: str) -> dict[str, dict[str, list[bool]]]:
    """concept id → {"pre-test": [correct, ...], "post-test": [...]} for the answers given in one session."""
    from app.db.repositories import ConceptTrackerRepository

    try:
        with db_session() as db:
            ids = {a["id"] for a in AdaptationRepository(db).list_assessments(learner_id, topic_id)
                   if a["session_id"] == session_id}
            events = ConceptTrackerRepository(db).list_events(learner_id, topic_id)
    except SQLAlchemyError:
        return {}
    out: dict[str, dict[str, list[bool]]] = {}
    for e in events:
        if e["kind"] == "attempt" and e["source_ref"].split(":", 1)[0] in ids:
            out.setdefault(e["concept_id"], {}).setdefault(e["source"], []).append(bool(e["correct"]))
    return out
