"""Concept tracker API (the contract the coaching nodes code against).

    get_concepts(topic_id) -> list[Concept]
    record_attempt(learner_id, topic_id, concept_id, correct, source) -> None     source: pretest | formative | posttest
    mark_covered(learner_id, topic_id, concept_id, kind) -> None                 kind: video | asked
    get_mastery(learner_id, topic_id) -> dict[str, ConceptState]                 every concept of the vocabulary

Built on the existing tracker (docs/concept_tracker.md): vocabularies in knowledge_prep/concepts/<topic_id>.json, rows
in learner_topic_strength, raw observations in concept_events. mastery = (correct + 1) / (attempts + 2); 0.5 with no
attempts. Coverage never changes mastery.

Idempotency (the assumption callers rely on): pass ``source_ref``, a stable id of the observation, and recording it
again is a no-op, whichever node or script does it.
- A graded answer: ``"<assessment id>:<item index>"`` for tests (the format the live test recording and both backfills
  use), e.g. ``"<check id>:<n>"`` for formative checks. One answer counts once, on one concept: if that answer was
  already recorded under another concept, the call is ignored (and logged).
- Coverage: e.g. ``"<session id>:<video id>"`` for a video, ``"<session id>:q<question no>"`` for a question.
Without ``source_ref`` every call counts (a unique id is generated), so a node that may run twice must pass one.

Every function takes optional ``db`` (a SQLAlchemy Session; default: a new session on the app DB, committed on
success) and record_attempt takes ``occurred_at``; both are keyword-only, so the agreed positional signatures hold.
"""
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ConceptEvent, LearnerTopicStrength
from app.db.repositories.concept_tracker import ConceptTrackerRepository, laplace_mastery
from knowledge_prep.concepts import Concept, Vocabulary, load_vocabulary

logger = logging.getLogger(__name__)

# Contract names -> names stored in concept_events (shared with the live test recording in agent/concept_tracking.py,
# so the same answer recorded by either path is recognised as one observation).
ATTEMPT_SOURCES = {"pretest": "pre-test", "formative": "formative", "posttest": "post-test"}
COVERAGE_SOURCES = {"video": "video", "asked": "question"}

__all__ = ["Concept", "ConceptState", "UnknownConceptError", "VocabularyNotFoundError", "get_concepts",
           "record_attempt", "mark_covered", "get_mastery"]


class VocabularyNotFoundError(LookupError):
    """The topic has no concept vocabulary (knowledge_prep/concepts/<topic_id>.json)."""


class UnknownConceptError(ValueError):
    """The concept id is not in the topic's vocabulary."""


class ConceptState(BaseModel):
    """A learner's standing on one concept."""

    mastery: float  # (correct + 1) / (attempts + 2)
    attempts: int
    correct: int
    covered: bool  # a video covered it
    asked: int  # questions the learner asked about it


def get_concepts(topic_id: str) -> list[Concept]:
    """The topic's concepts in vocabulary order. Raises VocabularyNotFoundError if it has none."""
    return list(_vocabulary(topic_id).concepts)


def record_attempt(learner_id: str, topic_id: str, concept_id: str, correct: bool, source: str, *,
                   source_ref: str | None = None, occurred_at: datetime | None = None,
                   db: Session | None = None) -> None:
    """Count one graded answer on a concept (updates mastery). See the module docstring for ``source_ref``."""
    stored_source = _check(ATTEMPT_SOURCES, source, "source")
    _check_concept(topic_id, concept_id)
    ref = source_ref or f"auto:{uuid.uuid4().hex}"
    with _session(db) as session:
        earlier = session.scalars(select(ConceptEvent.concept_id).where(
            ConceptEvent.kind == "attempt", ConceptEvent.source == stored_source,
            ConceptEvent.source_ref == ref)).first()
        if earlier is not None:
            if earlier != concept_id:
                logger.warning("answer %s/%s already recorded on %r; not counted again on %r",
                               stored_source, ref, earlier, concept_id)
            return
        ConceptTrackerRepository(session).record_attempt(learner_id, topic_id, concept_id, correct,
                                                         source=stored_source, source_ref=ref,
                                                         occurred_at=occurred_at)
        _row(session, learner_id, topic_id, concept_id).last_source = stored_source


def mark_covered(learner_id: str, topic_id: str, concept_id: str, kind: str, *, source_ref: str | None = None,
                 db: Session | None = None) -> None:
    """Record an exposure: ``video`` sets covered, ``asked`` adds one to asked. Mastery is not changed."""
    stored_source = _check(COVERAGE_SOURCES, kind, "kind")
    _check_concept(topic_id, concept_id)
    ref = source_ref or f"auto:{uuid.uuid4().hex}"
    with _session(db) as session:
        repo = ConceptTrackerRepository(session)
        before = repo.get_mastery(learner_id, topic_id, concept_id)["covered_count"]
        after = repo.mark_covered(learner_id, topic_id, concept_id, source=stored_source, source_ref=ref)
        if after["covered_count"] == before:  # this exposure was recorded before
            return
        row = _row(session, learner_id, topic_id, concept_id)
        if kind == "video":
            row.covered_video = True
        else:
            row.asked_count += 1


def get_mastery(learner_id: str, topic_id: str, *, db: Session | None = None) -> dict[str, ConceptState]:
    """Every concept of the topic's vocabulary (in vocabulary order); concepts without data get the 0.5 prior."""
    concepts = get_concepts(topic_id)
    with _session(db) as session:
        rows = {r.concept_id: r for r in session.scalars(select(LearnerTopicStrength).where(
            LearnerTopicStrength.student_id == learner_id, LearnerTopicStrength.topic_id == topic_id))}
        out = {}
        for concept in concepts:
            row = rows.get(concept.id)
            out[concept.id] = ConceptState(
                mastery=laplace_mastery(row.correct, row.attempts) if row else laplace_mastery(0, 0),
                attempts=row.attempts if row else 0, correct=row.correct if row else 0,
                covered=bool(row.covered_video) if row else False, asked=row.asked_count if row else 0)
        return out


# --- internals -----------------------------------------------------------------------------------------------------

def _vocabulary(topic_id: str) -> Vocabulary:
    vocabulary = load_vocabulary(topic_id)
    if vocabulary is None:
        raise VocabularyNotFoundError(f"Topic {topic_id!r} has no concept vocabulary "
                                      f"(knowledge_prep/concepts/{topic_id}.json).")
    return vocabulary


def _check(allowed: dict[str, str], value: str, what: str) -> str:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {sorted(allowed)}, got {value!r}")
    return allowed[value]


def _check_concept(topic_id: str, concept_id: str) -> None:
    vocabulary = _vocabulary(topic_id)
    if vocabulary.get(concept_id) is None:
        raise UnknownConceptError(f"{concept_id!r} is not a concept of {topic_id!r}; known: "
                                  f"{[c.id for c in vocabulary.concepts]}")


def _row(session: Session, learner_id: str, topic_id: str, concept_id: str) -> LearnerTopicStrength:
    session.flush()
    return session.scalars(select(LearnerTopicStrength).where(
        LearnerTopicStrength.student_id == learner_id, LearnerTopicStrength.topic_id == topic_id,
        LearnerTopicStrength.concept_id == concept_id)).one()


@contextmanager
def _session(db: Session | None) -> Iterator[Session]:
    """The caller's session (left to the caller to commit), or a new one committed on success."""
    if db is not None:
        yield db
        db.flush()
        return
    from app.db.session import SessionLocal  # imported lazily: creates the app DB engine

    with SessionLocal() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
