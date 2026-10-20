"""Concept tracker: per-learner, per-concept attempts, mastery and coverage (LearnerTopicStrength).

    record_attempt(student, topic, concept, correct, source=..., source_ref=...)  graded answer → mastery
    mark_covered(student, topic, concept, source=..., source_ref=...)              exposure, no effect on mastery
    get_mastery(student, topic[, concept])                                         current standing

mastery = (correct + 1) / (attempts + 2): the Laplace (rule-of-succession) estimate, 0.5 with no evidence, moving
towards the observed success rate as evidence grows. Every observation is stored once in `concept_events`
(unique per origin), so re-recording it (a re-run backfill, a retried graph node) is a no-op.
"""
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import utc_now
from app.db.errors import ReferencedRecordNotFoundError
from app.db.models import ConceptEvent, LearnerProfile, LearnerTopicStrength


def laplace_mastery(correct: int, attempts: int) -> float:
    """(correct + 1) / (attempts + 2)."""
    if attempts < 0 or not 0 <= correct <= attempts:
        raise ValueError(f"need 0 <= correct <= attempts, got correct={correct}, attempts={attempts}")
    return (correct + 1) / (attempts + 2)


class ConceptTrackerRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def record_attempt(self, student_id: str, topic_id: str, concept_id: str, correct: bool, *, source: str,
                       source_ref: str, occurred_at: datetime | None = None) -> dict[str, Any]:
        """Count one graded answer on a concept and return the updated standing (unchanged if already recorded)."""
        if not self._new_event("attempt", student_id, topic_id, concept_id, source, source_ref, occurred_at,
                               correct=bool(correct)):
            return self.get_mastery(student_id, topic_id, concept_id)
        row = self._row(student_id, topic_id, concept_id)
        row.attempts += 1
        row.correct += int(bool(correct))
        row.mastery = laplace_mastery(row.correct, row.attempts)
        at = occurred_at or utc_now()
        row.last_attempt_at = at if row.last_attempt_at is None else max(_naive(row.last_attempt_at), _naive(at))
        self.session.flush()
        return _to_dict(row)

    def mark_covered(self, student_id: str, topic_id: str, concept_id: str, *, source: str, source_ref: str,
                     occurred_at: datetime | None = None) -> dict[str, Any]:
        """Count one exposure to a concept (a question about it, a video covering it); mastery is not changed."""
        if not self._new_event("covered", student_id, topic_id, concept_id, source, source_ref, occurred_at):
            return self.get_mastery(student_id, topic_id, concept_id)
        row = self._row(student_id, topic_id, concept_id)
        at = _naive(occurred_at or utc_now())
        row.covered_count += 1
        row.first_covered_at = at if row.first_covered_at is None else min(_naive(row.first_covered_at), at)
        row.last_covered_at = at if row.last_covered_at is None else max(_naive(row.last_covered_at), at)
        self.session.flush()
        return _to_dict(row)

    def get_mastery(self, student_id: str, topic_id: str, concept_id: str | None = None
                    ) -> dict[str, Any] | list[dict[str, Any]]:
        """One concept's standing (the 0.5 prior with zero counts if never seen), or all tracked concepts of the
        topic ordered by concept id when `concept_id` is None."""
        if concept_id is not None:
            row = self._find(student_id, topic_id, concept_id)
            return _to_dict(row) if row is not None else _prior(student_id, topic_id, concept_id)
        rows = self.session.scalars(
            select(LearnerTopicStrength)
            .where(LearnerTopicStrength.student_id == student_id, LearnerTopicStrength.topic_id == topic_id)
            .order_by(LearnerTopicStrength.concept_id))
        return [_to_dict(r) for r in rows]

    def list_events(self, student_id: str, topic_id: str) -> list[dict[str, Any]]:
        """Raw observations, oldest first (for audits and the backfill report)."""
        rows = self.session.scalars(
            select(ConceptEvent)
            .where(ConceptEvent.student_id == student_id, ConceptEvent.topic_id == topic_id)
            .order_by(ConceptEvent.occurred_at, ConceptEvent.created_at))
        return [{"concept_id": e.concept_id, "kind": e.kind, "correct": e.correct, "source": e.source,
                 "source_ref": e.source_ref, "occurred_at": e.occurred_at} for e in rows]

    # internals

    def _new_event(self, kind: str, student_id: str, topic_id: str, concept_id: str, source: str, source_ref: str,
                   occurred_at: datetime | None, correct: bool | None = None) -> bool:
        """Store the observation; False if this exact observation was recorded before."""
        if self.session.get(LearnerProfile, student_id) is None:
            raise ReferencedRecordNotFoundError(f"Learner {student_id!r} was not found.")
        exists = self.session.scalars(select(ConceptEvent.id).where(
            ConceptEvent.kind == kind, ConceptEvent.source == source, ConceptEvent.source_ref == source_ref,
            ConceptEvent.concept_id == concept_id)).first()
        if exists is not None:
            return False
        self.session.add(ConceptEvent(student_id=student_id, topic_id=topic_id, concept_id=concept_id, kind=kind,
                                      correct=correct, source=source, source_ref=source_ref,
                                      occurred_at=occurred_at or utc_now()))
        return True

    def _find(self, student_id: str, topic_id: str, concept_id: str) -> LearnerTopicStrength | None:
        return self.session.scalars(select(LearnerTopicStrength).where(
            LearnerTopicStrength.student_id == student_id, LearnerTopicStrength.topic_id == topic_id,
            LearnerTopicStrength.concept_id == concept_id)).first()

    def _row(self, student_id: str, topic_id: str, concept_id: str) -> LearnerTopicStrength:
        row = self._find(student_id, topic_id, concept_id)
        if row is None:
            row = LearnerTopicStrength(student_id=student_id, topic_id=topic_id, concept_id=concept_id, attempts=0,
                                       correct=0, mastery=laplace_mastery(0, 0), covered_count=0)
            self.session.add(row)
        return row


def _naive(value: datetime) -> datetime:
    """SQLite returns naive datetimes; compare everything as naive UTC."""
    return value.replace(tzinfo=None)


def _prior(student_id: str, topic_id: str, concept_id: str) -> dict[str, Any]:
    return {"student_id": student_id, "topic_id": topic_id, "concept_id": concept_id, "attempts": 0, "correct": 0,
            "mastery": laplace_mastery(0, 0), "covered_count": 0, "covered": False, "first_covered_at": None,
            "last_covered_at": None, "last_attempt_at": None, "tracked": False}


def _to_dict(row: LearnerTopicStrength) -> dict[str, Any]:
    return {"student_id": row.student_id, "topic_id": row.topic_id, "concept_id": row.concept_id,
            "attempts": row.attempts, "correct": row.correct, "mastery": row.mastery,
            "covered_count": row.covered_count, "covered": row.covered_count > 0,
            "first_covered_at": row.first_covered_at, "last_covered_at": row.last_covered_at,
            "last_attempt_at": row.last_attempt_at, "tracked": True}
