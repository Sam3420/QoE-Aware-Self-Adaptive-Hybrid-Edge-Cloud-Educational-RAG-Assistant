"""Post-test and feedback service: what the agent's post_test_and_feedback node calls.

The post-test reuses the topic question bank created for the pre-test and never
generates a new one.
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.repositories import AdaptationRepository
from app.domain.enums import AssessmentType
from app.domain.models import SessionFeedbackCreate
from app.services.errors import QuestionBankNotFoundError
from app.services.mcq import session_scope
from app.services.pretest import (
    ensure_learner_session,
    record_scored_assessment,
    strip_answers,
    validate_answers,
)

logger = get_logger(__name__)


def prepare_posttest(
    student_id: str,
    topic_id: str,
    *,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    """Return the topic's existing question-bank items for display, with ``answer_index`` removed.

    Each item is ``{question_no, question, options, concept}``, in the same order as the
    pre-test. Raises QuestionBankNotFoundError if the topic has no bank; nothing is generated.
    """
    with session_scope(db) as session:
        items = _get_bank_items(AdaptationRepository(session), topic_id)
    logger.info("posttest_prepared student_id=%s topic_id=%s n_items=%s", student_id, topic_id, len(items))
    return strip_answers(items)


def submit_posttest_and_feedback(
    student_id: str,
    session_id: str,
    topic_id: str,
    answers: Sequence[Any],
    rating: int | None = None,
    comment: str | None = None,
    *,
    db: Session | None = None,
) -> dict[str, Any]:
    """Score the post-test, persist it (``type="post"``) and the learner's feedback, and return
    ``{assessment_id, score, competency, rating, comment}`` (score is a 0-1 fraction).

    ``rating`` must be an integer 1-5 or None; ``comment`` may be None. Invalid input is
    rejected before anything is written. Raises LearnerNotFoundError /
    LearningSessionNotFoundError for an unknown learner or a session owned by someone else,
    and QuestionBankNotFoundError if the topic has no question bank.
    """
    validate_answers(answers)
    feedback = SessionFeedbackCreate(
        session_id=session_id,
        student_id=student_id,
        rating=rating,
        comment=comment,
    )

    with session_scope(db) as session:
        ensure_learner_session(session, student_id, session_id)
        repository = AdaptationRepository(session)
        items = _get_bank_items(repository, topic_id)

        result = record_scored_assessment(
            repository,
            type=AssessmentType.POST,
            student_id=student_id,
            session_id=session_id,
            topic_id=topic_id,
            items=items,
            answers=answers,
        )
        stored_feedback = repository.create_session_feedback(feedback)
        logger.info(
            "posttest_submitted student_id=%s session_id=%s topic_id=%s score=%.3f competency=%s rating=%s",
            student_id,
            session_id,
            topic_id,
            result["score"],
            result["competency"],
            stored_feedback["rating"],
        )
        return {
            **result,
            "rating": stored_feedback["rating"],
            "comment": stored_feedback["comment"],
        }


def _get_bank_items(repository: AdaptationRepository, topic_id: str) -> list[dict[str, Any]]:
    bank = repository.get_question_bank(topic_id)
    if bank is None:
        raise QuestionBankNotFoundError(
            f"No question bank for topic {topic_id!r}; the post-test reuses the pre-test items."
        )
    return bank["items"]
