"""Pre-test service: what the agent's pre_test node calls. Pre-test only."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.repositories import AdaptationRepository, LearningTraceRepository
from app.domain.enums import AssessmentType, CompetencyLevel
from app.domain.models import AssessmentCreate
from app.llm.provider import LLMProvider
from app.services.errors import (
    LearnerNotFoundError,
    LearningSessionNotFoundError,
    QuestionBankNotFoundError,
)
from app.services.mcq import (
    KnowledgeRetriever,
    competency_from_score,
    get_or_create_topic_items,
    score_mcqs,
    session_scope,
)

logger = get_logger(__name__)


def prepare_pretest(
    student_id: str,
    topic_id: str,
    *,
    db: Session | None = None,
    retriever: KnowledgeRetriever | None = None,
    llm: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    """Return the topic's pre-test items for display, with ``answer_index`` removed.

    Each item is ``{question_no, question, options, concept}``; answer with letters
    ``"A"``-``"D"`` in the same order. Items are shared per topic, so ``student_id``
    is accepted for the node's interface but does not change the items.
    """
    items = get_or_create_topic_items(topic_id, db=db, retriever=retriever, llm=llm)
    logger.info("pretest_prepared student_id=%s topic_id=%s n_items=%s", student_id, topic_id, len(items))
    return [
        {
            "question_no": number,
            "question": item["question"],
            "options": list(item["options"]),
            "concept": item["concept"],
        }
        for number, item in enumerate(items, start=1)
    ]


def submit_pretest(
    student_id: str,
    session_id: str,
    topic_id: str,
    answers: Sequence[Any],
    *,
    db: Session | None = None,
) -> dict[str, Any]:
    """Score the learner's pre-test answers, persist the assessment, and return
    ``{assessment_id, score, competency}`` (score is a 0-1 fraction).

    Raises LearnerNotFoundError / LearningSessionNotFoundError if the learner or session
    does not exist (or the session belongs to another learner), and
    QuestionBankNotFoundError if prepare_pretest was never run for the topic.
    """
    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
        raise TypeError("answers must be a list of letters 'A'-'D' or indices 0-3.")

    with session_scope(db) as session:
        trace = LearningTraceRepository(session)
        if trace.get_learner(student_id) is None:
            raise LearnerNotFoundError(f"Learner {student_id!r} was not found.")
        learning_session = trace.get_learning_session(session_id)
        if learning_session is None or learning_session.learner_id != student_id:
            raise LearningSessionNotFoundError(
                f"Learning session {session_id!r} was not found for learner {student_id!r}."
            )

        repository = AdaptationRepository(session)
        bank = repository.get_question_bank(topic_id)
        if bank is None:
            raise QuestionBankNotFoundError(
                f"No question bank for topic {topic_id!r}; call prepare_pretest first."
            )

        score = score_mcqs(bank["items"], answers)
        competency = competency_from_score(score)
        assessment = repository.create_assessment(
            AssessmentCreate(
                session_id=session_id,
                student_id=student_id,
                topic_id=topic_id,
                type=AssessmentType.PRE,
                items=bank["items"],
                answers=list(answers),
                score=score,
                competency=CompetencyLevel(competency),
            )
        )
        logger.info(
            "pretest_submitted student_id=%s session_id=%s topic_id=%s score=%.3f competency=%s",
            student_id,
            session_id,
            topic_id,
            score,
            competency,
        )
        return {"assessment_id": assessment["id"], "score": score, "competency": competency}
