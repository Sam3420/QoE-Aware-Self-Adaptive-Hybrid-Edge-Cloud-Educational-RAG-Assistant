"""Pre-test service: what the agent's pre_test node calls.

The helpers below the public functions are shared with the post-test service.
"""

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
from app.core.config import get_settings
from app.services.errors import MCQGenerationError
from app.services.mcq import (
    KnowledgeRetriever,
    competency_from_score,
    extend_topic_items,
    get_or_create_topic_items,
    score_mcqs,
    session_scope,
)
from app.services.pretest_planning import PretestPlan, plan_pretest as plan_items

logger = get_logger(__name__)


def prepare_pretest(
    student_id: str,
    topic_id: str,
    *,
    db: Session | None = None,
    retriever: KnowledgeRetriever | None = None,
    llm: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    """Return this learner's pre-test items for display (see ``plan_pretest``)."""
    return plan_pretest(student_id, topic_id, db=db, retriever=retriever, llm=llm)["items"]


def plan_pretest(
    student_id: str,
    topic_id: str,
    *,
    db: Session | None = None,
    retriever: KnowledgeRetriever | None = None,
    llm: LLMProvider | None = None,
) -> dict[str, Any]:
    """Plan the learner's pre-test for a topic and return ``{mode, items, weak_concepts, previous, notes}``.

    A learner new to the topic gets the standard first questions of the bank (generated once per topic).
    A returning learner gets questions they missed last time plus unseen ones, preferring concepts they got
    wrong; if the bank has too few unseen questions, new ones focused on those concepts are generated and
    added first. Each item is ``{question_no, item_id, question, options, concept, reason}`` without the
    answer; ``item_id`` is the bank position to pass back to ``submit_pretest``.
    A failed top-up is not fatal: the plan falls back to review questions and says so in ``notes``.
    """
    settings = get_settings()
    notes: list[str] = []
    with session_scope(db) as session:
        bank_items = get_or_create_topic_items(topic_id, db=session, retriever=retriever, llm=llm)
        assessments = AdaptationRepository(session).list_assessments(student_id, topic_id)
        plan = _plan(bank_items, assessments, settings)
        if plan.mode == "returning" and plan.unseen_needed:
            try:
                added = extend_topic_items(topic_id, focus_concepts=plan.weak_concepts, db=session,
                                           retriever=retriever, llm=llm)
            except MCQGenerationError as exc:
                added = 0
                notes.append(f"Could not add new questions ({exc}); some questions are repeated for review.")
            if added:
                bank_items = AdaptationRepository(session).get_question_bank(topic_id)["items"]
                plan = _plan(bank_items, assessments, settings)
            elif not notes:
                notes.append("No new questions could be added to the bank; some questions are repeated for review.")

    items = strip_answers([bank_items[p.bank_index] for p in plan.items],
                          item_ids=[p.bank_index for p in plan.items], reasons=[p.reason for p in plan.items])
    logger.info("pretest_planned student_id=%s topic_id=%s mode=%s n_items=%s weak=%s",
                student_id, topic_id, plan.mode, len(items), plan.weak_concepts)
    return {"mode": plan.mode, "items": items, "weak_concepts": plan.weak_concepts,
            "previous": plan.previous, "notes": notes}


def _plan(bank_items: list[dict[str, Any]], assessments: list[dict[str, Any]], settings: Any) -> PretestPlan:
    return plan_items(bank_items, assessments, n=settings.mcq_n_items, retest_max=settings.mcq_retest_max)


def submit_pretest(
    student_id: str,
    session_id: str,
    topic_id: str,
    answers: Sequence[Any],
    *,
    item_ids: Sequence[int] | None = None,
    db: Session | None = None,
) -> dict[str, Any]:
    """Score the learner's pre-test answers, persist the assessment, and return
    ``{assessment_id, score, competency}`` (score is a 0-1 fraction).

    ``item_ids`` are the bank positions of the questions shown (from ``plan_pretest``); without them the
    first MCQ_N_ITEMS bank questions are assumed.

    Raises LearnerNotFoundError / LearningSessionNotFoundError if the learner or session
    does not exist (or the session belongs to another learner), and
    QuestionBankNotFoundError if prepare_pretest was never run for the topic.
    """
    validate_answers(answers)

    with session_scope(db) as session:
        ensure_learner_session(session, student_id, session_id)
        repository = AdaptationRepository(session)
        bank = repository.get_question_bank(topic_id)
        if bank is None:
            raise QuestionBankNotFoundError(
                f"No question bank for topic {topic_id!r}; call prepare_pretest first."
            )

        result = record_scored_assessment(
            repository,
            type=AssessmentType.PRE,
            student_id=student_id,
            session_id=session_id,
            topic_id=topic_id,
            items=select_bank_items(bank["items"], item_ids),
            answers=answers,
        )
        logger.info(
            "pretest_submitted student_id=%s session_id=%s topic_id=%s score=%.3f competency=%s",
            student_id,
            session_id,
            topic_id,
            result["score"],
            result["competency"],
        )
        return result


def validate_answers(answers: Sequence[Any]) -> None:
    """Raise TypeError unless ``answers`` is a list-like of per-item answers."""
    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
        raise TypeError("answers must be a list of letters 'A'-'D' or indices 0-3.")


def ensure_learner_session(session: Session, student_id: str, session_id: str) -> None:
    """Raise unless the learner exists and owns the learning session."""
    trace = LearningTraceRepository(session)
    if trace.get_learner(student_id) is None:
        raise LearnerNotFoundError(f"Learner {student_id!r} was not found.")
    learning_session = trace.get_learning_session(session_id)
    if learning_session is None or learning_session.learner_id != student_id:
        raise LearningSessionNotFoundError(
            f"Learning session {session_id!r} was not found for learner {student_id!r}."
        )


def select_bank_items(bank_items: Sequence[dict[str, Any]],
                      item_ids: Sequence[int] | None) -> list[dict[str, Any]]:
    """The bank items at ``item_ids`` (in that order), or the first MCQ_N_ITEMS when no ids are given."""
    if item_ids is None:
        return list(bank_items[: get_settings().mcq_n_items])
    invalid = [i for i in item_ids if not 0 <= i < len(bank_items)]
    if invalid:
        raise ValueError(f"Unknown question ids {invalid}; the bank has {len(bank_items)} questions.")
    return [bank_items[i] for i in item_ids]


def strip_answers(items: Sequence[dict[str, Any]], *, item_ids: Sequence[int] | None = None,
                  reasons: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Return items safe to show the learner: ``{question_no, question, options, concept}``, plus
    ``item_id`` and ``reason`` when given."""
    shown = []
    for number, item in enumerate(items, start=1):
        entry = {
            "question_no": number,
            "question": item["question"],
            "options": list(item["options"]),
            "concept": item["concept"],
        }
        if item_ids is not None:
            entry["item_id"] = item_ids[number - 1]
        if reasons is not None:
            entry["reason"] = reasons[number - 1]
        shown.append(entry)
    return shown


def record_scored_assessment(
    repository: AdaptationRepository,
    *,
    type: AssessmentType,
    student_id: str,
    session_id: str,
    topic_id: str,
    items: list[dict[str, Any]],
    answers: Sequence[Any],
) -> dict[str, Any]:
    """Score ``answers`` against ``items``, persist the assessment, and return
    ``{assessment_id, score, competency, correct, total}``."""
    score = score_mcqs(items, answers)
    competency = competency_from_score(score)
    assessment = repository.create_assessment(
        AssessmentCreate(
            session_id=session_id,
            student_id=student_id,
            topic_id=topic_id,
            type=type,
            items=items,
            answers=list(answers),
            score=score,
            competency=CompetencyLevel(competency),
        )
    )
    return {"assessment_id": assessment["id"], "score": score, "competency": competency,
            "correct": round(score * len(items)), "total": len(items)}
