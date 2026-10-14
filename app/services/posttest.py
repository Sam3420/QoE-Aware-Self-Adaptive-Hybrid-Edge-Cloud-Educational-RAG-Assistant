"""Post-test and feedback service: what the agent's post_test_and_feedback node calls.

The post-test asks the same questions as the session's pre-test (passed as bank ``item_ids``),
so the pre/post difference measures learning on identical items. It never generates questions.
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.repositories import AdaptationRepository
from app.domain.enums import AssessmentType
from app.domain.models import SessionFeedbackCreate
from app.llm.provider import LLMProvider
from app.services.errors import AssistantError, QuestionBankNotFoundError
from app.services.mcq import ANSWER_LETTERS, KnowledgeRetriever, explain_items, normalize_answer, session_scope
from app.services.pretest import (
    ensure_learner_session,
    record_scored_assessment,
    select_bank_items,
    strip_answers,
    validate_answers,
)

logger = get_logger(__name__)


def prepare_posttest(
    student_id: str,
    topic_id: str,
    *,
    item_ids: Sequence[int] | None = None,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    """Return the post-test items for display, with ``answer_index`` removed.

    ``item_ids`` are the bank positions used in this session's pre-test (same order); without them the
    first MCQ_N_ITEMS bank questions are used. Each item is ``{question_no, question, options, concept}``
    (plus ``item_id`` when ids are given). Raises QuestionBankNotFoundError if the topic has no bank.
    """
    with session_scope(db) as session:
        items = select_bank_items(_get_bank_items(AdaptationRepository(session), topic_id), item_ids)
    logger.info("posttest_prepared student_id=%s topic_id=%s n_items=%s", student_id, topic_id, len(items))
    return strip_answers(items, item_ids=list(item_ids) if item_ids is not None else None)


def submit_posttest_and_feedback(
    student_id: str,
    session_id: str,
    topic_id: str,
    answers: Sequence[Any],
    rating: int | None = None,
    comment: str | None = None,
    *,
    item_ids: Sequence[int] | None = None,
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
        items = select_bank_items(_get_bank_items(repository, topic_id), item_ids)

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


def review_posttest(
    topic_id: str,
    post_answers: Sequence[Any],
    *,
    item_ids: Sequence[int] | None = None,
    pre_answers: Sequence[Any] | None = None,
    db: Session | None = None,
    retriever: KnowledgeRetriever | None = None,
    llm: LLMProvider | None = None,
) -> dict[str, Any]:
    """Answer key for a finished post-test: ``{items, note}``.

    Each item is ``{question_no, question, options, concept, answer_index, answer, explanation,
    post_answer, post_correct, pre_answer, pre_correct}`` (answers as letters, None when not given).
    Call it only after the post-test is submitted: it reveals the answers. Questions without a stored
    explanation get one written by the LLM first (see ``mcq.explain_items``); if that fails, ``note`` says
    why and those items have an empty explanation.
    """
    note = None
    with session_scope(db) as session:
        ids = list(item_ids) if item_ids is not None else list(range(len(select_bank_items(
            _get_bank_items(AdaptationRepository(session), topic_id), None))))
        try:
            explain_items(topic_id, ids, db=session, retriever=retriever, llm=llm)
        except AssistantError as exc:
            note = f"Some explanations could not be written: {exc}"
            logger.warning("posttest_review_explanations_failed topic_id=%s reason=%s", topic_id, exc)
        items = select_bank_items(_get_bank_items(AdaptationRepository(session), topic_id), ids)

    def letter(answers: Sequence[Any] | None, index: int) -> str | None:
        if answers is None or index >= len(answers):
            return None
        chosen = normalize_answer(answers[index])
        return None if chosen is None else ANSWER_LETTERS[chosen]

    review = []
    for index, item in enumerate(items):
        answer = ANSWER_LETTERS[item["answer_index"]]
        post, pre = letter(post_answers, index), letter(pre_answers, index)
        review.append({
            "question_no": index + 1, "question": item["question"], "options": list(item["options"]),
            "concept": item["concept"], "answer_index": item["answer_index"], "answer": answer,
            "explanation": str(item.get("explanation") or ""),
            "post_answer": post, "post_correct": post == answer,
            "pre_answer": pre, "pre_correct": None if pre_answers is None else pre == answer,
        })
    return {"items": review, "note": note}


def _get_bank_items(repository: AdaptationRepository, topic_id: str) -> list[dict[str, Any]]:
    bank = repository.get_question_bank(topic_id)
    if bank is None:
        raise QuestionBankNotFoundError(
            f"No question bank for topic {topic_id!r}; the post-test reuses the pre-test items."
        )
    return bank["items"]
