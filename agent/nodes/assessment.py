"""Pre-test and post-test/feedback nodes (interrupts), backed by Person B's assessment services."""
import logging
from typing import Any

from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from agent.deps import get_services
from agent.knowledge import KnowledgeUnavailable, ensure_topic_index
from agent.schemas import PostTestReply, PreTestReply
from agent.state import AgentState
from app.services.errors import AssistantError, QuestionBankNotFoundError
from services.vector_store import EmbeddingError
from telemetry import tracing

logger = logging.getLogger(__name__)


def _ask(prompt: dict[str, Any], model: type[BaseModel]) -> BaseModel:
    """Interrupt until the resume value validates against `model`, re-asking with the error message."""
    while True:
        raw = interrupt(prompt)
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            prompt = {**prompt, "error": exc.errors(include_url=False)[0]["msg"]}


def plan_pretest(state: AgentState) -> dict[str, Any]:
    """Choose this session's pre-test questions from the learner's history (generating questions if needed).

    A separate node so the questions are fixed in the checkpoint before pre_test interrupts: pre_test
    re-runs on every resume and must show and score exactly the same items.
    """
    services = get_services()
    try:
        ensure_topic_index(services.crag.store, state["topic_id"])
        plan = services.app.plan_pretest(state["student_id"], state["topic_id"])
    except (AssistantError, KnowledgeUnavailable, EmbeddingError) as exc:  # continue without a pre-test
        logger.warning("pre_test skipped: %s", exc)
        return {"competency": "beginner", "pretest_plan": {"mode": "unavailable", "reason": str(exc)},
                "session_errors": [f"pre_test skipped: {exc}"]}
    meta = {k: v for k, v in plan.items() if k != "items"}
    if meta.get("previous") and meta["previous"].get("date") is not None:
        meta["previous"] = {**meta["previous"], "date": str(meta["previous"]["date"])}
    return {"test_items": plan["items"], "pretest_plan": meta,
            "session_errors": [f"pre_test: {note}" for note in plan["notes"]]}


def pre_test(state: AgentState) -> dict[str, Any]:
    """Show the planned pre-test, wait for {"answers": [...]}, then score it and set competency."""
    items = state["test_items"]
    if not items:  # planning failed; plan_pretest already recorded why
        return {}
    reply = _ask({"type": "pre_test", "items": items, "plan": state["pretest_plan"],
                  "prompt": "Answer each question with A, B, C or D."}, PreTestReply)
    result = get_services().app.submit_pretest(state["student_id"], state["session_id"], state["topic_id"],
                                               reply.answers, item_ids=_item_ids(items))
    pre_result = {"correct": result["correct"], "total": result["total"], "competency": result["competency"],
                  "answers": [_letter(a) for a in reply.answers]}
    return {"pre_score": result["score"], "competency": result["competency"], "pre_result": pre_result}


def _letter(answer: str | int) -> str:
    return "ABCD"[answer] if isinstance(answer, int) else str(answer).strip().upper()


def _item_ids(items: list[dict[str, Any]]) -> list[int] | None:
    ids = [item.get("item_id") for item in items]
    return None if None in ids else ids


def post_test_and_feedback(state: AgentState) -> dict[str, Any]:
    """Show the post-test, wait for {"answers", "rating", "comment"}, then score and store it with the feedback."""
    app = get_services().app
    errors: list[str] = []
    item_ids = _item_ids(state["test_items"]) if state["test_items"] else None
    try:
        items = app.prepare_posttest(state["student_id"], state["topic_id"], item_ids=item_ids)
    except QuestionBankNotFoundError as exc:  # pre-test generation failed earlier: feedback only
        items = []
        errors.append(f"post_test skipped: {exc}")

    prompt = {"type": "post_test_and_feedback", "items": items,
              "prompt": "Answer the post-test, then rate this session from 1 to 5 and add a comment."}
    reply = _ask(prompt, PostTestReply)
    if not items:
        # Without a question bank, B's submit function cannot store the feedback either.
        _score_feedback(state["session_id"], reply.rating, reply.comment, None)
        return {"rating": reply.rating, "comment": reply.comment or "", "session_errors": errors}

    result = app.submit_posttest_and_feedback(state["student_id"], state["session_id"], state["topic_id"],
                                              reply.answers, reply.rating, reply.comment, item_ids=item_ids)
    _score_feedback(state["session_id"], reply.rating, reply.comment, result["score"])
    # Only now, after scoring, are the answers revealed (with explanations written for any question without one).
    review = app.review_posttest(state["topic_id"], reply.answers, item_ids=item_ids,
                                 pre_answers=(state.get("pre_result") or {}).get("answers"))
    if review["note"]:
        errors.append(f"post_test review: {review['note']}")
    run_id = (state.get("recommendations") or {}).get("run_id")
    if run_id:  # the recommendation run's outcome includes the pre/post gain on this topic
        try:
            get_services().video_recommender().evaluate(run_id)
        except Exception as exc:
            errors.append(f"recommendation outcome: {type(exc).__name__}: {exc}")
    return {"post_score": result["score"], "post_review": {**review, "correct": result["correct"],
                                                           "total": result["total"]},
            "rating": result["rating"], "comment": result["comment"] or "", "session_errors": errors}


def _score_feedback(session_id: str, rating: int | None, comment: str | None, post_score: float | None) -> None:
    """Learner feedback and the post-test result as Langfuse session scores (named after the signal)."""
    tracing.score(name="session_rating", value=rating, session_id=session_id, data_type="NUMERIC", comment=comment)
    tracing.score(name="posttest_score", value=post_score, session_id=session_id, data_type="NUMERIC")
