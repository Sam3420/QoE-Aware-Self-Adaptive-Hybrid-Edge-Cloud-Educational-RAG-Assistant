"""Pre-test and post-test/feedback nodes (interrupts), backed by Person B's assessment services."""
import logging
from typing import Any

from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from agent.deps import get_services
from agent.schemas import PostTestReply, PreTestReply
from agent.state import AgentState
from app.services.errors import AssistantError, QuestionBankNotFoundError

logger = logging.getLogger(__name__)


def _ask(prompt: dict[str, Any], model: type[BaseModel]) -> BaseModel:
    """Interrupt until the resume value validates against `model`, re-asking with the error message."""
    while True:
        raw = interrupt(prompt)
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            prompt = {**prompt, "error": exc.errors(include_url=False)[0]["msg"]}


def pre_test(state: AgentState) -> dict[str, Any]:
    """Show the topic's pre-test, wait for {"answers": [...]}, then score it and set competency."""
    app = get_services().app
    # Runs again on every resume; cheap after the first time because B caches the question bank.
    try:
        items = app.prepare_pretest(state["student_id"], state["topic_id"])
    except AssistantError as exc:  # e.g. MCQ generation failed: continue without a pre-test
        logger.warning("pre_test skipped: %s", exc)
        return {"competency": "beginner", "session_errors": [f"pre_test skipped: {exc}"]}

    reply = _ask({"type": "pre_test", "items": items,
                  "prompt": "Answer each question with A, B, C or D."}, PreTestReply)
    result = app.submit_pretest(state["student_id"], state["session_id"], state["topic_id"], reply.answers)
    return {"test_items": items, "pre_score": result["score"], "competency": result["competency"]}


def post_test_and_feedback(state: AgentState) -> dict[str, Any]:
    """Show the post-test, wait for {"answers", "rating", "comment"}, then score and store it with the feedback."""
    app = get_services().app
    errors: list[str] = []
    try:
        items = app.prepare_posttest(state["student_id"], state["topic_id"])
    except QuestionBankNotFoundError as exc:  # pre-test generation failed earlier: feedback only
        items = []
        errors.append(f"post_test skipped: {exc}")

    prompt = {"type": "post_test_and_feedback", "items": items,
              "prompt": "Answer the post-test, then rate this session from 1 to 5 and add a comment."}
    reply = _ask(prompt, PostTestReply)
    if not items:
        # Without a question bank, B's submit function cannot store the feedback either.
        return {"rating": reply.rating, "comment": reply.comment or "", "session_errors": errors}

    result = app.submit_posttest_and_feedback(state["student_id"], state["session_id"], state["topic_id"],
                                              reply.answers, reply.rating, reply.comment)
    return {"test_items": state["test_items"] + items, "post_score": result["score"],
            "rating": result["rating"], "comment": result["comment"] or ""}
