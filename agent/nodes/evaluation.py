"""Session evaluation and experience storage."""
import logging
from typing import Any

from agent.deps import get_services
from agent.interfaces import evaluate_session_stub
from agent.state import AgentState

logger = logging.getLogger(__name__)


def evaluate_session(state: AgentState) -> dict[str, Any]:
    """Score the session (QoE and SRE before/after, reward) — STUB until Person B's metrics exist."""
    evaluation = evaluate_session_stub(state["qa_log"], state["rating"], state["pre_score"], state["post_score"],
                                       state["context"], state["runtime_config"])
    return {"evaluation": evaluation, "qoe": evaluation["qoe_after"], "sre": evaluation["sre_after"],
            "reward": evaluation["reward"]}


def store_experience(state: AgentState) -> dict[str, Any]:
    """Save (context, policy, evaluation) through Person B's store_experience for future policy selection."""
    try:
        get_services().app.store_experience(state["student_id"], state["session_id"], state["topic_id"],
                                            state["policy_id"], state["context"], state["evaluation"])
    except Exception as exc:  # a storage failure must not lose the learner's finished session
        logger.error("store_experience failed: %s", exc)
        return {"session_errors": [f"store_experience failed: {exc}"]}
    return {}
