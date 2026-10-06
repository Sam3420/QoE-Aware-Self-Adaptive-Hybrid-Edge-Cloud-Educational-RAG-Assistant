"""Case-based runtime policy selection."""
import logging
from typing import Any

from agent.deps import get_services
from agent.policies import runtime_config_from_row
from agent.policy_selection import choose_policy
from agent.state import AgentState
from config import settings

logger = logging.getLogger(__name__)


def select_policy(state: AgentState) -> dict[str, Any]:
    """Pick the predefined policy with the highest expected utility among similar past sessions."""
    app = get_services().app
    policy_ids = [p["policy_id"] for p in app.list_policies()]
    default = app.get_default_policy()
    default_id = default["policy_id"] if default else settings.default_policy_id

    decision = choose_policy(
        state["context"],
        app.get_experiences(settings.experience_fetch_limit),
        policy_ids=policy_ids,
        default_policy_id=default_id,
        k=settings.knn_k,
        min_experiences=settings.min_experiences,
        epsilon=settings.epsilon,
        rng=get_services().rng,
        rtt_max_ms=settings.rtt_max_ms,
        default_last_qoe=settings.default_last_qoe,
    )
    row = app.get_policy(decision.policy_id)
    logger.info("select_policy %s", decision.reason)
    return {
        "policy_id": decision.policy_id,
        "runtime_config": runtime_config_from_row(row),
        "decision_reason": decision.reason,
        "similar_exps": [
            {"session_id": n.experience.get("session_id"), "policy_id": n.experience["policy_id"],
             "reward": n.experience["reward"], "distance": round(n.distance, 4)}
            for n in decision.neighbours
        ],
    }
