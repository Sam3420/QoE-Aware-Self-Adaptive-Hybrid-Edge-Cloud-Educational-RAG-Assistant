"""Adaptive runtime policy selection (FAISS experience retrieval + estimated utility per action)."""
from typing import Any

from agent.deps import get_services
from agent.observability import log_event, observe
from agent.policies import runtime_config_from_row
from agent.policy_engine import PolicyEngine
from agent.state import AgentState
from config import settings


def select_policy(state: AgentState) -> dict[str, Any]:
    """Pick the predefined policy with the highest estimated utility for the current state."""
    services = get_services()
    app, store = services.app, services.experience_store()
    candidates = [p["policy_id"] for p in app.list_policies()]
    default = app.get_default_policy()
    default_id = default["policy_id"] if default else settings.default_policy_id
    prev_state, prev_record = store.previous_state(state["student_id"])

    with observe("select_policy", input={"context": state["context"], "candidates": candidates},
                 metadata={"session_id": state["session_id"]}) as span:
        decision = PolicyEngine(store, services.rng).select(state["context"], candidates, default_id)
        span.update(output={"policy_id": decision.action_id, "mode": decision.mode, "reason": decision.reason})
    log_event("policy_selected", session_id=state["session_id"], student_id=state["student_id"],
              policy_id=decision.action_id, mode=decision.mode, n_history=decision.n_history,
              estimates={a: e.estimated_value for a, e in decision.estimates.items()})

    row = app.get_policy(decision.action_id)
    return {
        "policy_id": decision.action_id,
        "runtime_config": runtime_config_from_row(row),
        "decision_reason": decision.reason,
        "candidate_utilities": {a: {"estimated": e.estimated_value, "support": e.support,
                                    "weight_sum": e.weight_sum, "target": decision.target}
                                for a, e in decision.estimates.items()},
        "similar_exps": decision.neighbours,
        "prev_state": prev_state.model_dump(mode="json"),
        "prev_state_at": prev_record["created_at"].isoformat() if prev_record else None,
    }
