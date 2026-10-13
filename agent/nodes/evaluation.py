"""Session evaluation (QoE, SRE, utility, reward) and experience storage."""
import logging
from datetime import datetime
from typing import Any

from agent.deps import get_services
from agent.observability import log_event, observe
from agent.session_metrics import build_next_state, sample_resources
from agent.state import AgentState
from config import settings
from metrics.reward import AdaptationAction, IncompleteTransition, RewardResult, RuntimeState, compute_reward

logger = logging.getLogger(__name__)


def evaluate_session(state: AgentState) -> dict[str, Any]:
    """Measure s_next, score s and s_next, and compute R = μ1·ΔQoE + μ2·ΔSRE for the executed policy."""
    services = get_services()
    store = services.experience_store()
    if not state["qa_log"]:  # the policy never ran: there is no transition to evaluate
        return {"evaluation": {"status": "not_executed",
                               "reason": "no question was answered, so the selected policy was never applied"}}

    since = datetime.fromisoformat(state["prev_state_at"]) if state.get("prev_state_at") else None
    end = sample_resources(services.monitor, settings)
    s_next = build_next_state(state, end=end, session_factory=services.app.session_factory, cfg=settings,
                              since=since)
    s = RuntimeState.model_validate(state["prev_state"]) if state.get("prev_state") else RuntimeState(
        context={}, baseline=True)
    action = AdaptationAction(action_id=state["policy_id"], config=state["runtime_config"])
    weights = store.weights()
    with observe("evaluate_session", input={"policy_id": action.action_id},
                 metadata={"session_id": state["session_id"]}) as span:
        try:
            result = compute_reward(s, s_next, action, weights, policy=settings.metric_missing_policy,
                                    baseline_qoe=settings.baseline_qoe, baseline_sre=settings.baseline_sre)
        except IncompleteTransition as exc:
            span.update(output={"status": "incomplete", "reason": str(exc)})
            return {"evaluation": {"status": "incomplete", "reason": str(exc)},
                    "session_errors": [f"evaluation: {exc}"]}
        span.update(output={"reward": result.reward, "qoe": result.after.qoe, "sre": result.after.sre})
        for name, value in (("qoe", result.after.qoe), ("sre", result.after.sre), ("reward", result.reward)):
            try:
                span.score(name=name, value=value)
            except Exception:  # tracing must never break evaluation
                pass

    predicted = (state.get("candidate_utilities") or {}).get(action.action_id, {}).get("estimated")
    qoe_detail = result.after.qoe_detail
    evaluation = {
        "status": "completed",
        "qoe_before": result.before.qoe, "qoe_after": result.after.qoe,
        "sre_before": result.before.sre, "sre_after": result.after.sre,
        "utility_before": result.before.utility, "utility_after": result.after.utility,
        "predicted_utility": predicted,  # estimate made before execution, kept separate from the observation
        "delta_qoe": result.delta_qoe, "delta_sre": result.delta_sre, "reward": result.reward,
        "baseline_used": result.baseline_used, "weights_version": result.weights_version,
        "qoe_components": qoe_detail.components if qoe_detail else {},
        "qoe_coverage": qoe_detail.coverage if qoe_detail else None,
        "metrics": s_next.qoe_metrics.model_dump() if s_next.qoe_metrics else {},
        "sre_components": result.after.sre_detail.components if result.after.sre_detail else {},
        "sre_notes": result.after.sre_detail.notes if result.after.sre_detail else [],
        "result": result.model_dump(mode="json"),
        "next_state": s_next.model_dump(mode="json"),
    }
    log_event("reward_computed", session_id=state["session_id"], policy_id=action.action_id, reward=result.reward,
              delta_qoe=result.delta_qoe, delta_sre=result.delta_sre, qoe=result.after.qoe, sre=result.after.sre,
              predicted_utility=predicted, observed_utility=result.after.utility,
              baseline_used=result.baseline_used, weights_version=result.weights_version)
    return {"evaluation": evaluation, "qoe": result.after.qoe, "sre": result.after.sre, "reward": result.reward}


def store_experience(state: AgentState) -> dict[str, Any]:
    """Persist (s, a, R, s_next) once, in the DB and the FAISS index; skip transitions that did not complete."""
    evaluation = state.get("evaluation") or {}
    if evaluation.get("status") != "completed":
        log_event("transition_not_stored", session_id=state["session_id"], status=evaluation.get("status"))
        return {"session_errors": [f"experience not stored: {evaluation.get('reason', 'evaluation incomplete')}"]}
    store = get_services().experience_store()
    try:
        with observe("store_experience", metadata={"session_id": state["session_id"]}) as span:
            record, created = store.record_transition(
                student_id=state["student_id"], session_id=state["session_id"], topic_id=state["topic_id"],
                context=state["context"], s=RuntimeState.model_validate(state["prev_state"] or {"context": {},
                                                                                                "baseline": True}),
                s_next=RuntimeState.model_validate(evaluation["next_state"]),
                action=AdaptationAction(action_id=state["policy_id"], config=state["runtime_config"]),
                result=RewardResult.model_validate(evaluation["result"]),
                predicted_utility=evaluation.get("predicted_utility"))
            span.update(output={"experience_id": record["id"], "created": created})
    except Exception as exc:  # a storage failure must not lose the learner's finished session
        logger.error("store_experience failed: %s", exc)
        return {"session_errors": [f"store_experience failed: {exc}"]}
    log_event("transition_stored", session_id=state["session_id"], experience_id=record["id"], created=created)
    return {}
