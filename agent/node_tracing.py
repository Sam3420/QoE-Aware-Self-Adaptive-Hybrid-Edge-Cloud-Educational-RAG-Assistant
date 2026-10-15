"""Langfuse observations for graph nodes: a verb-first name, the most specific type, and a small readable
input/output summary per node (never the whole state, which holds chunks, histories and ids).

`generate_response` is not wrapped: the LLM clients already record each model call as a
`generation` (and the extractive fallback as a `tool`), so a wrapper would only add an empty layer.
A node that pauses for the learner (langgraph `interrupt`) is recorded as waiting, not as an error;
it runs again (and is traced again) in the next step's trace when the learner replies.
"""
from collections.abc import Callable
from typing import Any

from langgraph.errors import GraphInterrupt

from telemetry import tracing

Summary = Callable[[dict[str, Any]], Any]


def _errors(update: dict[str, Any]) -> list[str]:
    return list(update.get("session_errors") or []) + list(update.get("q_errors") or [])


def _chunks(update: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"chunk_id": c["chunk_id"], "score": round(c["score"], 3), "text": tracing.preview(c["text"], 240)}
            for c in update.get("chunks") or []]


# node → (observation name, type, input summary from state, output summary from the node's update)
NODE_SPECS: dict[str, tuple[str, str, Summary, Summary]] = {
    "load_profile": ("load-learner-profile", "span",
                     lambda s: {"topic_id": s["topic_id"]},
                     lambda u: {"language": (u.get("profile") or {}).get("language"),
                                "pretest_already_taken": u.get("pre_score") is not None}),
    "pre_test": ("run-pretest", "span",
                 lambda s: {"topic_id": s["topic_id"]},
                 lambda u: {"pre_score": u.get("pre_score"), "competency": u.get("competency"),
                            "errors": _errors(u)}),
    "collect_context": ("measure-context", "tool",
                        lambda s: {"competency": s["competency"]},
                        lambda u: u.get("context")),
    "select_policy": ("select-policy", "chain",
                      lambda s: {"context": s["context"]},
                      lambda u: {"policy_id": u.get("policy_id"), "runtime_config": u.get("runtime_config"),
                                 "reason": u.get("decision_reason"), "estimates": u.get("candidate_utilities")}),
    "ask_question": ("receive-question", "span",
                     lambda s: {"question_no": len(s["qa_log"]) + 1},
                     lambda u: {"query": u.get("query")} if u.get("continue_session", True) else {"end": True}),
    "crag_retrieve": ("retrieve-lesson-context", "retriever",
                      lambda s: {"query": s["retrieval_query"], "top_k": s["runtime_config"].get("top_k"),
                                 "topic_id": s["topic_id"], "attempt": s["crag_retries"] + 1},
                      lambda u: {"best_score": u.get("retrieval_score"), "chunks": _chunks(u), "errors": _errors(u)}),
    "corrective_retrieve": ("rewrite-query", "span",
                            lambda s: {"query": s["query"], "retry": s["crag_retries"] + 1},
                            lambda u: {"rewritten_query": u.get("retrieval_query")}),
    "llm_router": ("route-llm", "span",
                   lambda s: {"preferred_route": s["runtime_config"].get("route")},
                   lambda u: {"chosen": u.get("model_used"), "errors": _errors(u)}),
    "post_test_and_feedback": ("score-posttest", "span",
                               lambda s: {"topic_id": s["topic_id"]},
                               lambda u: {"post_score": u.get("post_score"), "rating": u.get("rating"),
                                          "comment": u.get("comment"), "errors": _errors(u)}),
    "evaluate_session": ("evaluate-session", "evaluator",
                         lambda s: {"policy_id": s["policy_id"], "questions": len(s["qa_log"])},
                         lambda u: {k: (u.get("evaluation") or {}).get(k) for k in
                                    ("status", "qoe_before", "qoe_after", "sre_before", "sre_after", "reward",
                                     "predicted_utility", "utility_after", "qoe_components", "qoe_coverage",
                                     "baseline_used", "reason")}),
    "store_experience": ("store-experience", "span",
                         lambda s: {"policy_id": s["policy_id"]},
                         lambda u: {"stored": not _errors(u), "errors": _errors(u)}),
}


def traced(node_name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a node in its observation (no-op when tracing is off)."""
    spec = NODE_SPECS.get(node_name)
    if spec is None:
        return fn
    name, as_type, input_fn, output_fn = spec

    def node(state: dict[str, Any]) -> dict[str, Any]:
        if not tracing.enabled():
            return fn(state)
        pause: GraphInterrupt | None = None
        with tracing.observation(name, as_type=as_type, input=_safe(input_fn, state),
                                 metadata={"graph_node": node_name}) as obs:
            try:
                update = fn(state)
            except GraphInterrupt as exc:  # control flow, not a failure: close the span normally
                pause = exc
                payload = _interrupt_payload(exc)
                obs.update(output={"waiting_for_learner": payload.get("type"), "reask_error": payload.get("error")})
            else:
                obs.update(output=_safe(output_fn, update), **({"level": "WARNING"} if _errors(update) else {}))
        if pause is not None:
            raise pause
        return update

    node.__name__ = getattr(fn, "__name__", node_name)
    node.__doc__ = fn.__doc__
    return node


def _interrupt_payload(exc: GraphInterrupt) -> dict[str, Any]:
    """The payload the node sent to the learner (interrupt value), or {}."""
    try:
        value = exc.args[0][0].value
        return value if isinstance(value, dict) else {}
    except (IndexError, AttributeError, TypeError):
        return {}


def _safe(fn: Summary, value: dict[str, Any]) -> Any:
    try:
        return fn(value)
    except Exception:  # a summary must never break the node
        return None
