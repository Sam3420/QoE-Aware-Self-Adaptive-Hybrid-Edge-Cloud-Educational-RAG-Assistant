"""Question-answering loop: ask, CRAG retrieve/correct, route, generate."""
import time
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from agent.deps import get_services
from agent.schemas import QALogEntry, QuestionReply
from agent.state import AgentState
from config import settings
from services.llm_clients import LLMError, build_request
from services.vector_store import EmbeddingError, IndexNotFoundError, RetrievedChunk

_ROUTES = ("local", "cloud")


def _provider_order(preferred: str) -> list[str]:
    """Preferred route, then the other route, then the model-free extractive fallback."""
    return [preferred, *(r for r in _ROUTES if r != preferred), "extractive"]


def ask_question(state: AgentState) -> dict[str, Any]:
    """Pause for {"query": str} or {"end": true} (re-asking on invalid input) and reset per-question fields."""
    prompt: dict[str, Any] = {
        "type": "ask_question",
        "question_no": len(state["qa_log"]) + 1,
        "prompt": 'Ask your question about this topic, or send {"end": true} to finish.',
    }
    # interrupt() must come first: this node re-runs from the top on each resume.
    while True:
        raw = interrupt(prompt)
        try:
            reply = QuestionReply.model_validate(raw)
            break
        except ValidationError as exc:
            prompt = {**prompt, "error": exc.errors(include_url=False)[0]["msg"]}
    if reply.end:
        return {"continue_session": False}
    return {
        "query": reply.query,
        "retrieval_query": reply.query,
        "question_started_at": time.time(),
        "q_errors": [],
        "continue_session": True,
        "chunks": [],
        "retrieval_score": 0.0,
        "crag_retries": 0,  # retry budget is per question
        "model_used": "",
        "response": "",
    }


def crag_retrieve(state: AgentState) -> dict[str, Any]:
    """Retrieve the policy's top_k chunks, merge them with earlier attempts, and record the best score."""
    crag = get_services().crag
    top_k = state["runtime_config"]["top_k"]
    try:
        # The chosen video's transcript index; sessions from before video selection used the topic index.
        resource_id = state.get("resource_id") or state["topic_id"]
        result = crag.retrieve(resource_id, state["retrieval_query"], top_k, grade_query=state["query"])
    except (IndexNotFoundError, EmbeddingError) as exc:
        return {"q_errors": [*state["q_errors"], f"retrieval: {exc}"]}
    previous = [RetrievedChunk.model_validate(c) for c in state["chunks"]]
    merged = crag.merge(previous, result.chunks, top_k)
    return {
        "chunks": [c.model_dump() for c in merged],
        "retrieval_score": max(state["retrieval_score"], result.top_score),
    }


def corrective_retrieve(state: AgentState) -> dict[str, Any]:
    """Rewrite the query for the next retrieval attempt and count the retry."""
    attempt = state["crag_retries"] + 1
    rewritten = get_services().crag.rewrite_query(state["query"], state["topic_id"], attempt)
    return {"retrieval_query": rewritten, "crag_retries": attempt}


def llm_router(state: AgentState) -> dict[str, Any]:
    """Pick the policy's route if healthy, else the other route, else the extractive fallback."""
    router = get_services().router
    preferred = state["runtime_config"]["route"]
    provider = next(p for p in _provider_order(preferred) if p == "extractive" or router.is_available(p))
    update: dict[str, Any] = {"model_used": f"{provider}:{router.model_name(provider)}"}
    if provider != preferred:
        update["q_errors"] = [*state["q_errors"], f"router: '{preferred}' unavailable, used '{provider}'"]
    return update


def generate_response(state: AgentState) -> dict[str, Any]:
    """Generate the answer, falling back along the provider order on failure, and append a qa_log entry."""
    router = get_services().router
    preferred = state["runtime_config"]["route"]
    chosen = state["model_used"].split(":", 1)[0]
    request = build_request(
        state["query"],
        [c["text"] for c in state["chunks"]],
        grounded=state["retrieval_score"] >= state["runtime_config"]["threshold"],
        competency=state["competency"],
        language=state["profile"].get("language", "en"),
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    errors = list(state["q_errors"])
    order = _provider_order(preferred)
    candidates = [chosen, *(p for p in order[order.index(chosen) + 1:] if p == "extractive" or router.is_available(p))]
    for provider in candidates:
        try:
            result = router.generate(provider, request)
            break
        except LLMError as exc:
            errors.append(f"generation ({provider}): {exc}")
    if result.provider != chosen:
        errors.append(f"router: '{chosen}' failed, answered by '{result.provider}'")

    model_used = f"{result.provider}:{result.model}"
    entry = QALogEntry(
        question_no=len(state["qa_log"]) + 1,
        query=state["query"],
        retrieval_query=state["retrieval_query"],
        retrieval_score=state["retrieval_score"],
        crag_retries=state["crag_retries"],
        n_chunks=len(state["chunks"]),
        preferred_route=preferred,
        model_used=model_used,
        latency_ms=max(0.0, (time.time() - state["question_started_at"]) * 1000),
        gen_latency_ms=result.latency_ms,
        error="; ".join(errors) or None,
    )
    return {"response": result.text, "model_used": model_used, "q_errors": errors,
            "qa_log": [entry.model_dump()]}
