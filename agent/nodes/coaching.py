"""Coaching nodes: plan_session (goals before the videos), update_tracker and monitor_progress (after each answer).

    pre_test ─▶ plan_session ─▶ recommend_videos ...            (plan_pretest ─▶ plan_session when there is no pre-test)
    generate_response ─▶ update_tracker ─▶ monitor_progress ─▶ next question / post-test

The goal rules live in agent/planning.py. Every node here is best effort: a topic without a concept vocabulary, a
tracker error or an unimplemented matcher (matching.py is still a stub) is recorded in the state and never stops the
session, which then runs exactly as before.
"""
import logging
from typing import Any

import matching
import tracker
from agent.deps import get_services
from agent.planning import goal_progress, missed_concepts, plan_goals, recommendation_query
from agent.state import AgentState
from app.db.repositories import AdaptationRepository
from config import settings
from knowledge_prep.concepts import load_vocabulary

logger = logging.getLogger(__name__)


def plan_session(state: AgentState) -> dict[str, Any]:
    """Pick 2-4 goal concepts (pre-test misses, then low-mastery history) and the video query built from them."""
    topic_id = state["topic_id"]
    vocabulary = load_vocabulary(topic_id)
    if vocabulary is None:
        return {"session_plan": {"status": "unavailable", "goals": [], "query": {},
                                 "reason": f"no concept vocabulary for {topic_id}"}}
    services = get_services()
    try:
        with services.app._db() as db:
            pretest = _session_pretest(db, state)
            missed = missed_concepts(vocabulary, pretest, getattr(services.crag.store, "embedder", None))
            mastery = tracker.get_mastery(state["student_id"], topic_id, db=db)
    except Exception as exc:  # planning is an addition: without it the session continues unplanned
        logger.exception("plan_session failed")
        message = f"{type(exc).__name__}: {exc}"
        return {"session_plan": {"status": "failed", "goals": [], "query": {}, "reason": message},
                "session_errors": [f"plan_session: {message}"]}
    goals = plan_goals(vocabulary, missed, mastery, min_goals=settings.plan_min_goals,
                       max_goals=settings.plan_max_goals, low_mastery=settings.plan_low_mastery)
    return {"session_plan": {
        "status": "ok", "goals": [g.model_dump() for g in goals], "query": recommendation_query(topic_id, goals),
        "pretest_missed": missed, "used_pretest": pretest is not None, "vocabulary_status": vocabulary.status}}


def update_tracker(state: AgentState) -> dict[str, Any]:
    """Record which goal-topic concept the learner just asked about (mark_covered "asked"), via matching.py."""
    entry = state["qa_log"][-1] if state["qa_log"] else None
    if entry is None or load_vocabulary(state["topic_id"]) is None:
        return {}
    update: dict[str, Any] = {"question_no": entry["question_no"], "concept_id": None, "confidence": None}
    try:
        concept_id, confidence = matching.match_question(entry["query"], state["topic_id"])
    except NotImplementedError:
        return {"tracker_updates": [{**update, "status": "stub", "detail": "matching.match_question is a stub"}]}
    except Exception as exc:
        return {"tracker_updates": [{**update, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}]}
    update |= {"concept_id": concept_id, "confidence": round(float(confidence), 4)}
    if confidence < settings.coach_match_min_confidence:
        return {"tracker_updates": [{**update, "status": "low confidence", "detail": ""}]}
    try:
        with get_services().app._db() as db:
            tracker.mark_covered(state["student_id"], state["topic_id"], concept_id, "asked",
                                 source_ref=f"{state['session_id']}:q{entry['question_no']}", db=db)
    except Exception as exc:  # e.g. the matcher returned an id outside the vocabulary
        return {"tracker_updates": [{**update, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}]}
    return {"tracker_updates": [{**update, "status": "recorded", "detail": ""}]}


def monitor_progress(state: AgentState) -> dict[str, Any]:
    """Standing on each session goal now (from the tracker), for the plan card and later coaching decisions."""
    goals = (state.get("session_plan") or {}).get("goals") or []
    if not goals:
        return {"progress": {}}
    try:
        with get_services().app._db() as db:
            mastery = tracker.get_mastery(state["student_id"], state["topic_id"], db=db)
    except Exception as exc:
        return {"progress": {"error": f"{type(exc).__name__}: {exc}"}}
    return {"progress": {**goal_progress(goals, mastery), "after_question": len(state["qa_log"])}}


def _session_pretest(db: Any, state: AgentState) -> dict[str, Any] | None:
    """This session's scored pre-test as stored (items keep answer_index there, unlike the state), or None."""
    for assessment in AdaptationRepository(db).list_assessments(state["student_id"], state["topic_id"]):
        if assessment["session_id"] == state["session_id"] and assessment["type"] == "pre":
            return assessment
    return None
