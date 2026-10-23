"""Coaching nodes: the session plan, tracking after each answer, the rule-based coach and formative checks.

    pre_test ─▶ plan_session ─▶ recommend_videos ...            (plan_pretest ─▶ plan_session when there is no pre-test)
    generate_response ─▶ update_tracker ─▶ monitor_progress ─▶ coach ─▶ ask_question / post-test
    ask_question (formative_check accepted) ─▶ plan_formative ─▶ formative_check ─▶ record_formative ─▶ update_tracker

Rules live in agent/planning.py (goals) and agent/coach.py (suggestions). Data goes only through agent/coach_data.py
(tracker / matching / formative pool, with stubs while those are unimplemented). Every node is best effort: a topic
without a vocabulary or a data error is recorded in the state and never stops the session.
"""
import logging
from typing import Any

from agent import coach_data
from agent.coach import choose_action, choose_scaffold, muted_suggestions
from agent.deps import get_services
from agent.nodes.assessment import _ask
from agent.planning import goal_progress, missed_concepts, plan_goals, recommendation_query
from agent.reflection import build_reflection
from agent.schemas import FormativeReply
from agent.state import AgentState
from app.db.repositories import AdaptationRepository
from app.services.mcq import normalize_answer
from config import settings
from knowledge_prep import concepts as concept_files
from telemetry import tracing

LETTERS = "ABCD"
_vocabularies: dict[str, tuple[float | None, Any]] = {}


def load_vocabulary(topic_id: str) -> Any:
    """The topic's vocabulary, re-read only when its file changes (several nodes need it on every turn; each read
    parses and validates the JSON, ~0.7 ms)."""
    path = concept_files.path_for(topic_id)
    mtime = path.stat().st_mtime if path.exists() else None
    cached = _vocabularies.get(str(path))
    if cached is None or cached[0] != mtime:
        cached = (mtime, concept_files.load_vocabulary(topic_id))
        _vocabularies[str(path)] = cached
    return cached[1]

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
            mastery = coach_data.get_mastery(state["student_id"], topic_id, db=db)
    except Exception as exc:  # planning is an addition: without it the session continues unplanned
        logger.exception("plan_session failed")
        message = f"{type(exc).__name__}: {exc}"
        return {"session_plan": {"status": "failed", "goals": [], "query": {}, "reason": message},
                "session_errors": [f"plan_session: {message}"]}
    start = {cid: s.model_dump() for cid, s in mastery.items()}
    if state["coach_mode"] == "off":  # the reactive system: no plan; mastery is still logged for the comparison
        return {"session_plan": {"status": "off", "goals": [], "query": {}}, "mastery_start": start}
    carry_over = (state["profile"].get("weak_concepts") or {}).get(topic_id) or []
    goals = plan_goals(vocabulary, missed, mastery, min_goals=settings.plan_min_goals,
                       max_goals=settings.plan_max_goals, low_mastery=settings.plan_low_mastery, carry_over=carry_over)
    return {"session_plan": {
        "status": "ok", "goals": [g.model_dump() for g in goals], "query": recommendation_query(topic_id, goals),
        "pretest_missed": missed, "used_pretest": pretest is not None, "vocabulary_status": vocabulary.status},
        "mastery_start": start}


def update_tracker(state: AgentState) -> dict[str, Any]:
    """Record which goal-topic concept the learner just asked about (mark_covered "asked"), via matching.py."""
    entry = state["qa_log"][-1] if state["qa_log"] else None
    if entry is None or load_vocabulary(state["topic_id"]) is None:
        return {}
    if any(u["question_no"] == entry["question_no"] for u in state["tracker_updates"]):
        return {}  # already tracked (e.g. coming back from a formative check)
    tagged = state.get("question_concept") or {}
    if tagged.get("question_no") != entry["question_no"]:  # not tagged in ask_question (e.g. an older checkpoint)
        tagged = classify_question(state, entry["query"], None, question_no=entry["question_no"])
    update: dict[str, Any] = {"question_no": entry["question_no"], "concept_id": tagged.get("concept_id"),
                              "confidence": tagged.get("confidence")}
    if tagged.get("status") != "matched":
        return {"tracker_updates": [{**update, "status": tagged.get("status", "no match"),
                                     "detail": tagged.get("detail", "")}]}
    concept_id = tagged["concept_id"]
    try:
        with get_services().app._db() as db:
            coach_data.mark_covered(state["student_id"], state["topic_id"], concept_id, "asked",
                                    source_ref=f"{state['session_id']}:q{entry['question_no']}", db=db)
    except Exception as exc:  # e.g. the matcher returned an id outside the vocabulary
        return {"tracker_updates": [{**update, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}]}
    return {"tracker_updates": [{**update, "status": "recorded", "detail": ""}]}


def monitor_progress(state: AgentState) -> dict[str, Any]:
    """Standing on each session goal now (plan card) and every concept's state (tracker_snapshot, for the coach)."""
    goals = (state.get("session_plan") or {}).get("goals") or []
    if not goals:
        return {"progress": {}, "tracker_snapshot": {}}
    try:
        with get_services().app._db() as db:
            mastery = coach_data.get_mastery(state["student_id"], state["topic_id"], db=db)
    except Exception as exc:
        return {"progress": {"error": f"{type(exc).__name__}: {exc}"}}
    misses: dict[str, int] = {}
    for answer in state["formative_history"]:
        if not answer["correct"]:
            misses[answer["concept_id"]] = misses.get(answer["concept_id"], 0) + 1
    return {"progress": {**goal_progress(goals, mastery), "after_question": len(state["qa_log"])},
            "tracker_snapshot": {cid: {**s.model_dump(), "formative_misses": misses.get(cid, 0)}
                                 for cid, s in mastery.items()}}


def coach(state: AgentState) -> dict[str, Any]:
    """Pick the one next-step suggestion shown with the next ask_question (agent/coach.py rules; no LLM)."""
    goals = (state.get("session_plan") or {}).get("goals") or []
    if not goals or state["coach_mode"] == "off":
        return {"suggestion": {}}
    entry = state["qa_log"][-1] if state["qa_log"] else {}
    tracked = next((u for u in reversed(state["tracker_updates"]) if u["question_no"] == entry.get("question_no")),
                   {})
    last_concept = tracked.get("concept_id") if tracked.get("status") == "recorded" else None
    if (state.get("formative_result") or {}).get("question_no") == entry.get("question_no"):
        last_concept = None  # this question's concept was just practised: do not offer the same check again
    coverage, video_concepts = _coverage(state)
    last_answer = {"escalated": entry.get("escalated"), "check_supported": entry.get("check_supported"),
                   "retrieval_score": entry.get("retrieval_score"),
                   "threshold": state["runtime_config"].get("threshold", settings.crag_retrieval_threshold)}
    suggestion = choose_action(goals, state.get("tracker_snapshot") or {}, last_concept=last_concept,
                               last_answer=last_answer if entry else {}, coverage=coverage,
                               muted=muted_suggestions(state["suggestion_log"]))
    return {"suggestion": {**suggestion, "after_question": entry.get("question_no")} if suggestion else {},
            "video_concepts": video_concepts}


def plan_formative(state: AgentState) -> dict[str, Any]:
    """Fix this check's items in the checkpoint before formative_check interrupts (it re-runs on each resume).

    exclude_ids: every question-bank item of the topic (any of them can be a pre/post-test question) plus the
    formative items already used this session, so formative items are never pre/post questions.
    """
    concept_id = (state.get("suggestion") or {}).get("concept_id")
    exclude = list(state["formative_seen"])
    try:
        with get_services().app._db() as db:
            bank = AdaptationRepository(db).get_question_bank(state["topic_id"])
        exclude = [coach_data.bank_item_id(state["topic_id"], i)
                   for i in range(len((bank or {}).get("items") or []))] + exclude
        items = coach_data.get_formative_items(state["topic_id"], concept_id, 2, exclude)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {"formative": {"concept_id": concept_id, "items": [], "error": message,
                              "question_no": len(state["qa_log"])},
                "session_errors": [f"formative check: {message}"]}
    return {"formative": {"concept_id": concept_id, "items": [i.model_dump() for i in items],
                          "exclude_ids": exclude, "question_no": len(state["qa_log"])}}


def formative_check(state: AgentState) -> dict[str, Any]:
    """Ask the planned items one at a time (like pre_test), giving instant feedback on each answer.

    The feedback on an answer travels with the next question's interrupt; the last one is shown back at the
    questions (formative_result). Answers are recorded by record_formative, not here: this node re-runs from the
    top on every resume, so recording here would count earlier answers again.
    """
    formative = state["formative"]
    items, results = formative.get("items") or [], []
    feedback = None
    for index, item in enumerate(items):
        reply = _ask({"type": "formative_check", "concept_id": formative["concept_id"],
                      "concept": _concept_name(state, formative["concept_id"]), "index": index, "total": len(items),
                      "item": {k: item[k] for k in ("item_id", "question", "options")}, "feedback": feedback,
                      "prompt": 'Answer with A, B, C or D ({"answer": "B"}).'}, FormativeReply)
        chosen = normalize_answer(reply.answer)
        feedback = {"item_id": item["item_id"], "question": item["question"],
                    "correct": chosen == item["answer_index"],
                    "your_answer": LETTERS[chosen] if chosen is not None else None,
                    "answer": LETTERS[item["answer_index"]], "explanation": item.get("explanation") or ""}
        results.append(feedback)
    return {"formative": {**formative, "results": results}}


def record_formative(state: AgentState) -> dict[str, Any]:
    """Record each answer as a formative attempt (once: this node has no interrupt), then summarise it for the UI."""
    formative = state["formative"]
    results, concept_id = formative.get("results") or [], formative.get("concept_id")
    errors = []
    if results and concept_id:
        try:
            with get_services().app._db() as db:
                for r in results:
                    coach_data.record_attempt(state["student_id"], state["topic_id"], concept_id, r["correct"],
                                              "formative",
                                              source_ref=f"{state['session_id']}:formative:{r['item_id']}", db=db)
        except Exception as exc:
            errors.append(f"formative check not recorded: {type(exc).__name__}: {exc}")
    note = "" if results else (formative.get("error") or "no practice questions are available for this concept")
    return {"formative_result": {"question_no": formative.get("question_no"), "concept_id": concept_id,
                                 "concept": _concept_name(state, concept_id),
                                 "correct": sum(r["correct"] for r in results), "total": len(results),
                                 "results": results, "note": note},
            "formative_seen": [r["item_id"] for r in results], "session_errors": errors,
            "formative_history": [{"concept_id": concept_id, "item_id": r["item_id"], "correct": r["correct"]}
                                  for r in results]}


def reflect(state: AgentState) -> dict[str, Any]:
    """After the post-test: goals met, mastery before vs after, checks and suggestions, a templated summary; the
    concepts still weak are saved to the learner's profile for the next session's plan."""
    vocabulary = load_vocabulary(state["topic_id"])
    if vocabulary is None:
        return {"reflection": {"status": "unavailable", "coach_mode": state["coach_mode"],
                               "reason": f"no concept vocabulary for {state['topic_id']}"}}
    services = get_services()
    try:
        with services.app._db() as db:
            after = coach_data.get_mastery(state["student_id"], state["topic_id"], db=db)
    except Exception as exc:
        return {"reflection": {"status": "failed", "coach_mode": state["coach_mode"], "reason": str(exc)},
                "session_errors": [f"reflect: {type(exc).__name__}: {exc}"]}
    reflection = build_reflection(
        (state.get("session_plan") or {}).get("goals") or [], state.get("mastery_start") or {},
        {cid: s.model_dump() for cid, s in after.items()}, {c.id: c.name for c in vocabulary.concepts},
        state["suggestion_log"],
        formative_checks=sum(1 for e in state["suggestion_log"] if e["action"] == "formative_check" and e["accepted"]),
        coach_mode=state["coach_mode"])
    errors = []
    try:
        services.app.save_weak_concepts(state["student_id"], state["topic_id"], reflection["weak_next_time"])
    except Exception as exc:  # the reflection is still shown and stored with the session
        errors.append(f"reflect: weak concepts not saved to the profile ({type(exc).__name__}: {exc})")
    return {"reflection": {"status": "ok", **reflection}, "session_errors": errors}


def classify_question(state: AgentState, query: str, accepted_concept: str | None, *,
                      question_no: int | None = None) -> dict[str, Any]:
    """The concept a learner's question is about: the accepted suggestion's, else matching.match_question.
    status: matched (confident enough to record) | low confidence | no match | stub | no vocabulary | error."""
    tagged: dict[str, Any] = {"question_no": question_no or len(state["qa_log"]) + 1, "concept_id": None,
                              "confidence": None, "status": "no vocabulary", "detail": ""}
    if load_vocabulary(state["topic_id"]) is None:
        return tagged
    if accepted_concept:
        return {**tagged, "concept_id": accepted_concept, "confidence": 1.0, "status": "matched",
                "detail": "from the accepted suggestion"}
    try:
        concept_id, confidence = coach_data.match_question(query, state["topic_id"])
    except Exception as exc:
        return {**tagged, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    if concept_id is None:
        stub = not coach_data.implemented()["match_question"]
        return {**tagged, "status": "stub" if stub else "no match",
                "detail": "matching.match_question is a stub" if stub else ""}
    confidence = round(float(confidence), 4)
    status = "matched" if confidence >= settings.coach_match_min_confidence else "low confidence"
    return {**tagged, "concept_id": concept_id, "confidence": confidence, "status": status}


def scaffold_for(state: AgentState) -> dict[str, Any]:
    """The answer shape for the current question ({} when coach_mode is off: answers are unchanged)."""
    if state["coach_mode"] == "off":
        return {}
    tagged = state.get("question_concept") or {}
    concept_id = tagged.get("concept_id") if tagged.get("status") == "matched" else None
    return choose_scaffold(state.get("tracker_snapshot") or {}, concept_id, state.get("profile") or {})


def log_suggestion(state: AgentState, accepted: bool) -> list[dict[str, Any]]:
    """The suggestion_log entry for the suggestion shown with this ask_question (empty if none was shown)."""
    suggestion = state.get("suggestion") or {}
    if not suggestion.get("action"):
        return []
    tracing.score(name="suggestion_accepted", value=int(accepted), session_id=state["session_id"],
                  data_type="BOOLEAN",
                  metadata={"action": suggestion["action"], "concept_id": suggestion["concept_id"]})
    return [{"question_no": suggestion.get("after_question"), "action": suggestion["action"],
             "concept_id": suggestion["concept_id"], "accepted": accepted}]


def _coverage(state: AgentState) -> tuple[list[str] | None, dict[str, Any]]:
    """Concepts the current video covers (cached per video in the state; None while unknown)."""
    video_id = (state.get("video") or {}).get("video_id")
    cached = dict(state.get("video_concepts") or {})
    if not video_id:
        return None, cached
    if video_id in cached:
        return cached[video_id], cached
    try:
        coverage = coach_data.video_coverage(video_id, state["topic_id"])
    except Exception as exc:  # unknown coverage: rule 2 is skipped
        logger.info("video coverage unavailable: %s", exc)
        coverage = None
    return coverage, {**cached, video_id: coverage}


def video_covers(state: AgentState, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recommended videos with "covers" (concept names) when coach_mode is rules and coverage is known; the
    coverage is also cached for the coach (video_concepts)."""
    cache = dict(state.get("video_concepts") or {})
    vocabulary = load_vocabulary(state["topic_id"]) if state["coach_mode"] == "rules" else None
    if vocabulary is None:
        return items, cache
    names = {c.id: c.name for c in vocabulary.concepts}
    out = []
    for item in items:
        if item["video_id"] not in cache:
            try:
                cache[item["video_id"]] = coach_data.video_coverage(item["video_id"], state["topic_id"])
            except Exception as exc:  # unknown coverage: no line on the card
                logger.info("video coverage of %s unavailable: %s", item["video_id"], exc)
                cache[item["video_id"]] = None
        covered = cache[item["video_id"]]
        out.append({**item, "covers": [names.get(c, c) for c in covered] if covered is not None else None})
    return out, cache


def _concept_name(state: AgentState, concept_id: str | None) -> str:
    goal = next((g for g in (state.get("session_plan") or {}).get("goals") or [] if g["concept_id"] == concept_id),
                None)
    return goal["name"] if goal else (concept_id or "").replace("_", " ")


def _session_pretest(db: Any, state: AgentState) -> dict[str, Any] | None:
    """This session's scored pre-test as stored (items keep answer_index there, unlike the state), or None."""
    for assessment in AdaptationRepository(db).list_assessments(state["student_id"], state["topic_id"]):
        if assessment["session_id"] == state["session_id"] and assessment["type"] == "pre":
            return assessment
    return None
