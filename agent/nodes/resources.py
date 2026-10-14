"""Topic and learning-resource nodes: select_topic, recommend_videos, select_video, prepare_knowledge.

Flow: select_topic → (pre-test) → recommend_videos → select_video → [prepare_knowledge] → collect_context.
Questions are answered from the chosen video's transcript index (yt_<video_id>); the topic's lesson notes
only ground the topic-level pre/post-test.
"""
import logging
from typing import Any

from agent.deps import get_services
from agent.knowledge import KnowledgeUnavailable, index_ready, index_video, transcript_languages, video_resource_id
from agent.nodes.assessment import _ask
from agent.schemas import TopicReply, VideoReply
from agent.state import AgentState
from knowledge_prep.catalog import TOPICS, topic_label

logger = logging.getLogger(__name__)

# Recommendation fields kept in the state and shown to the learner.
VIDEO_FIELDS = ("item_id", "rank", "video_id", "title", "description", "thumbnail_url", "channel_title", "duration_s",
                "url", "score", "explanation", "evidence_level", "caption_language", "caption_type")


def select_topic(state: AgentState) -> dict[str, Any]:
    """Pause for {"topic_id": ...} from the catalogue, unless the session was started with a topic."""
    if state["topic_id"]:
        return {}
    topics = [{"topic_id": t.topic_id, "title": t.title, "label": topic_label(t.topic_id), "grade": t.grade,
               "subject": t.subject} for t in TOPICS]
    reply = _ask({"type": "select_topic", "topics": topics, "prompt": "Choose the topic you want to study."},
                 TopicReply)
    return {"topic_id": reply.topic_id}


def recommend_videos(state: AgentState) -> dict[str, Any]:
    """Search YouTube for the topic using the learner's competency and preferred language, and rank results."""
    recommender = get_services().video_recommender()
    refresh = state["video_action"] == "refresh"
    update: dict[str, Any] = {"video_action": "", "knowledge_error": ""}
    try:
        run = recommender.recommend(state["student_id"], state["topic_id"], session_id=state["session_id"])
    except Exception as exc:  # unexpected failure (API errors are reported in the run's status/errors)
        logger.exception("recommend_videos failed")
        message = f"{type(exc).__name__}: {exc}"
        return {**update, "recommendations": {"run_id": None, "status": "failed", "errors": [message], "items": []},
                "session_errors": [f"recommend_videos: {message}"]}
    items = [{k: rec.get(k) for k in VIDEO_FIELDS} for rec in run["recommendations"]]
    return {**update, "recommendations": {
        "run_id": run["id"], "status": run["status"], "errors": list(run.get("errors") or []),
        "strategy_id": run.get("strategy_id"), "decision_reason": run.get("decision_reason"),
        "query": run.get("query"), "items": items, "refreshed": refresh}}


def _unavailable(state: AgentState) -> dict[str, str]:
    """video_id → why it could not be indexed earlier in this session."""
    return {a["video_id"]: a["reason"] for a in state["knowledge_attempts"] if a["status"] != "ok"}


def select_video(state: AgentState) -> dict[str, Any]:
    """Pause for the learner's choice of video, a request for new recommendations, or the end of the session."""
    recs = state["recommendations"]
    unavailable = _unavailable(state)
    items = [{**item, "unavailable": unavailable.get(item["video_id"])} for item in recs.get("items", [])]
    choosable = {item["item_id"]: item for item in items if not item["unavailable"]}
    prompt: dict[str, Any] = {
        "type": "select_video", "topic_id": state["topic_id"], "items": items,
        "status": recs.get("status"), "errors": recs.get("errors", []), "strategy_id": recs.get("strategy_id"),
        "decision_reason": recs.get("decision_reason"), "notice": state["knowledge_error"] or None,
        "prompt": 'Pick a video to learn from ({"item_id": ...}), ask for new recommendations '
                  '({"refresh": true}) or end the session ({"end": true}).'}
    while True:
        reply = _ask(prompt, VideoReply)
        if reply.item_id is None or reply.item_id in choosable:
            break
        prompt = {**prompt, "error": f"item_id {reply.item_id!r} is not one of the available recommendations"}

    recommender, run_id = get_services().video_recommender(), recs.get("run_id")
    errors: list[str] = []

    def event(event_type: str, item_id: str) -> None:
        if run_id is None:
            return
        try:
            recommender.record_event(run_id, event_type, item_id=item_id)
        except Exception as exc:  # outcome logging must not block the session
            errors.append(f"recommendation event '{event_type}': {type(exc).__name__}: {exc}")

    if reply.end:
        return {"video_action": "end"}
    if reply.refresh:
        for item_id in choosable:  # the learner asked for something else: these were not wanted
            event("dismissed", item_id)
        return {"video_action": "refresh", "session_errors": errors}

    video = choosable[reply.item_id]
    event("accepted", video["item_id"])
    resource_id = video_resource_id(video["video_id"])
    reused = index_ready(get_services().crag.store, resource_id)
    return {"video_action": "video", "video": {k: video.get(k) for k in VIDEO_FIELDS} | {"reused_index": reused},
            "resource_id": resource_id if reused else "", "knowledge_error": "", "session_errors": errors}


def prepare_knowledge(state: AgentState) -> dict[str, Any]:
    """Index the chosen video's transcript (video → transcript → chunks → embeddings → FAISS).

    If the transcript is unavailable, try the next recommendations in rank order; if none works, record why
    so select_video can ask the learner again.
    """
    services = get_services()
    store, transcripts = services.crag.store, services.video_recommender().transcripts
    languages = transcript_languages(state["profile"].get("language"))
    chosen = state["video"]
    tried = {a["video_id"] for a in state["knowledge_attempts"]}
    queue = [chosen, *(i for i in state["recommendations"].get("items", [])
                       if i["video_id"] != chosen["video_id"] and i["video_id"] not in tried)]
    attempts: list[dict[str, Any]] = []
    for video in queue:
        resource_id = video_resource_id(video["video_id"])
        try:
            if index_ready(store, resource_id):
                info: dict[str, Any] = {"resource_id": resource_id, "reused_index": True}
            else:
                info = {**index_video(store, transcripts, video, languages), "reused_index": False}
        except KnowledgeUnavailable as exc:
            attempts.append({"video_id": video["video_id"], "title": video.get("title"), "status": "failed",
                             "reason": str(exc)})
            continue
        attempts.append({"video_id": video["video_id"], "title": video.get("title"), "status": "ok", "reason": ""})
        errors = [f"prepare_knowledge: '{a['title']}' skipped ({a['reason']})" for a in attempts if a["status"] != "ok"]
        try:
            info["resource_db_id"] = services.app.register_resource(
                source="youtube", external_id=video["video_id"], title=video.get("title") or video["video_id"],
                url=video.get("url"), topic=state["topic_id"],
                metadata={k: info.get(k) for k in ("resource_id", "n_chunks", "embedding_model", "caption_language",
                                                   "caption_type", "truncated", "words") if info.get(k) is not None})
        except Exception as exc:  # metadata only; the index is usable
            errors.append(f"prepare_knowledge: resource metadata not stored ({type(exc).__name__}: {exc})")
        video_state = {k: video.get(k) for k in VIDEO_FIELDS} | info
        if video["video_id"] != chosen["video_id"]:
            video_state["replaces"] = {"video_id": chosen["video_id"], "title": chosen.get("title")}
        return {"video": video_state, "resource_id": resource_id,
                "knowledge_attempts": [*state["knowledge_attempts"], *attempts], "session_errors": errors}

    reasons = "; ".join(f"'{a['title']}': {a['reason']}" for a in attempts)
    return {"resource_id": "", "knowledge_attempts": [*state["knowledge_attempts"], *attempts],
            "knowledge_error": f"None of the recommended videos could be prepared for questions ({reasons}). "
                               "Ask for new recommendations or end the session.",
            "session_errors": [f"prepare_knowledge: no usable transcript ({reasons})"]}
