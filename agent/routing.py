"""Conditional-edge functions; each returns the name of the next node."""
from typing import Literal

from agent.state import AgentState
from config import settings


def need_pretest(state: AgentState) -> Literal["pre_test", "recommend_videos"]:
    """Every session takes the planned pre-test; skip it only when no questions could be prepared."""
    return "pre_test" if state["test_items"] else "recommend_videos"


def _to_questions(state: AgentState) -> Literal["collect_context", "ask_question"]:
    """The first video goes through context measurement and policy selection; later videos keep that policy."""
    return "ask_question" if state["policy_id"] else "collect_context"


def route_video(state: AgentState) -> Literal["recommend_videos", "post_test_and_feedback", "prepare_knowledge",
                                              "collect_context", "ask_question"]:
    """After select_video: new recommendations, end, or the chosen video (indexing it first if needed)."""
    action = state["video_action"]
    if action == "refresh":
        return "recommend_videos"
    if action == "end":
        return "post_test_and_feedback"
    return _to_questions(state) if state["resource_id"] else "prepare_knowledge"


def knowledge_ready(state: AgentState) -> Literal["collect_context", "ask_question", "select_video"]:
    """After prepare_knowledge: continue with the indexed video, or let the learner choose again."""
    return _to_questions(state) if state["resource_id"] else "select_video"


def continue_session(state: AgentState) -> Literal["crag_retrieve", "select_video", "post_test_and_feedback"]:
    """After ask_question: answer the doubt, go back to the videos, or take the post-test."""
    if state["continue_session"]:
        return "crag_retrieve"
    return "select_video" if state["video_action"] == "more" else "post_test_and_feedback"


def crag_ok(state: AgentState) -> Literal["llm_router", "corrective_retrieve"]:
    """Accept retrieval when the score meets the threshold or the retry budget is spent."""
    cfg = state["runtime_config"]
    threshold = cfg.get("threshold", settings.crag_retrieval_threshold)
    max_retries = cfg.get("max_retries", 0)
    if state["retrieval_score"] >= threshold or state["crag_retries"] >= max_retries:
        return "llm_router"
    return "corrective_retrieve"


def more_questions(state: AgentState) -> Literal["ask_question", "post_test_and_feedback"]:
    """After an answer: ask for the next question unless the per-session cap is reached."""
    if len(state["qa_log"]) < settings.max_questions_per_session:
        return "ask_question"
    return "post_test_and_feedback"
