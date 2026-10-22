"""Wiring of the single learning-session StateGraph."""
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent import nodes
from agent.node_tracing import traced
from agent.routing import continue_session, crag_ok, knowledge_ready, more_questions, need_pretest, route_video
from agent.state import AgentState
from config import settings


def build_builder() -> StateGraph:
    """Return the uncompiled graph with every node and edge."""
    g = StateGraph(AgentState)

    for name in ("load_profile", "select_topic", "plan_pretest", "pre_test", "recommend_videos", "select_video",
                 "prepare_knowledge", "collect_context", "select_policy", "ask_question", "crag_retrieve",
                 "corrective_retrieve", "llm_router", "generate_response", "post_test_and_feedback",
                 "evaluate_session", "store_experience", "plan_session", "update_tracker", "monitor_progress"):
        g.add_node(name, traced(name, getattr(nodes, name)))  # Langfuse observation per node (no-op when off)

    g.add_edge(START, "load_profile")
    g.add_edge("load_profile", "select_topic")
    # Every session takes a topic-level pre-test planned from the learner's history (skipped only when no
    # questions could be prepared), then gets videos recommended for the topic, competency and language.
    g.add_edge("select_topic", "plan_pretest")
    g.add_conditional_edges("plan_pretest", need_pretest, ["pre_test", "plan_session"])
    # The session plan (2-4 goal concepts: pre-test misses, then low mastery) shapes the video query.
    g.add_edge("pre_test", "plan_session")
    g.add_edge("plan_session", "recommend_videos")
    g.add_edge("recommend_videos", "select_video")
    # A chosen video is indexed once (transcript → chunks → embeddings → FAISS) and reused afterwards.
    # The first video continues to context/policy; a video chosen later goes straight back to questions.
    g.add_conditional_edges("select_video", route_video,
                            ["recommend_videos", "post_test_and_feedback", "prepare_knowledge", "collect_context",
                             "ask_question"])
    g.add_conditional_edges("prepare_knowledge", knowledge_ready, ["collect_context", "ask_question", "select_video"])
    g.add_edge("collect_context", "select_policy")
    g.add_edge("select_policy", "ask_question")
    # At ask_question the learner asks a doubt (→ retrieval), watches another video (→ select_video)
    # or takes the post-test ({"end": true}).
    g.add_conditional_edges("ask_question", continue_session,
                            ["crag_retrieve", "select_video", "post_test_and_feedback"])
    # CRAG loop: retry retrieval until the score is good enough or the retries run out.
    g.add_conditional_edges("crag_retrieve", crag_ok, ["llm_router", "corrective_retrieve"])
    g.add_edge("corrective_retrieve", "crag_retrieve")
    g.add_edge("llm_router", "generate_response")
    # After each answer: record the concept asked about, update progress on the goals, then the question loop
    # (bounded by max_questions_per_session).
    g.add_edge("generate_response", "update_tracker")
    g.add_edge("update_tracker", "monitor_progress")
    g.add_conditional_edges("monitor_progress", more_questions, ["ask_question", "post_test_and_feedback"])
    g.add_edge("post_test_and_feedback", "evaluate_session")
    g.add_edge("evaluate_session", "store_experience")
    g.add_edge("store_experience", END)
    return g


def get_checkpointer(db_path: Path = settings.checkpoint_db_path) -> SqliteSaver:
    """Open the SQLite checkpointer (separate from the app DB), creating its folder if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compile the agent graph with a checkpointer (SQLite at data/checkpoints.db by default)."""
    return build_builder().compile(checkpointer=checkpointer or get_checkpointer())
