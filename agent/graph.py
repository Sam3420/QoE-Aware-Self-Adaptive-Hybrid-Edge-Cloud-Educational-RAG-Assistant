"""Wiring of the single learning-session StateGraph."""
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent import nodes
from agent.routing import continue_session, crag_ok, more_questions, need_pretest
from agent.state import AgentState
from config import settings


def build_builder() -> StateGraph:
    """Return the uncompiled graph with every node and edge."""
    g = StateGraph(AgentState)

    g.add_node("load_profile", nodes.load_profile)
    g.add_node("pre_test", nodes.pre_test)
    g.add_node("collect_context", nodes.collect_context)
    g.add_node("select_policy", nodes.select_policy)
    g.add_node("ask_question", nodes.ask_question)
    g.add_node("crag_retrieve", nodes.crag_retrieve)
    g.add_node("corrective_retrieve", nodes.corrective_retrieve)
    g.add_node("llm_router", nodes.llm_router)
    g.add_node("generate_response", nodes.generate_response)
    g.add_node("post_test_and_feedback", nodes.post_test_and_feedback)
    g.add_node("evaluate_session", nodes.evaluate_session)
    g.add_node("store_experience", nodes.store_experience)

    g.add_edge(START, "load_profile")
    g.add_conditional_edges("load_profile", need_pretest, ["pre_test", "collect_context"])
    g.add_edge("pre_test", "collect_context")
    g.add_edge("collect_context", "select_policy")
    g.add_edge("select_policy", "ask_question")
    # {"end": true} at ask_question finishes the session; a query goes to retrieval.
    g.add_conditional_edges("ask_question", continue_session, ["crag_retrieve", "post_test_and_feedback"])
    # CRAG loop: retry retrieval until the score is good enough or the retries run out.
    g.add_conditional_edges("crag_retrieve", crag_ok, ["llm_router", "corrective_retrieve"])
    g.add_edge("corrective_retrieve", "crag_retrieve")
    g.add_edge("llm_router", "generate_response")
    # Question loop (bounded by max_questions_per_session).
    g.add_conditional_edges("generate_response", more_questions, ["ask_question", "post_test_and_feedback"])
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
