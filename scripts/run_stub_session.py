"""Run one full session offline with fake retrieval/LLMs, resuming each interrupt programmatically.

The real CRAG retriever and LLM router run; only the vector store and LLM clients are fakes.
Scenario: 2 questions (the first is vague and triggers one CRAG retry), then the post-test.
It also sends one invalid reply to check that the interrupt re-asks.
Exits non-zero if the visited path or the final state is not as expected.
"""
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.types import Command  # noqa: E402

from agent.deps import make_fake_services, set_services  # noqa: E402
from agent.graph import build_graph, get_checkpointer  # noqa: E402
from agent.runner import run_until_pause  # noqa: E402
from agent.state import initial_state  # noqa: E402

# Resume values, consumed in order, one per interrupt.
RESUMES: list[Any] = [
    "photosynthesis?",  # vague: triggers 1 CRAG retry (bare string = continue)
    {"query": "   ", "continue_session": False},  # invalid: blank, so ask_question must re-ask
    {"query": "How do plants convert sunlight into chemical energy?", "continue_session": False},
    {"answers": ["A", "A", "A", "B", "C"], "rating": 4, "comment": "Clear answers, a bit slow."},
]

EXPECTED_PATH = [
    "load_profile", "pre_test", "collect_context", "select_policy",
    "ask_question", "crag_retrieve", "corrective_retrieve", "crag_retrieve", "llm_router", "generate_response",
    "ask_question", "crag_retrieve", "llm_router", "generate_response",
    "post_test_and_feedback", "evaluate_session", "store_experience",
]


def print_node(node: str, update: dict[str, Any]) -> None:
    """Print one visited node and the keys it updated."""
    print(f"  -> {node:<24} updates: {', '.join(sorted(update)) or '-'}")


def main() -> int:
    """Drive the session and check the outcome."""
    set_services(make_fake_services())
    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)
    state = initial_state(student_id="student-001", topic_id="photosynthesis")
    cfg = {"configurable": {"thread_id": state["session_id"]}}
    print(f"Session {state['session_id']} (checkpoints: data/checkpoints.db)\n")

    visited: list[str] = []
    reasked = False
    pause, seen = run_until_pause(graph, state, cfg, print_node)
    visited += seen
    resumes = iter(RESUMES)
    while pause is not None:
        if "error" in pause:
            reasked = True
        print(f"  || interrupt: {pause['type']}" + (f"  [error: {pause['error']}]" if "error" in pause else ""))
        value = next(resumes)
        print(f"  >> resume with: {value!r}")
        pause, seen = run_until_pause(graph, Command(resume=value), cfg, print_node)
        visited += seen

    final = graph.get_state(cfg)
    checkpointer.conn.close()
    s = final.values

    print("\nFinal state:")
    for key in ("policy_id", "runtime_config", "competency", "pre_score", "post_score",
                "rating", "qoe", "sre", "reward"):
        print(f"  {key:<15} {s[key]}")
    print("  qa_log:")
    for entry in s["qa_log"]:
        print(f"    {entry}")

    q1, q2 = (s["qa_log"] + [{}, {}])[:2]
    checks = {
        "path matches expected": visited == EXPECTED_PATH,
        "graph finished": not final.next,
        "invalid reply was re-asked": reasked,
        "2 questions logged": len(s["qa_log"]) == 2,
        "Q1 had 1 CRAG retry with a rewritten query": q1.get("retries") == 1
        and q1.get("retrieval_query") != q1.get("query"),
        "Q2 had 0 CRAG retries": q2.get("retries") == 0,
        "answers came from the local route (P1)": all(e["model"].startswith("local:") for e in s["qa_log"]),
        "post-test scored": s["post_score"] is not None,
        "reward computed": s["reward"] > 0,
    }
    print("\nChecks:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not checks["path matches expected"]:
        print(f"  visited:  {visited}\n  expected: {EXPECTED_PATH}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
