"""Run one session with the real services: FAISS + CRAG retrieval and the local/cloud LLM router.

Needs Ollama running (embedding + local model); the cloud route is used only if HF_TOKEN is set.
Builds the topic's index from knowledge_prep/samples/<topic>.txt if it does not exist yet.

Usage:
  .venv\\Scripts\\python scripts\\run_session.py
  .venv\\Scripts\\python scripts\\run_session.py --topic photosynthesis -q "what is chlorophyll?" -q "why do leaves need stomata?"
  .venv\\Scripts\\python scripts\\run_session.py --policy P3      # force a policy (cloud route)
"""
import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.types import Command  # noqa: E402

from agent.deps import get_services  # noqa: E402
from agent.graph import build_graph, get_checkpointer  # noqa: E402
from agent.policies import POLICIES  # noqa: E402
from agent.runner import run_until_pause  # noqa: E402
from agent.state import initial_state  # noqa: E402
from config import settings  # noqa: E402
from knowledge_prep.build_index import build_topic_index  # noqa: E402

DEFAULT_QUESTIONS = ["photosynthesis?", "How do plants convert sunlight into chemical energy?"]


def main() -> int:
    """Build the index if needed, then drive a session through its interrupts."""
    sys.stdout.reconfigure(encoding="utf-8")  # answers may contain non-ASCII (quotes, Indic scripts)
    parser = argparse.ArgumentParser(description="Run a real learning session.")
    parser.add_argument("--topic", default="photosynthesis")
    parser.add_argument("-q", "--question", action="append", dest="questions")
    parser.add_argument("--policy", choices=sorted(POLICIES), help="override select_policy (for experiments)")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the topic index first")
    args = parser.parse_args()
    questions = args.questions or DEFAULT_QUESTIONS

    store = get_services().crag.store
    if args.rebuild or not store.exists(args.topic):
        print(f"Building index for '{args.topic}' (first embedding call may take ~1 min while Ollama loads)...")
        meta = build_topic_index(args.topic, settings.knowledge_samples_dir / f"{args.topic}.txt")
        print(f"  {len(meta.chunks)} chunks, {meta.embedding_model}")

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)
    state = initial_state(student_id="student-001", topic_id=args.topic)
    cfg = {"configurable": {"thread_id": state["session_id"]}}

    def on_node(node: str, update: dict[str, Any]) -> None:
        """Print progress, overriding the policy right after select_policy if asked."""
        detail = ""
        if node == "crag_retrieve" and "retrieval_score" in update:
            detail = f"score={update['retrieval_score']:.3f} chunks={len(update['chunks'])}"
        elif node == "corrective_retrieve":
            detail = f"rewritten: {update['retrieval_query']!r}"
        elif node in ("llm_router", "select_policy"):
            detail = update.get("model_used") or update.get("policy_id", "")
        print(f"  -> {node:<24} {detail}")

    pause, _ = run_until_pause(graph, state, cfg, on_node)
    if args.policy:
        # Experiments only: replace the selected policy before the first question.
        graph.update_state(cfg, {"policy_id": args.policy,
                                 "runtime_config": POLICIES[args.policy].runtime_config()})
        print(f"  (policy overridden to {args.policy})")

    pending = iter([{"query": q, "continue_session": i < len(questions) - 1} for i, q in enumerate(questions)])
    while pause is not None:
        if pause["type"] == "ask_question":
            reply: Any = next(pending)
            print(f"\n  Q{pause['question_no']}: {reply['query']}")
        else:
            reply = {"answers": ["A"] * len(pause["items"]), "rating": 4, "comment": "real-session run"}
        pause, _ = run_until_pause(graph, Command(resume=reply), cfg, on_node)
        values = graph.get_state(cfg).values
        if pause is not None and pause["type"] in ("ask_question", "post_test_and_feedback") and values["response"]:
            print(textwrap.fill(values["response"], 100, initial_indent="     A: ", subsequent_indent="        "))

    final = graph.get_state(cfg).values
    checkpointer.conn.close()
    print("\nqa_log:")
    for e in final["qa_log"]:
        print(f"  Q{e['question_no']} score={e['score']:.3f} retries={e['retries']} chunks={e['n_chunks']} "
              f"model={e['model']} latency={e['latency_ms']:.0f}ms (gen {e['gen_latency_ms']:.0f}ms) "
              f"error={e['error']}")
    print(f"\npolicy={final['policy_id']} qoe={final['qoe']} sre={final['sre']} reward={final['reward']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
