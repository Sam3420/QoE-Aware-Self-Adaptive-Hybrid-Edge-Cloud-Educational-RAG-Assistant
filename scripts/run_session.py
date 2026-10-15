"""Run one scripted session through the runner API, printing nodes, the policy decision and qa_log.

Real mode (default): FAISS + CRAG via Ollama embeddings, local/cloud LLM router, psutil + RTT probe,
and Person B's DB at data/app.db (the experience is stored there and feeds future kNN decisions).
The topic index is built from knowledge_prep/samples/<topic>.txt if missing.

--mock-llm: fully offline (no Ollama/HF/network): fake retrieval, LLMs, monitor and MCQ generation,
on a private in-memory DB, so nothing is written to data/app.db.

Usage:
  .venv\\Scripts\\python scripts\\run_session.py
  .venv\\Scripts\\python scripts\\run_session.py --mock-llm
  .venv\\Scripts\\python scripts\\run_session.py --student <learner_id> -q "what is chlorophyll?" -q "why are leaves flat?"
"""
import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.deps import get_services, make_fake_services, set_services  # noqa: E402
from agent.graph import build_graph, get_checkpointer  # noqa: E402
from agent.runner import SessionRunner  # noqa: E402
from config import settings  # noqa: E402

DEFAULT_QUESTIONS = ["photosynthesis?", "How do plants convert sunlight into chemical energy?"]


def on_node(node: str, update: dict[str, Any]) -> None:
    """Print each node visited with its most useful detail."""
    detail = ""
    if node == "crag_retrieve" and "retrieval_score" in update:
        detail = f"score={update['retrieval_score']:.3f} chunks={len(update['chunks'])}"
    elif node == "corrective_retrieve":
        detail = f"rewritten: {update['retrieval_query']!r}"
    elif node == "llm_router":
        detail = update["model_used"]
    elif node == "collect_context":
        detail = str(update["context"])
    elif node == "select_policy":
        detail = f"{update['policy_id']} {update['runtime_config']}"
    elif node == "pre_test" and "pre_score" in update:
        detail = f"pre_score={update['pre_score']} competency={update['competency']}"
    print(f"  -> {node:<24} {detail}")


def main() -> int:
    """Set up services, drive one session through every interrupt, and print the outcome."""
    sys.stdout.reconfigure(encoding="utf-8")  # answers may contain non-ASCII
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run a scripted learning session.")
    parser.add_argument("--topic", default="photosynthesis")
    parser.add_argument("--student", help="existing learner id (default: create a new demo learner)")
    parser.add_argument("-q", "--question", action="append", dest="questions")
    parser.add_argument("--mock-llm", action="store_true", help="offline: fake retrieval/LLMs/monitor, in-memory DB")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the topic's FAISS index first (real mode)")
    args = parser.parse_args()

    if args.mock_llm:
        set_services(make_fake_services())
    else:
        store = get_services().crag.store
        if args.rebuild or not store.exists(args.topic):
            from knowledge_prep.build_index import build_topic_index

            print(f"Building index for '{args.topic}'...")
            build_topic_index(args.topic, settings.knowledge_samples_dir / f"{args.topic}.txt")
    app = get_services().app
    app.ensure_ready()
    student = args.student or app.create_learner("Demo Learner")

    checkpointer = get_checkpointer()
    runner = SessionRunner(build_graph(checkpointer), on_node=on_node)
    questions = iter(args.questions or DEFAULT_QUESTIONS)
    print(f"Learner {student}, topic '{args.topic}', mode: {'mock' if args.mock_llm else 'real'}\n")

    out = runner.start_session(student, args.topic)
    tid = out["thread_id"]
    while "interrupt" in out:
        pause = out["interrupt"]
        values = runner.get_state(tid)["values"]
        if pause["type"] == "ask_question" and values["response"] and pause["question_no"] > 1:
            print(textwrap.fill(values["response"], 100, initial_indent="     A: ", subsequent_indent="        "))
        if pause["type"] == "pre_test":
            reply: Any = {"answers": ["A"] * len(pause["items"])}
            print(f"\n  [pre_test] {len(pause['items'])} items, e.g. {pause['items'][0]['question']!r}")
        elif pause["type"] == "ask_question":
            query = next(questions, None)
            reply = {"query": query} if query else {"end": True}
            print(f"\n  Q{pause['question_no']}: {query}" if query else "\n  (end)")
        else:
            reply = {"answers": ["A"] * len(pause["items"]), "rating": 4, "comment": "scripted run"}
            print("\n  [post_test_and_feedback]")
        out = runner.resume(tid, reply)

    s = out["final_state"]
    checkpointer.conn.close()
    print(f"\ndecision_reason: {s['decision_reason']}")
    print(f"policy={s['policy_id']} {s['runtime_config']}  similar_exps={len(s['similar_exps'])}")
    print(f"pre_score={s['pre_score']} post_score={s['post_score']} rating={s['rating']} evaluation={s['evaluation']}")
    print("qa_log:")
    for e in s["qa_log"]:
        print(f"  Q{e['question_no']} score={e['retrieval_score']:.3f} retries={e['crag_retries']} "
              f"chunks={e['n_chunks']} preferred={e['preferred_route']} model={e['model_used']} "
              f"latency={e['latency_ms']:.0f}ms (gen {e['gen_latency_ms']:.0f}ms) error={e['error']}")
    if s["session_errors"]:
        print("session_errors:", *s["session_errors"], sep="\n  ")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        from telemetry import tracing

        tracing.shutdown()  # send buffered Langfuse data before the script exits
    sys.exit(code)
