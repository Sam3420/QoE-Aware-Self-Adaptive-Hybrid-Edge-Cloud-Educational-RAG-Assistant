"""Run one full session offline through the runner API and check the outcome.

Fakes: retrieval, LLMs, monitor and MCQ generation. Real: the graph, CRAG/router logic, policy
selection, and Person B's DB code on a private in-memory SQLite DB.
Scenario: 3 seeded experiences make kNN pick P2 (1 CRAG retry allowed); pre-test; Q1 is vague and
triggers one retry; a blank question is re-asked; Q2; {"end": true}; post-test with feedback.
Exits non-zero if the visited path or the final state is not as expected.
"""
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.deps import make_fake_services, set_services  # noqa: E402
from agent.graph import build_graph, get_checkpointer  # noqa: E402
from agent.runner import SessionRunner  # noqa: E402

ANSWERS = ["A", "B", "C", "D", "A"]  # all correct for the fake MCQ items
RESUMES: list[Any] = [
    {"answers": ANSWERS},  # pre_test
    "photosynthesis?",  # vague: triggers 1 CRAG retry (bare string = {"query": ...})
    {"query": "   "},  # invalid: blank, so ask_question re-asks
    {"query": "How do plants convert sunlight into chemical energy?"},
    {"end": True},
    {"answers": ["A", "B", "C", "A", "A"], "rating": 4, "comment": "Clear answers, a bit slow."},
]
EXPECTED_PATH = [
    "load_profile", "pre_test", "collect_context", "select_policy",
    "ask_question", "crag_retrieve", "corrective_retrieve", "crag_retrieve", "llm_router", "generate_response",
    "ask_question", "crag_retrieve", "llm_router", "generate_response",
    "ask_question", "post_test_and_feedback", "evaluate_session", "store_experience",
]
CTX = {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 1.0, "last_qoe": 0.5}


def seed(services) -> None:
    """Store 3 past transitions (through the real experience store + FAISS) in which P2 did best."""
    from agent.experience_store import record_synthetic_transition
    from agent.policies import POLICIES

    for policy_id, qoe, sre in [("P2", 0.9, 0.7), ("P2", 0.85, 0.7), ("P1", 0.5, 0.8)]:
        record_synthetic_transition(services.experience_store(), services.app, policy_id=policy_id,
                                    runtime_config=POLICIES[policy_id].runtime_config(), context=CTX, qoe=qoe, sre=sre)


def main() -> int:
    """Drive the session and check the outcome."""
    sys.stdout.reconfigure(encoding="utf-8")
    services = make_fake_services()
    set_services(services)
    seed(services)
    student = services.app.create_learner("Demo Learner")

    visited: list[str] = []

    def on_node(node: str, update: dict[str, Any]) -> None:
        visited.append(node)
        print(f"  -> {node:<24} updates: {', '.join(sorted(update)) or '-'}")

    checkpointer = get_checkpointer()
    runner = SessionRunner(build_graph(checkpointer), on_node=on_node)
    out = runner.start_session(student, "photosynthesis")
    tid = out["thread_id"]
    print(f"Session {tid} (checkpoints: data/checkpoints.db)\n")
    reasked = False
    resumes = iter(RESUMES)
    while "interrupt" in out:
        pause = out["interrupt"]
        reasked |= "error" in pause
        print(f"  || interrupt: {pause['type']}" + (f"  [error: {pause['error']}]" if "error" in pause else ""))
        value = next(resumes)
        print(f"  >> resume with: {value!r}")
        out = runner.resume(tid, value)

    s = out["final_state"]
    finished = runner.get_state(tid)["finished"]
    checkpointer.conn.close()

    print(f"\ndecision_reason: {s['decision_reason']}")
    for key in ("policy_id", "runtime_config", "competency", "pre_score", "post_score", "rating"):
        print(f"  {key:<15} {s[key]}")
    ev = s["evaluation"]
    print(f"  evaluation      QoE {ev.get('qoe_before')} -> {ev.get('qoe_after')}, SRE {ev.get('sre_before')} -> "
          f"{ev.get('sre_after')}, reward {ev.get('reward')}, predicted utility {ev.get('predicted_utility')} vs "
          f"observed {ev.get('utility_after')}")
    print(f"  QoE components  {ev.get('qoe_components')} (coverage {ev.get('qoe_coverage')})")
    print("  qa_log:")
    for e in s["qa_log"]:
        print(f"    Q{e['question_no']} {e['query']!r} score={e['retrieval_score']} retries={e['crag_retries']} "
              f"model={e['model_used']} error={e['error']}")

    q1, q2 = (s["qa_log"] + [{}, {}])[:2]
    checks = {
        "path matches expected": visited == EXPECTED_PATH,
        "graph finished": finished,
        "FAISS policy engine chose P2 greedily": s["policy_id"] == "P2"
        and s["decision_reason"].startswith("P2: highest estimated utility"),
        "invalid reply was re-asked": reasked,
        "Q1 had 1 CRAG retry with a rewritten query": q1.get("crag_retries") == 1
        and q1.get("retrieval_query") != q1.get("query"),
        "Q2 had 0 CRAG retries": q2.get("crag_retries") == 0,
        "answers came from the local route": all(e["model_used"].startswith("local:") for e in s["qa_log"]),
        "pre/post-test scored by B's services": s["pre_score"] == 1.0 and s["post_score"] == 0.8,  # 4 of 5 correct
        "reward computed from measured QoE/SRE": s["evaluation"].get("status") == "completed"
        and s["evaluation"]["qoe_after"] is not None and s["evaluation"]["predicted_utility"] is not None,
        "transition stored in DB and FAISS": any(e["session_id"] == tid for e in services.app.get_experiences())
        and services.experience_store().count() == 4,
        "no session errors": not s["session_errors"],
    }
    print("\nChecks:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not checks["path matches expected"]:
        print(f"  visited:  {visited}\n  expected: {EXPECTED_PATH}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
