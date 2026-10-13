"""Tests for collect_context and select_policy against fakes and an in-memory DB with B's tables."""
from agent.nodes.context import collect_context
from agent.nodes.policy import select_policy
from agent.policies import POLICIES
from agent.state import initial_state
from app.core.config import POLICY_SEED
from config import settings
from services.fakes import FakeMonitor
from tests.conftest import seed_experiences


def test_collect_context_uses_monitor_and_default_last_qoe(services):
    student = services.app.create_learner("x")
    update = collect_context(initial_state(student, "photosynthesis") | {"competency": "intermediate"})
    assert update["context"] == {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 0.5,
                                 "last_qoe": settings.default_last_qoe}
    assert update["session_errors"] == []


def test_collect_context_probe_failure_uses_rtt_fail_ms(services):
    services.monitor = FakeMonitor(rtt_ms=None)
    update = collect_context(initial_state(services.app.create_learner("x"), "photosynthesis"))
    assert update["context"]["rtt_ms"] == settings.rtt_fail_ms
    assert "RTT_FAIL_MS" in update["session_errors"][0]


def test_collect_context_reads_last_qoe_from_experiences(services):
    seed_experiences(services, [("P2", 0.7)])
    student = services.app.get_experiences()[0]["student_id"]
    assert collect_context(initial_state(student, "photosynthesis"))["context"]["last_qoe"] == 0.7


def test_select_policy_cold_start(services):
    update = select_policy(initial_state(services.app.create_learner("x"), "t") | {
        "context": {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 0.0, "last_qoe": 0.5}})
    assert update["policy_id"] == "P1"
    assert update["decision_reason"].startswith("cold start")
    assert update["runtime_config"] == {"top_k": 3, "threshold": settings.crag_retrieval_threshold,
                                        "max_retries": 0, "route": "local"}


def test_select_policy_greedy_reads_runtime_config_from_db(services):
    services.rng.seed(3)  # first draw is 0.238 ≥ ε=0.2, so the choice is greedy
    seed_experiences(services, [("P4", 0.9), ("P4", 0.8), ("P1", 0.2), ("P2", 0.3)])
    ctx = {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 0.0, "last_qoe": 0.5}
    update = select_policy(initial_state(services.app.create_learner("x"), "t") | {"context": ctx})
    assert update["policy_id"] == "P4"
    assert update["decision_reason"].startswith("P4: highest estimated utility 0.850")  # equal distances → mean
    assert update["candidate_utilities"]["P4"]["support"] == 2
    assert update["candidate_utilities"]["P3"]["estimated"] is None  # never executed nearby
    assert update["runtime_config"] == {"top_k": 5, "threshold": settings.crag_retrieval_threshold,
                                        "max_retries": 1, "route": "cloud"}
    assert len(update["similar_exps"]) == 4
    assert update["prev_state"]["baseline"] is True  # learner without history → baseline s


def test_agent_policies_match_db_seed():
    seed = {p["policy_id"]: p for p in POLICY_SEED}
    assert set(seed) == set(POLICIES)
    for pid, policy in POLICIES.items():
        assert (policy.top_k, policy.max_retries, policy.route) == \
            (seed[pid]["top_k"], seed[pid]["max_retries"], seed[pid]["route"])
    assert next(p for p in POLICY_SEED if p["is_default"])["policy_id"] == settings.default_policy_id
