"""Shared fixtures: fake services on an in-memory DB with Person B's tables, and an in-memory runner."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.deps import make_fake_services, set_services
from agent.graph import build_graph
from agent.runner import SessionRunner

ANSWERS = ["A", "B", "C", "D", "A"]  # all correct for FakeMCQProvider's items
CTX = {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 0.0, "last_qoe": 0.5}


@pytest.fixture
def services():
    """Fake retrieval/LLMs/monitor; rng seed 0 (first draw 0.84 ≥ ε, so selection is greedy)."""
    svc = make_fake_services()
    set_services(svc)
    yield svc
    set_services(None)


@pytest.fixture
def visited():
    """Collects node names as the runner executes them."""
    return []


@pytest.fixture
def runner(services, visited):
    """Runner over an in-memory checkpointer that records visited nodes."""
    return SessionRunner(build_graph(InMemorySaver()), on_node=lambda node, _: visited.append(node))


@pytest.fixture
def learner(services):
    """A learner row in the fake DB."""
    return services.app.create_learner("Test Learner")


def seed_experiences(services, rows, context=None):
    """Store (policy_id, value) transitions through the real experience store (DB + FAISS).

    Each transition's s_next has QoE = SRE = value, so its observed utility is exactly `value`.
    """
    from agent.experience_store import record_synthetic_transition
    from agent.policies import POLICIES

    for policy_id, value in rows:
        record_synthetic_transition(services.experience_store(), services.app, policy_id=policy_id,
                                    runtime_config=POLICIES[policy_id].runtime_config(), context=context or CTX,
                                    qoe=value, sre=value)
