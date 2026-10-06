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


def seed_experiences(app, rows, context=None):
    """Store (policy_id, reward) experiences through B's store_experience, one learner/session each."""
    for policy_id, reward in rows:
        student = app.create_learner("seed")
        session = app.create_learning_session(student)
        app.store_experience(student, session, "photosynthesis", policy_id, context or CTX,
                             {"qoe_before": 0.5, "qoe_after": 0.6, "sre_before": 0.5, "sre_after": 0.6,
                              "reward": reward})
