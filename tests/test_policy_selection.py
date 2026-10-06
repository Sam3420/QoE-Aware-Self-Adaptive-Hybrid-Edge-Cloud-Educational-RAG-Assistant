"""Unit tests for the pure case-based policy selection functions."""
import random

import numpy as np
import pytest

from agent.policy_selection import choose_policy, estimate_utility, nearest_experiences, normalize_state

POLICY_IDS = ["P1", "P2", "P3", "P4"]
NORM = {"rtt_max_ms": 2000.0, "default_last_qoe": 0.5}
CTX = {"rtt_ms": 200.0, "cpu": 0.3, "mem": 0.5, "competency_num": 0.5, "last_qoe": 0.6}


def exp(policy_id, reward, **state):
    """Build an experience dict shaped like B's get_experiences()."""
    return {"policy_id": policy_id, "reward": reward, "session_id": f"s-{policy_id}-{reward}", "state": {**CTX, **state}}


def choose(experiences, *, epsilon=0.0, seed=0, k=5, min_experiences=3):
    return choose_policy(CTX, experiences, policy_ids=POLICY_IDS, default_policy_id="P1", k=k,
                         min_experiences=min_experiences, epsilon=epsilon, rng=random.Random(seed), **NORM)


def test_normalize_state_bounds_and_defaults():
    vec = normalize_state({"rtt_ms": 99999, "cpu": 1.5, "mem": -0.2, "competency_num": 0.5, "last_qoe": None}, **NORM)
    assert vec.shape == (5,)
    assert np.all((vec >= 0.0) & (vec <= 1.0))
    assert vec[0] == 1.0  # RTT clipped at rtt_max_ms
    assert vec[4] == 0.5  # missing last_qoe → default
    assert normalize_state({"rtt_ms": 1000, "cpu": 0, "mem": 0, "competency_num": 0, "last_qoe": 0}, **NORM)[0] == 0.5


def test_cold_start_returns_default():
    decision = choose([exp("P3", 0.9), exp("P3", 0.9)])
    assert decision.policy_id == "P1" and decision.mode == "cold_start"
    assert "cold start" in decision.reason


def test_experiences_without_reward_do_not_count():
    decision = choose([exp("P3", None), exp("P3", None), exp("P3", 0.9)])
    assert decision.mode == "cold_start"


def test_knn_picks_best_reward_policy_among_neighbours():
    near = [exp("P2", 0.8), exp("P2", 0.7), exp("P1", 0.3), exp("P3", 0.4)]
    far = [exp("P4", 1.0, rtt_ms=2000.0, cpu=1.0, mem=1.0, competency_num=0.0, last_qoe=0.0)] * 3
    decision = choose(near + far, k=4)
    assert decision.policy_id == "P2" and decision.mode == "greedy"
    assert decision.utilities == pytest.approx({"P1": 0.3, "P2": 0.75, "P3": 0.4})
    assert decision.reason == "P2: mean reward 0.75 over 2 of 4 neighbours"


def test_nearest_experiences_orders_by_distance():
    vec = normalize_state(CTX, **NORM)
    far = exp("P4", 1.0, cpu=1.0)
    close = exp("P2", 0.5)
    neighbours = nearest_experiences(vec, [far, close], k=1, **NORM)
    assert [n.experience["policy_id"] for n in neighbours] == ["P2"]
    assert neighbours[0].distance == pytest.approx(0.0)


def test_estimate_utility_is_mean_reward_per_policy():
    vec = normalize_state(CTX, **NORM)
    neighbours = nearest_experiences(vec, [exp("P1", 0.2), exp("P1", 0.4), exp("P3", 0.9)], k=3, **NORM)
    assert estimate_utility(neighbours) == pytest.approx({"P1": 0.3, "P3": 0.9})


def test_tie_goes_to_default():
    decision = choose([exp("P3", 0.5), exp("P1", 0.5), exp("P3", 0.5), exp("P1", 0.5)])
    assert decision.policy_id == "P1"
    assert "tied with P3; default wins" in decision.reason


def test_tie_without_default_takes_lowest_id():
    decision = choose([exp("P4", 0.5), exp("P2", 0.5), exp("P4", 0.5)])
    assert decision.policy_id == "P2"


def test_seeded_epsilon_is_deterministic_and_explores_non_greedy():
    experiences = [exp("P2", 0.8), exp("P2", 0.7), exp("P1", 0.3)]
    runs = [[choose(experiences, epsilon=0.5, seed=s).policy_id for s in range(20)] for _ in range(2)]
    assert runs[0] == runs[1]  # same seeds → same choices
    explored = [choose(experiences, epsilon=1.0, seed=s) for s in range(10)]
    assert all(d.mode == "exploration" and d.policy_id != "P2" for d in explored)
    assert choose(experiences, epsilon=0.0).mode == "greedy"


def test_cold_start_never_explores():
    assert choose([], epsilon=1.0).mode == "cold_start"
