"""Case-based policy selection as pure functions (no I/O, no LangGraph).

Current context → normalised vector → k nearest stored experiences → mean reward per
predefined policy → argmax (ties go to the default), with ε-exploration.
This is not reinforcement learning: nothing is trained; past experiences are only looked up.
"""
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Vector order; matches Person B's EXPERIENCE_STATE_KEYS.
STATE_KEYS: tuple[str, ...] = ("rtt_ms", "cpu", "mem", "competency_num", "last_qoe")
_TIE_EPS = 1e-9


@dataclass(frozen=True)
class Neighbour:
    """One stored experience close to the current context."""

    experience: Mapping[str, Any]
    distance: float


@dataclass(frozen=True)
class PolicyDecision:
    """The chosen policy plus everything needed to explain it."""

    policy_id: str
    reason: str
    mode: str  # "cold_start" | "greedy" | "exploration"
    utilities: dict[str, float] = field(default_factory=dict)
    neighbours: list[Neighbour] = field(default_factory=list)


def normalize_state(context: Mapping[str, Any], *, rtt_max_ms: float, default_last_qoe: float) -> np.ndarray:
    """Map a context dict to a vector in [0, 1]^5 (RTT clipped at rtt_max_ms; missing last_qoe → default)."""
    last_qoe = context.get("last_qoe")
    raw = [
        float(context.get("rtt_ms", rtt_max_ms)) / rtt_max_ms,
        float(context.get("cpu", 0.0)),
        float(context.get("mem", 0.0)),
        float(context.get("competency_num", 0.0)),
        default_last_qoe if last_qoe is None else float(last_qoe),
    ]
    return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)


def nearest_experiences(
    vector: np.ndarray,
    experiences: Sequence[Mapping[str, Any]],
    k: int,
    *,
    rtt_max_ms: float,
    default_last_qoe: float,
) -> list[Neighbour]:
    """Return the k experiences with a reward whose normalised state is closest (Euclidean) to `vector`."""
    usable = [e for e in experiences if e.get("reward") is not None and e.get("state")]
    if not usable or k <= 0:
        return []
    matrix = np.stack([normalize_state(e["state"], rtt_max_ms=rtt_max_ms, default_last_qoe=default_last_qoe)
                       for e in usable])
    distances = np.linalg.norm(matrix - vector, axis=1)
    order = np.argsort(distances, kind="stable")[:k]
    return [Neighbour(experience=usable[i], distance=float(distances[i])) for i in order]


def estimate_utility(neighbours: Sequence[Neighbour]) -> dict[str, float]:
    """Return {policy_id: mean reward} over the neighbours that used each policy."""
    rewards: dict[str, list[float]] = {}
    for n in neighbours:
        rewards.setdefault(str(n.experience["policy_id"]), []).append(float(n.experience["reward"]))
    return {pid: float(np.mean(vals)) for pid, vals in sorted(rewards.items())}


def choose_policy(
    context: Mapping[str, Any],
    experiences: Sequence[Mapping[str, Any]],
    *,
    policy_ids: Sequence[str],
    default_policy_id: str,
    k: int,
    min_experiences: int,
    epsilon: float,
    rng: random.Random,
    rtt_max_ms: float,
    default_last_qoe: float,
) -> PolicyDecision:
    """Pick a predefined policy: cold start → default; else ε-explore, else argmax mean reward (tie → default)."""
    known = [e for e in experiences if e.get("reward") is not None and e.get("policy_id") in policy_ids]
    if len(known) < min_experiences:
        return PolicyDecision(default_policy_id, f"cold start: {len(known)} experience(s) < "
                              f"MIN_EXPERIENCES={min_experiences} → default {default_policy_id}", "cold_start")

    vector = normalize_state(context, rtt_max_ms=rtt_max_ms, default_last_qoe=default_last_qoe)
    neighbours = nearest_experiences(vector, known, k, rtt_max_ms=rtt_max_ms, default_last_qoe=default_last_qoe)
    utilities = estimate_utility(neighbours)
    if not utilities:
        return PolicyDecision(default_policy_id, f"cold start: no neighbour data → default {default_policy_id}",
                              "cold_start", utilities, neighbours)

    best_value = max(utilities.values())
    tied = sorted(pid for pid, u in utilities.items() if best_value - u <= _TIE_EPS)
    greedy = default_policy_id if default_policy_id in tied else tied[0]

    # Draw once per decision so a seeded RNG gives the same sequence of choices.
    if rng.random() < epsilon:
        others = [pid for pid in policy_ids if pid != greedy]
        if others:
            explored = rng.choice(sorted(others))
            return PolicyDecision(explored, f"exploration (ε={epsilon}): {explored} instead of greedy {greedy}",
                                  "exploration", utilities, neighbours)

    n_used = sum(1 for n in neighbours if n.experience["policy_id"] == greedy)
    tie_rule = "default wins" if greedy == default_policy_id else "lowest id wins"
    tie_note = f" (tied with {', '.join(p for p in tied if p != greedy)}; {tie_rule})" if len(tied) > 1 else ""
    reason = (f"{greedy}: mean reward {utilities[greedy]:.2f} over {n_used} of {len(neighbours)} "
              f"neighbours{tie_note}")
    return PolicyDecision(greedy, reason, "greedy", utilities, neighbours)
