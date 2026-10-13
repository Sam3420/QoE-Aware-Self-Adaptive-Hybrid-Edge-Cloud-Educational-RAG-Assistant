"""Adaptive policy selection: a* = argmax_a Û(s, a) over the predefined candidate actions.

1. Embed the current context s (agent.experience_store.StateEmbedder) and query FAISS for the top-K
   most similar past states; load their full records from the database.
2. For each candidate a, use only the neighbours that executed a (same action id). Weight each by
   state similarity w_i = exp(−d_i² / 2σ²), and estimate
       Û(s, a) = Σ_i w_i · U_i / Σ_i w_i,
   where U_i = λ1·QoE_after_i + λ2·SRE_after_i is recomputed from neighbour i's stored post-adaptation
   metrics with the current weights (or its stored reward R_i when POLICY_ESTIMATE_TARGET=reward).
   An action needs at least POLICY_MIN_SUPPORT such neighbours AND a total similarity weight of at
   least POLICY_MIN_WEIGHT_SUM; otherwise it has no estimate (evidence from very different states is
   not trusted, however good its value).
3. Selection:
   - cold start (fewer than MIN_EXPERIENCES stored transitions) → the safe default policy;
   - no candidate has an estimate → the safe default policy (fallback);
   - otherwise ε-greedy with a seeded RNG: with probability ε a uniformly random other candidate
     (this is how unseen actions get tried); else the highest Û, with ties going to the default.

The estimate is a prediction made before execution. The observed post-adaptation QoE/SRE/reward are
measured later and stored separately (experience_transitions.predicted_utility vs. utility_after).
"""
import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from agent.experience_store import ExperienceStore
from config import Settings, settings
from metrics.reward import IncompleteTransition, RuntimeState, score_state
from metrics.weights import WeightSet

_TIE_EPS = 1e-9


class ActionEstimate(BaseModel):
    action_id: str
    estimated_value: float | None  # Û(s, a); None = no usable evidence
    support: int  # neighbours that executed this action
    weight_sum: float
    evidence: list[dict[str, Any]] = Field(default_factory=list)  # [{experience_id, distance, weight, value}]


class SelectionResult(BaseModel):
    action_id: str
    mode: str  # "cold_start" | "fallback" | "greedy" | "exploration"
    reason: str
    target: str  # "utility" | "reward"
    estimates: dict[str, ActionEstimate]
    n_history: int
    neighbours: list[dict[str, Any]]  # every retrieved neighbour, any action

    @property
    def predicted_value(self) -> float | None:
        return self.estimates[self.action_id].estimated_value if self.action_id in self.estimates else None


def similarity_weight(distance: float, sigma: float) -> float:
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))


def neighbour_value(record: dict[str, Any], weights: WeightSet, target: str, cfg: Settings) -> float | None:
    """Observed value of a past transition under the current weights (utility of s_next, or its reward)."""
    if target == "reward":
        return record.get("reward")
    try:
        state = RuntimeState.model_validate(record["next_state"])
        return score_state(state, weights, cfg.metric_missing_policy, baseline_qoe=cfg.baseline_qoe,
                           baseline_sre=cfg.baseline_sre, allow_baseline=False).utility
    except (IncompleteTransition, KeyError, ValueError):
        return record.get("utility_after")


def estimate_action_values(neighbours: Sequence[tuple[dict[str, Any], float]], candidates: Sequence[str], *,
                           value_fn: Callable[[dict[str, Any]], float | None], sigma: float,
                           min_support: int, min_weight_sum: float = 0.0) -> dict[str, ActionEstimate]:
    """Similarity-weighted mean of each candidate's own neighbours' observed values (pure)."""
    estimates = {}
    for action in candidates:
        evidence, weighted = [], []
        for record, distance in neighbours:
            if record.get("policy_id") != action:
                continue
            value = value_fn(record)
            if value is None:
                continue
            weight = similarity_weight(distance, sigma)
            weighted.append((weight, float(value)))
            evidence.append({"experience_id": record["id"], "session_id": record.get("session_id"),
                             "distance": round(distance, 6), "weight": round(weight, 6), "value": round(float(value), 6)})
        weight_sum = sum(w for w, _ in weighted)
        value = (sum(w * v for w, v in weighted) / weight_sum
                 if len(weighted) >= min_support and weight_sum > 0 and weight_sum >= min_weight_sum else None)
        estimates[action] = ActionEstimate(action_id=action, estimated_value=None if value is None else round(value, 6),
                                           support=len(evidence), weight_sum=round(weight_sum, 6), evidence=evidence)
    return estimates


def choose_action(estimates: dict[str, ActionEstimate], candidates: Sequence[str], default: str, *, n_history: int,
                  min_experiences: int, epsilon: float, rng: random.Random, target: str) -> tuple[str, str, str]:
    """(action, mode, reason) using the documented cold-start / fallback / ε-greedy rules (pure)."""
    if n_history < min_experiences:
        return default, "cold_start", (f"cold start: {n_history} stored transition(s) < MIN_EXPERIENCES="
                                       f"{min_experiences} → default {default}")
    eligible = {a: e for a, e in estimates.items() if e.estimated_value is not None}
    if not eligible:
        return default, "fallback", f"no similar past session executed any candidate → default {default}"
    best = max(e.estimated_value for e in eligible.values())
    tied = sorted(a for a, e in eligible.items() if best - e.estimated_value <= _TIE_EPS)
    greedy = default if default in tied else tied[0]
    if rng.random() < epsilon:  # one draw per decision: a seeded RNG reproduces the sequence
        others = sorted(a for a in candidates if a != greedy)
        if others:
            explored = rng.choice(others)
            return explored, "exploration", f"exploration (ε={epsilon}): {explored} instead of greedy {greedy}"
    summary = ", ".join(f"{a} {e.estimated_value:.3f} (n={e.support})" if e.estimated_value is not None
                        else f"{a} no evidence" for a, e in sorted(estimates.items()))
    tie = f"; tie → {'default' if greedy == default else 'lowest id'}" if len(tied) > 1 else ""
    return greedy, "greedy", f"{greedy}: highest estimated {target} {eligible[greedy].estimated_value:.3f}{tie} [{summary}]"


class PolicyEngine:
    """FAISS-backed selection over the stored transitions."""

    def __init__(self, store: ExperienceStore, rng: random.Random, cfg: Settings = settings) -> None:
        self.store = store
        self.rng = rng
        self.cfg = cfg

    def select(self, context: dict[str, Any], candidates: Sequence[str], default: str) -> SelectionResult:
        cfg = self.cfg
        n_history = self.store.count()
        neighbours = self.store.neighbours(context, cfg.knn_k) if n_history else []
        weights = self.store.weights()
        target = cfg.policy_estimate_target
        estimates = estimate_action_values(
            neighbours, candidates, value_fn=lambda r: neighbour_value(r, weights, target, cfg),
            sigma=cfg.policy_kernel_sigma, min_support=cfg.policy_min_support,
            min_weight_sum=cfg.policy_min_weight_sum)
        action, mode, reason = choose_action(estimates, candidates, default, n_history=n_history,
                                             min_experiences=cfg.min_experiences, epsilon=cfg.epsilon,
                                             rng=self.rng, target=target)
        return SelectionResult(
            action_id=action, mode=mode, reason=reason, target=target, estimates=estimates, n_history=n_history,
            neighbours=[{"experience_id": r["id"], "session_id": r.get("session_id"), "policy_id": r["policy_id"],
                         "distance": round(d, 6), "reward": r.get("reward"), "utility_after": r.get("utility_after")}
                        for r, d in neighbours])
