"""Reward of a completed transition (s, a, s_next):  R = μ1·ΔQoE + μ2·ΔSRE,  μ1 + μ2 = 1.

QoE_old/SRE_old are recomputed from the metrics stored with s, and QoE_new/SRE_new from s_next, with
the same weights and policy, so the reward is deterministic for given inputs and configuration.
Improvement → R > 0, regression → R < 0, no change → R = 0. When s has no measured predecessor (a
learner's first session) the configured baseline scores are used and flagged.

This module only computes; storing (s, a, R, s_next) is done by agent.experience_store, which makes
the write idempotent per session.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from metrics.qoe import QoEMetrics, QoEResult, compute_qoe
from metrics.sre import SREInputs, SREResult, compute_sre
from metrics.utility import compute_utility
from metrics.weights import MissingPolicy, WeightSet

ZERO_TOLERANCE = 1e-9


class IncompleteTransition(ValueError):
    """The post-adaptation state has no measurable QoE or SRE, so no reward can be computed."""


class RuntimeState(BaseModel):
    """A measured runtime state: the selection context plus the metrics observed in it."""

    context: dict[str, float | None]  # rtt_ms, cpu, mem, competency_num, last_qoe (what kNN embeds)
    qoe_metrics: QoEMetrics | None = None
    sre_inputs: SREInputs | None = None
    session_id: str | None = None
    measured_at: datetime | None = None
    baseline: bool = False  # True: no measured predecessor; scores come from the configured baseline
    coach: dict[str, Any] | None = None  # coach_mode and the session's coaching outcomes (per-concept mastery gain)


class AdaptationAction(BaseModel):
    """An executed (or candidate) runtime adaptation."""

    action_id: str  # e.g. "P2"
    config: dict[str, Any] = Field(default_factory=dict)  # e.g. {"top_k": 5, "max_retries": 1, ...}


class StateScore(BaseModel):
    qoe: float
    sre: float
    utility: float
    qoe_detail: QoEResult | None = None
    sre_detail: SREResult | None = None
    baseline_parts: list[str] = Field(default_factory=list)  # which of "qoe"/"sre" came from the baseline


class RewardResult(BaseModel):
    reward: float
    delta_qoe: float
    delta_sre: float
    before: StateScore
    after: StateScore
    action: AdaptationAction
    weights: dict[str, float]  # μ
    weights_version: str
    baseline_used: bool


def score_state(state: RuntimeState, weights: WeightSet, policy: MissingPolicy, *, baseline_qoe: float,
                baseline_sre: float, allow_baseline: bool) -> StateScore:
    """QoE, SRE and utility of one state. Without allow_baseline, unmeasurable QoE/SRE raises."""
    qoe_res = None if state.baseline or state.qoe_metrics is None else compute_qoe(state.qoe_metrics, weights, policy)
    sre_res = None if state.baseline or state.sre_inputs is None else compute_sre(state.sre_inputs, weights, policy)
    qoe = qoe_res.qoe if qoe_res else None
    sre = sre_res.sre if sre_res else None
    baseline_parts = []
    if qoe is None or sre is None:
        if not allow_baseline:
            raise IncompleteTransition(f"state {state.session_id or '?'} has no measurable "
                                       f"{'QoE' if qoe is None else 'SRE'}")
        if qoe is None:
            qoe, baseline_parts = baseline_qoe, baseline_parts + ["qoe"]
        if sre is None:
            sre, baseline_parts = baseline_sre, baseline_parts + ["sre"]
    utility = compute_utility(qoe, sre, weights).utility
    return StateScore(qoe=qoe, sre=sre, utility=utility, qoe_detail=qoe_res, sre_detail=sre_res,
                      baseline_parts=baseline_parts)


def compute_reward(s: RuntimeState, s_next: RuntimeState, action: AdaptationAction, weights: WeightSet, *,
                   policy: MissingPolicy = "renormalize", baseline_qoe: float = 0.5,
                   baseline_sre: float = 0.5) -> RewardResult:
    """R = μ1·(QoE_new − QoE_old) + μ2·(SRE_new − SRE_old). Raises IncompleteTransition if s_next is unmeasurable."""
    before = score_state(s, weights, policy, baseline_qoe=baseline_qoe, baseline_sre=baseline_sre, allow_baseline=True)
    after = score_state(s_next, weights, policy, baseline_qoe=baseline_qoe, baseline_sre=baseline_sre,
                        allow_baseline=False)
    mu = weights.get("reward")
    delta_qoe = round(after.qoe - before.qoe, 6)
    delta_sre = round(after.sre - before.sre, 6)
    reward = mu["QOE"] * delta_qoe + mu["SRE"] * delta_sre
    reward = 0.0 if abs(reward) < ZERO_TOLERANCE else round(reward, 6)
    return RewardResult(reward=reward, delta_qoe=delta_qoe, delta_sre=delta_sre, before=before, after=after,
                        action=action, weights=mu, weights_version=weights.version,
                        baseline_used=bool(before.baseline_parts))
