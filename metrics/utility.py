"""Utility U(s, a) = λ1·QoE + λ2·SRE (λ1 + λ2 = 1), on normalised scores.

`compute_utility` scores an *observed* state. Choosing an action that has not run yet uses an
*estimate* of this value from similar past experiences (agent.policy_engine), never post-adaptation
measurements that do not exist yet.
"""
from pydantic import BaseModel

from metrics.normalization import fraction
from metrics.weights import WeightSet


class UtilityResult(BaseModel):
    utility: float
    qoe: float
    sre: float
    weights: dict[str, float]


def compute_utility(qoe: float, sre: float, weights: WeightSet) -> UtilityResult:
    """Deterministic weighted utility; QoE and SRE must be in [0, 1]."""
    q, r = fraction(qoe), fraction(sre)
    if q is None or r is None:
        raise ValueError(f"utility needs QoE and SRE in [0, 1], got qoe={qoe!r}, sre={sre!r}")
    w = weights.get("utility")
    return UtilityResult(utility=round(w["QOE"] * q + w["SRE"] * r, 6), qoe=q, sre=r, weights=w)
