"""Entropy Weight Method (EWM) for one weight group, from a matrix of observations.

For n observations × m metrics (all "higher = better" or deltas):
    x'_ij = (x_ij − min_j) / (max_j − min_j)                (min–max per column)
    p_ij  = x'_ij / Σ_i x'_ij
    e_j   = −(1 / ln n) · Σ_i p_ij · ln p_ij               (0 · ln 0 := 0)
    d_j   = 1 − e_j                                          (divergence)
    w_j   = d_j / Σ_j d_j

Rows with any missing value are dropped (complete cases). A constant column carries no information
(d_j = 0). Fallback to the baseline weights, with the reason recorded, when there are fewer than
`min_observations` complete rows, every column is constant, or the result is not finite.

EWM measures how much a metric *varies* in the observed data, not how important it is educationally
or operationally. Learned weights are therefore stored, inspectable, compared with the baseline and
only used when explicitly activated. Estimate each group from its own metrics only.
"""
import math
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel


class EWMResult(BaseModel):
    group: str
    weights: dict[str, float]  # the weights to use (EWM, or the baseline on fallback)
    method: str  # "ewm" | "fallback"
    fallback_reason: str | None = None
    entropy: dict[str, float | None]
    divergence: dict[str, float | None]
    n_observations: int
    n_used: int
    baseline: dict[str, float]

    def comparison(self) -> list[dict[str, float | str]]:
        """Baseline vs. derived weight per metric."""
        return [{"metric": k, "baseline": self.baseline[k], "derived": self.weights[k],
                 "difference": round(self.weights[k] - self.baseline[k], 6)} for k in self.baseline]


def entropy_weights(group: str, observations: Sequence[Mapping[str, float | None]], keys: Sequence[str],
                    baseline: Mapping[str, float], *, min_observations: int = 10) -> EWMResult:
    """Derive weights for `keys` from observations; fall back to `baseline` when EWM is not meaningful."""
    keys = list(keys)
    rows = [[float(o[k]) for k in keys] for o in observations
            if all(o.get(k) is not None and math.isfinite(float(o[k])) for k in keys)]
    base = {k: float(baseline[k]) for k in keys}

    def fallback(reason: str, entropy=None, divergence=None) -> EWMResult:
        return EWMResult(group=group, weights=base, method="fallback", fallback_reason=reason,
                         entropy=entropy or {k: None for k in keys}, divergence=divergence or {k: None for k in keys},
                         n_observations=len(observations), n_used=len(rows), baseline=base)

    if len(rows) < max(min_observations, 2):
        return fallback(f"only {len(rows)} complete observations (need {max(min_observations, 2)})")
    x = np.asarray(rows, dtype=float)
    span = x.max(axis=0) - x.min(axis=0)
    scaled = np.divide(x - x.min(axis=0), span, out=np.zeros_like(x), where=span > 0)
    totals = scaled.sum(axis=0)
    p = np.divide(scaled, totals, out=np.zeros_like(scaled), where=totals > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    entropy = np.where(totals > 0, -plogp.sum(axis=0) / math.log(len(rows)), 1.0)  # constant column → e = 1
    divergence = np.clip(1.0 - entropy, 0.0, None)
    ent = {k: round(float(e), 6) for k, e in zip(keys, entropy)}
    div = {k: round(float(d), 6) for k, d in zip(keys, divergence)}
    if divergence.sum() <= 0:
        return fallback("every metric is constant in the window (no information)", ent, div)
    weights = divergence / divergence.sum()
    if not np.all(np.isfinite(weights)):
        return fallback("non-finite weights", ent, div)
    rounded = {k: round(float(w), 6) for k, w in zip(keys, weights)}
    rounded[keys[-1]] = round(1.0 - sum(rounded[k] for k in keys[:-1]), 6)  # exact sum to 1 after rounding
    return EWMResult(group=group, weights=rounded, method="ewm", entropy=ent, divergence=div,
                     n_observations=len(observations), n_used=len(rows), baseline=base)
