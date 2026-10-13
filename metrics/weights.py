"""Validated weight groups and the weighted aggregation with an explicit missing-data policy."""
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "qoe": ("EQ", "AQ", "NQ", "UX"),  # wEQ, wAQ, wNQ, wUX
    "eq": ("LG", "PT", "RA", "LS"),  # alpha1..4
    "aq": ("RQ", "LQ", "SQ", "TQ"),  # beta1..4
    "nq": ("PL", "NR", "RU", "AV"),  # gamma1..4
    "ux": ("SR", "FS", "CR"),  # delta1..3
    "sre": ("CPU", "MEM", "EN", "NE"),  # eta1..4
    "utility": ("QOE", "SRE"),  # lambda1, lambda2
    "reward": ("QOE", "SRE"),  # mu1, mu2
}
SUM_TOLERANCE = 1e-6
MissingPolicy = Literal["renormalize", "neutral", "strict"]
NEUTRAL_VALUE = 0.5


class MissingMetricError(ValueError):
    """Raised under the 'strict' policy when a weighted input is missing."""


class WeightGroup(BaseModel):
    """One group's weights: exactly the group's keys, finite, non-negative, summing to 1."""

    name: str
    weights: dict[str, float]

    @model_validator(mode="after")
    def _validate(self) -> "WeightGroup":
        expected = GROUP_KEYS.get(self.name)
        if expected is None:
            raise ValueError(f"unknown weight group {self.name!r}; known: {sorted(GROUP_KEYS)}")
        if set(self.weights) != set(expected):
            raise ValueError(f"group {self.name!r} needs exactly {list(expected)}, got {sorted(self.weights)}")
        if any(not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError(f"group {self.name!r}: weights must be finite and non-negative")
        if abs(sum(self.weights.values()) - 1.0) > SUM_TOLERANCE:
            raise ValueError(f"group {self.name!r}: weights must sum to 1 (got {sum(self.weights.values()):.6f})")
        return self


class WeightSet(BaseModel):
    """All weight groups used by one computation, with a version string recorded on every experience."""

    groups: dict[str, WeightGroup]
    source: str = "fixed"

    @field_validator("groups")
    @classmethod
    def _complete(cls, value: dict[str, WeightGroup]) -> dict[str, WeightGroup]:
        missing = set(GROUP_KEYS) - set(value)
        if missing:
            raise ValueError(f"weight set is missing groups {sorted(missing)}")
        return value

    @classmethod
    def from_config(cls, mapping: Mapping[str, Mapping[str, float]], source: str = "fixed") -> "WeightSet":
        return cls(groups={name: WeightGroup(name=name, weights=dict(w)) for name, w in mapping.items()},
                   source=source)

    def get(self, name: str) -> dict[str, float]:
        return self.groups[name].weights

    @property
    def version(self) -> str:
        """Deterministic id of these exact weights (changes whenever any weight changes)."""
        payload = json.dumps({k: g.weights for k, g in sorted(self.groups.items())}, sort_keys=True)
        return f"{self.source}:{hashlib.sha1(payload.encode()).hexdigest()[:12]}"

    def replace(self, name: str, weights: Mapping[str, float], source: str) -> "WeightSet":
        groups = dict(self.groups)
        groups[name] = WeightGroup(name=name, weights=dict(weights))
        return WeightSet(groups=groups, source=source)


class Aggregate(BaseModel):
    """Result of a weighted aggregation."""

    score: float | None  # None when nothing was available (renormalize/strict)
    coverage: float  # share of the group's weight backed by available inputs
    used: dict[str, float]
    missing: list[str]
    policy: str


def aggregate(values: Mapping[str, float | None], weights: Mapping[str, float], policy: MissingPolicy) -> Aggregate:
    """Σ w·x with the missing-data policy:

    - renormalize: Σ_available w·x / Σ_available w (None if nothing is available);
    - neutral: missing inputs count as 0.5;
    - strict: any missing input with positive weight raises MissingMetricError.
    Inputs with zero weight never count as missing.
    """
    used = {k: float(v) for k, v in values.items() if v is not None and k in weights}
    missing = [k for k, w in weights.items() if w > 0 and k not in used]
    coverage = round(sum(weights[k] for k in used), 6)
    if policy == "strict" and missing:
        raise MissingMetricError(f"missing inputs {missing}")
    if policy == "neutral":
        score = sum(w * used.get(k, NEUTRAL_VALUE) for k, w in weights.items())
    else:
        score = sum(weights[k] * x for k, x in used.items()) / coverage if coverage > 0 else None
    return Aggregate(score=None if score is None else round(min(max(score, 0.0), 1.0), 6), coverage=coverage,
                     used=used, missing=missing, policy=policy)
