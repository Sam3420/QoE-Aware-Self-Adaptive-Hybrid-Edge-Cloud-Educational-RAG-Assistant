"""Normalisation to [0, 1] where higher is always better.

Every function returns None (= missing) for absent, non-numeric, NaN/inf or physically invalid
inputs, and for invalid bounds (zero or negative ranges/denominators). None is never turned into 0
here; the aggregation's missing-data policy decides what happens to it.
"""
import math
from typing import Any


def as_number(value: Any) -> float | None:
    """A finite real number, or None (bools are rejected)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def fraction(value: Any) -> float | None:
    """A quantity that is already a 0–1 fraction (e.g. a rate); slightly out-of-range values are clipped."""
    x = as_number(value)
    if x is None or x < -1e-9 or x > 1 + 1e-9:
        return None
    return clip01(x)


def benefit(value: Any, lo: float, hi: float) -> float | None:
    """Benefit metric (more is better): lo → 0, hi → 1, linear, clipped."""
    x = as_number(value)
    if x is None or not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return None
    return clip01((x - lo) / (hi - lo))


def cost(value: Any, good: float, bad: float) -> float | None:
    """Cost metric (less is better, e.g. latency): ≤ good → 1, ≥ bad → 0, linear in between."""
    x = as_number(value)
    if x is None or x < 0 or not (math.isfinite(good) and math.isfinite(bad)) or bad <= good:
        return None
    return clip01((bad - x) / (bad - good))


def utilization_band(value: Any, target: float, saturation: float) -> float | None:
    """Resource-utilisation quality: load up to `target` is healthy (1); from `saturation` it is
    overloaded (0); linear in between. Low utilisation is not penalised (an idle edge device is fine)."""
    x = fraction(value)
    if x is None or saturation <= target:
        return None
    return cost(x, target, saturation)


def ratio_efficiency(usage: Any, maximum: Any) -> float | None:
    """1 − usage / maximum, clipped to [0, 1]; None if usage is missing/negative or the budget is not positive."""
    u, m = as_number(usage), as_number(maximum)
    if u is None or m is None or u < 0 or m <= 0:
        return None
    return clip01(1.0 - u / m)


def stars(value: Any, lo: int = 1, hi: int = 5) -> float | None:
    """Star rating lo..hi → [0, 1]."""
    x = as_number(value)
    if x is None or x < lo or x > hi:
        return None
    return (x - lo) / (hi - lo)


def signed(value: Any) -> float | None:
    """A score in [−1, 1] (e.g. sentiment, Hake gain clipped) → [0, 1]."""
    x = as_number(value)
    if x is None:
        return None
    return clip01((max(-1.0, min(1.0, x)) + 1.0) / 2.0)
