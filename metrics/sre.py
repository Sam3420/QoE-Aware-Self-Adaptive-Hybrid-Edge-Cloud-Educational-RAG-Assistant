"""System Resource Efficiency: SRE = η1·CPU_eff + η2·MEM_eff + η3·EN_eff + η4·NE_eff, each in [0, 1].

    CPU_eff = 1 − CPU_usage / CPU_max        (utilisation fraction vs. the allowed budget)
    MEM_eff = 1 − Memory_usage / Memory_max
    EN_eff  = 1 − Energy_consumed / Energy_max   (Wh per session vs. a per-session budget)
    NE_eff  = 1 − Bandwidth_used / Bandwidth_max (bytes per session vs. a per-session budget)

Usage above the budget clips to 0. Missing or invalid measurements (negative usage, non-positive
budget) are unavailable and handled by the missing-data policy.

Limitation (by design): SRE alone would reward a configuration that uses fewer resources even if it
makes the service useless. It is therefore never used on its own: utility and reward always combine it
with QoE, which drops when answers, latency or learning outcomes suffer.
"""
from pydantic import BaseModel

from metrics.normalization import as_number, ratio_efficiency
from metrics.weights import Aggregate, MissingPolicy, WeightSet, aggregate


class SREInputs(BaseModel):
    """Resource usage over one session and the budgets it is measured against."""

    cpu_usage: float | None = None  # mean utilisation fraction (0–1)
    cpu_max: float = 1.0
    memory_usage: float | None = None  # mean utilisation fraction (0–1)
    memory_max: float = 1.0
    energy_wh: float | None = None
    energy_max_wh: float = 2.0
    energy_estimated: bool = False  # True when energy comes from the CPU power model, not a meter
    bandwidth_bytes: float | None = None
    bandwidth_max_bytes: float = 50e6
    bandwidth_scope: str | None = None  # e.g. "system-wide" when measured with psutil counters


class SREResult(BaseModel):
    sre: float | None
    components: dict[str, float | None]  # CPU, MEM, EN, NE efficiencies
    detail: Aggregate
    notes: list[str]


def compute_sre(inputs: SREInputs, weights: WeightSet, policy: MissingPolicy = "renormalize") -> SREResult:
    """Efficiency per resource, then the weighted SRE."""
    components = {
        "CPU": ratio_efficiency(inputs.cpu_usage, inputs.cpu_max),
        "MEM": ratio_efficiency(inputs.memory_usage, inputs.memory_max),
        "EN": ratio_efficiency(inputs.energy_wh, inputs.energy_max_wh),
        "NE": ratio_efficiency(inputs.bandwidth_bytes, inputs.bandwidth_max_bytes),
    }
    detail = aggregate(components, weights.get("sre"), policy)
    notes = []
    if inputs.energy_estimated and components["EN"] is not None:
        notes.append("energy estimated from a linear CPU power model (no meter)")
    if inputs.bandwidth_scope and components["NE"] is not None:
        notes.append(f"bandwidth measured {inputs.bandwidth_scope}")
    return SREResult(sre=detail.score, components=components, detail=detail, notes=notes)


def estimate_energy_wh(cpu_utilization: float | None, active_seconds: float | None, *, idle_power_w: float,
                       max_power_w: float) -> float | None:
    """Energy ≈ (P_idle + (P_max − P_idle)·u) · t for the time the pipeline was busy. A rough estimate:
    it ignores GPU and remote (cloud) energy. None if inputs are missing or invalid."""
    u, t = as_number(cpu_utilization), as_number(active_seconds)
    if u is None or t is None or t < 0 or not 0 <= u <= 1 or max_power_w < idle_power_w:
        return None
    return round((idle_power_w + (max_power_w - idle_power_w) * u) * t / 3600.0, 6)
