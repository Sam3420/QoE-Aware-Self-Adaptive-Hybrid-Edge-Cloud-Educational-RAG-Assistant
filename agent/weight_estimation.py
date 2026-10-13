"""Entropy-weight estimation over stored transitions, one weight group at a time.

Each group is estimated only from its own metrics, observed in the post-adaptation state of each
stored transition:

| Group | Observations |
|---|---|
| eq, aq, nq, ux | the normalised metrics of s_next |
| qoe | EQ, AQ, NQ, UX component scores of s_next |
| sre | CPU, MEM, EN, NE efficiencies of s_next |
| utility | (QoE_after, SRE_after) |
| reward | (ΔQoE, ΔSRE) |

Results are stored in `weight_sets` with the data window. They are used only if activated *and*
METRIC_WEIGHTS_SOURCE=ewm. Remember: EWM reflects how much a metric varies, not how much it matters.
"""
from typing import Any

from agent.experience_store import ExperienceStore
from app.db.repositories import AdaptiveLearningRepository
from config import Settings, settings
from metrics.ewm import EWMResult, entropy_weights
from metrics.weights import GROUP_KEYS, WeightSet

METRIC_GROUPS = ("eq", "aq", "nq", "ux")


def observations(group: str, transitions: list[dict[str, Any]]) -> list[dict[str, float | None]]:
    """The observation rows for one group, from stored transitions."""
    rows = []
    for t in transitions:
        after = (t.get("scores") or {}).get("after") or {}
        if group in METRIC_GROUPS:
            metrics = (t.get("next_state") or {}).get("qoe_metrics") or {}
            rows.append({k: metrics.get(k) for k in GROUP_KEYS[group]})
        elif group == "qoe":
            rows.append(dict((after.get("qoe_detail") or {}).get("components") or {}))
        elif group == "sre":
            rows.append(dict((after.get("sre_detail") or {}).get("components") or {}))
        elif group == "utility":
            rows.append({"QOE": t.get("qoe_after"), "SRE": t.get("sre_after")})
        elif group == "reward":
            scores = t.get("scores") or {}
            rows.append({"QOE": scores.get("delta_qoe"), "SRE": scores.get("delta_sre")})
    return rows


def estimate_group(store: ExperienceStore, group: str, *, activate: bool = False,
                   cfg: Settings = settings) -> tuple[EWMResult, dict[str, Any]]:
    """Run EWM for one group over every stored transition, persist the result, optionally activate it.

    Only an "ewm" result can be activated; a fallback is stored for the record but never activated.
    """
    if group not in GROUP_KEYS:
        raise ValueError(f"unknown weight group {group!r}; choose from {sorted(GROUP_KEYS)}")
    with store._db() as db:
        transitions = AdaptiveLearningRepository(db).list_transitions()
    baseline = WeightSet.from_config(cfg.metric_weights).get(group)
    result = entropy_weights(group, observations(group, transitions), GROUP_KEYS[group], baseline,
                             min_observations=cfg.ewm_min_observations)
    times = [t["created_at"] for t in transitions]
    with store._db() as db:
        record = AdaptiveLearningRepository(db).save_weight_set(
            group=group, method=result.method, weights=result.weights, baseline=result.baseline,
            details={"entropy": result.entropy, "divergence": result.divergence,
                     "fallback_reason": result.fallback_reason, "comparison": result.comparison()},
            n_observations=result.n_observations, n_used=result.n_used,
            window_start=min(times) if times else None, window_end=max(times) if times else None,
            data_version=f"{len(transitions)}:{transitions[0]['id'] if transitions else '-'}",
            activate=activate and result.method == "ewm")
    return result, record
