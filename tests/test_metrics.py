"""Unit tests: normalisation, weights, QoE, SRE, utility, reward, EWM and sentiment (pure, synthetic data)."""
import math

import pytest
from pydantic import ValidationError

from config import settings
from metrics.ewm import entropy_weights
from metrics.normalization import benefit, cost, ratio_efficiency, signed, stars, utilization_band
from metrics.qoe import QoEBounds, QoEMetrics, QoERaw, compute_qoe, normalize_qoe
from metrics.reward import AdaptationAction, IncompleteTransition, RuntimeState, compute_reward
from metrics.sentiment import lexicon_sentiment
from metrics.sre import SREInputs, compute_sre, estimate_energy_wh
from metrics.utility import compute_utility
from metrics.weights import GROUP_KEYS, MissingMetricError, WeightGroup, WeightSet, aggregate

W = WeightSet.from_config(settings.metric_weights)
ALL = list(QoEMetrics.model_fields)


def metrics(value=1.0, **overrides):
    return QoEMetrics(**({k: value for k in ALL} | overrides))


# ---- normalisation -------------------------------------------------------------------------------

def test_normalisers_handle_direction_bounds_and_invalid_input():
    assert benefit(0.85, 0.4, 0.85) == 1.0 and benefit(0.4, 0.4, 0.85) == 0.0
    assert cost(1000, 1500, 15000) == 1.0 and cost(15000, 1500, 15000) == 0.0  # lower latency = better
    assert cost(8250, 1500, 15000) == 0.5
    assert utilization_band(0.5, 0.7, 0.95) == 1.0 and utilization_band(0.95, 0.7, 0.95) == 0.0
    assert ratio_efficiency(0.25, 1.0) == 0.75 and ratio_efficiency(2.0, 1.0) == 0.0  # over budget clips
    assert stars(5) == 1.0 and stars(1) == 0.0 and signed(-1) == 0.0 and signed(0) == 0.5
    for bad in (None, float("nan"), float("inf"), "x", True):
        assert benefit(bad, 0, 1) is None and cost(bad, 1, 2) is None and ratio_efficiency(bad, 1) is None
    assert ratio_efficiency(0.5, 0) is None and ratio_efficiency(-1, 1) is None  # zero budget / negative usage
    assert benefit(0.5, 1, 1) is None and stars(6) is None


# ---- weights --------------------------------------------------------------------------------------

def test_every_configured_group_sums_to_one():
    for name, group in W.groups.items():
        assert math.isclose(sum(group.weights.values()), 1.0) and set(group.weights) == set(GROUP_KEYS[name])


@pytest.mark.parametrize("weights", [{"QOE": 0.6, "SRE": 0.6}, {"QOE": 1.2, "SRE": -0.2},
                                     {"QOE": 1.0}, {"QOE": float("nan"), "SRE": 0.5}])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValidationError):
        WeightGroup(name="utility", weights=weights)


def test_missing_policies():
    w = {"a": 0.5, "b": 0.5}
    assert aggregate({"a": 1.0, "b": None}, w, "renormalize").score == 1.0
    assert aggregate({"a": 1.0, "b": None}, w, "renormalize").coverage == 0.5
    assert aggregate({"a": 1.0, "b": None}, w, "neutral").score == 0.75
    assert aggregate({"a": None, "b": None}, w, "renormalize").score is None  # never 0
    with pytest.raises(MissingMetricError):
        aggregate({"a": 1.0, "b": None}, w, "strict")


# ---- QoE ------------------------------------------------------------------------------------------

def test_best_inputs_give_maximal_qoe():
    result = compute_qoe(metrics(1.0), W)
    assert result.qoe == 1.0 and result.coverage == 1.0 and all(v == 1.0 for v in result.components.values())


def test_poor_education_lowers_qoe():
    good = compute_qoe(metrics(0.8), W).qoe
    poor = compute_qoe(metrics(0.8, LG=0.1, PT=0.1, RA=0.1, LS=0.1), W)
    assert poor.qoe < good and poor.components["EQ"] == pytest.approx(0.1)


def test_higher_latency_lowers_latency_quality_and_qoe():
    bounds = QoEBounds()
    fast = normalize_qoe(QoERaw(latency_ms=1000, post_test=0.8), bounds)
    slow = normalize_qoe(QoERaw(latency_ms=12000, post_test=0.8), bounds)
    assert fast.PL == 1.0 and slow.PL < fast.PL
    assert compute_qoe(slow, W).qoe < compute_qoe(fast, W).qoe


def test_missing_metrics_follow_documented_policy():
    partial = QoEMetrics(PT=0.8, RQ=0.6, PL=1.0, SR=1.0)  # no AQ besides RQ, no LQ/SQ/TQ, no FS/CR...
    result = compute_qoe(partial, W)
    assert result.groups["aq"].missing == ["LQ", "SQ", "TQ"] and result.components["AQ"] == 0.6  # renormalised
    assert 0 < result.coverage < 1 and result.missing_groups == []
    only_eq = compute_qoe(QoEMetrics(PT=0.9), W)
    assert only_eq.missing_groups == ["AQ", "NQ", "UX"] and only_eq.qoe == 0.9  # unavailable ≠ 0
    assert compute_qoe(QoEMetrics(), W).qoe is None
    with pytest.raises(MissingMetricError):
        compute_qoe(partial, W, "strict")
    assert compute_qoe(QoEMetrics(), W, "neutral").qoe == 0.5


def test_normalised_metrics_reject_out_of_range():
    with pytest.raises(ValidationError):
        QoEMetrics(PT=1.5)


# ---- SRE ------------------------------------------------------------------------------------------

def test_lower_consumption_increases_efficiency_and_stays_bounded():
    low = compute_sre(SREInputs(cpu_usage=0.2, memory_usage=0.3, energy_wh=0.5, bandwidth_bytes=5e6), W)
    high = compute_sre(SREInputs(cpu_usage=0.9, memory_usage=0.9, energy_wh=1.9, bandwidth_bytes=45e6), W)
    assert low.sre > high.sre and all(0 <= s <= 1 for s in (low.sre, high.sre))
    over = compute_sre(SREInputs(cpu_usage=1.0, memory_usage=1.0, energy_wh=10, bandwidth_bytes=1e9), W)
    assert over.sre == 0.0
    assert compute_sre(SREInputs(), W).sre is None  # nothing measured → unavailable


def test_sre_invalid_budgets_and_estimates():
    result = compute_sre(SREInputs(cpu_usage=0.5, cpu_max=0.0, memory_usage=0.5, energy_wh=-1), W)
    assert result.components["CPU"] is None and result.components["EN"] is None and result.components["MEM"] == 0.5
    assert estimate_energy_wh(0.5, 3600, idle_power_w=10, max_power_w=30) == 20.0
    assert estimate_energy_wh(1.5, 10, idle_power_w=10, max_power_w=30) is None
    noted = compute_sre(SREInputs(cpu_usage=0.5, energy_wh=0.1, energy_estimated=True), W)
    assert "estimated" in noted.notes[0]


# ---- utility --------------------------------------------------------------------------------------

def test_utility_formula_and_validation():
    assert compute_utility(0.8, 0.4, W).utility == pytest.approx(0.7 * 0.8 + 0.3 * 0.4)
    custom = W.replace("utility", {"QOE": 0.5, "SRE": 0.5}, source="test")
    assert compute_utility(0.8, 0.4, custom).utility == pytest.approx(0.6)
    with pytest.raises(ValueError):
        compute_utility(1.2, 0.5, W)


# ---- reward ---------------------------------------------------------------------------------------

def _state(qoe, sre):
    return RuntimeState(context={}, qoe_metrics=metrics(qoe),
                        sre_inputs=SREInputs(cpu_usage=1 - sre, memory_usage=1 - sre, energy_wh=(1 - sre) * 2,
                                             bandwidth_bytes=(1 - sre) * 50e6))


ACTION = AdaptationAction(action_id="P2", config={"top_k": 5})


def test_reward_sign_follows_improvement():
    up = compute_reward(_state(0.5, 0.5), _state(0.8, 0.6), ACTION, W)
    assert up.reward == pytest.approx(0.7 * 0.3 + 0.3 * 0.1) and up.reward > 0
    assert up.delta_qoe == pytest.approx(0.3) and up.delta_sre == pytest.approx(0.1)
    down = compute_reward(_state(0.8, 0.6), _state(0.5, 0.5), ACTION, W)
    assert down.reward == pytest.approx(-up.reward)
    assert compute_reward(_state(0.6, 0.6), _state(0.6, 0.6), ACTION, W).reward == 0.0


def test_reward_is_deterministic_and_records_weights():
    a = compute_reward(_state(0.4, 0.5), _state(0.7, 0.4), ACTION, W)
    b = compute_reward(_state(0.4, 0.5), _state(0.7, 0.4), ACTION, W)
    assert a.model_dump() == b.model_dump() and a.weights_version == W.version


def test_first_session_uses_flagged_baseline_and_unmeasurable_next_state_fails():
    first = compute_reward(RuntimeState(context={}, baseline=True), _state(0.7, 0.7), ACTION, W,
                           baseline_qoe=0.5, baseline_sre=0.5)
    assert first.baseline_used and first.before.qoe == 0.5 and first.reward > 0
    with pytest.raises(IncompleteTransition):
        compute_reward(_state(0.5, 0.5), RuntimeState(context={}, qoe_metrics=QoEMetrics()), ACTION, W)


# ---- EWM ------------------------------------------------------------------------------------------

def test_ewm_matches_a_hand_computed_example():
    obs = [{"a": 1, "b": 1, "c": 5}, {"a": 2, "b": 1, "c": 5}, {"a": 3, "b": 4, "c": 5}]
    result = entropy_weights("t", obs, ["a", "b", "c"], {"a": 0.2, "b": 0.3, "c": 0.5}, min_observations=3)
    # a: p = (0, 1/3, 2/3); b: p = (0, 0, 1); c constant → no information
    e_a = -((1 / 3) * math.log(1 / 3) + (2 / 3) * math.log(2 / 3)) / math.log(3)
    d_a, d_b = 1 - e_a, 1.0
    assert result.method == "ewm" and result.weights["c"] == 0.0
    assert result.weights["a"] == pytest.approx(d_a / (d_a + d_b), abs=1e-5)
    assert sum(result.weights.values()) == pytest.approx(1.0) and all(w >= 0 for w in result.weights.values())


def test_ewm_fallbacks():
    base = {"a": 0.5, "b": 0.5}
    few = entropy_weights("t", [{"a": 1, "b": 2}] * 3, ["a", "b"], base, min_observations=10)
    assert few.method == "fallback" and few.weights == base and "complete observations" in few.fallback_reason
    const = entropy_weights("t", [{"a": 1, "b": 2}] * 12, ["a", "b"], base, min_observations=10)
    assert const.method == "fallback" and "constant" in const.fallback_reason
    with_missing = [{"a": i, "b": None if i % 2 else i * 2} for i in range(30)]
    result = entropy_weights("t", with_missing, ["a", "b"], base, min_observations=10)
    assert result.n_used == 15 and result.n_observations == 30  # complete cases only
    zeros = entropy_weights("t", [{"a": 0, "b": i} for i in range(12)], ["a", "b"], base, min_observations=10)
    assert zeros.weights == {"a": 0.0, "b": 1.0}
    assert {r["metric"] for r in result.comparison()} == {"a", "b"}


# ---- sentiment ------------------------------------------------------------------------------------

def test_lexicon_sentiment():
    assert lexicon_sentiment("Clear and helpful answers") == 1.0
    assert lexicon_sentiment("confusing and slow") == -1.0
    assert lexicon_sentiment("not helpful") == -1.0  # negation flips
    assert lexicon_sentiment("ok") is None and lexicon_sentiment(None) is None
