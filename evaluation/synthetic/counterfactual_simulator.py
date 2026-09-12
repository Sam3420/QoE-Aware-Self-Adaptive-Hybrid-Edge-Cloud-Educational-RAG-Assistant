"""Deterministic counterfactual assumptions for synthetic architecture evaluation.

These rules are deliberately synthetic: they neither implement the research
QoE/SRE equations nor represent real telemetry.  They make the evaluation
pipeline testable by applying bounded, action-specific trade-offs to the same
underlying scenario after the existing AdaptiveIntelligenceService decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean, median

from app.services.adaptive_intelligence_service import AdaptiveIntelligenceService


DEFAULT_SEED = 2026
DEFAULT_SCENARIO_COUNT = 100


@dataclass(frozen=True)
class CounterfactualScenario:
    """Synthetic underlying conditions shared by baseline and adaptive modes."""

    scenario_id: str
    eq: float
    aq: float
    nq: float
    ux: float
    latency_ms: int
    cpu_percent: float
    memory_percent: float
    energy_score: float
    network_condition: str
    retrieval_quality: float
    preferred_language: str
    provider: str
    response_language: str
    detail_level: str
    response_style: str
    baseline_qoe: float
    baseline_sre: float
    baseline_resource_efficiency: float
    baseline_learning_gain: float
    baseline_success: bool


@dataclass(frozen=True)
class SyntheticOutcome:
    qoe: float
    sre: float
    latency_ms: int
    resource_efficiency: float
    learning_gain: float
    success: bool


@dataclass(frozen=True)
class CounterfactualRecord:
    scenario: CounterfactualScenario
    baseline: SyntheticOutcome
    adaptive: SyntheticOutcome
    selected_action: str | None
    adaptive_configuration: dict[str, object]


@dataclass(frozen=True)
class CounterfactualAggregate:
    mode: str
    scenario_count: int
    mean_qoe: float
    mean_sre: float
    mean_latency_ms: float
    mean_resource_efficiency: float
    mean_learning_gain: float
    success_rate: float
    adaptation_count: int


@dataclass(frozen=True)
class CounterfactualExperiment:
    scenarios: tuple[CounterfactualScenario, ...]
    records: tuple[CounterfactualRecord, ...]
    baseline: CounterfactualAggregate
    adaptive: CounterfactualAggregate


def generate_counterfactual_scenarios(
    *, seed: int = DEFAULT_SEED, scenario_count: int = DEFAULT_SCENARIO_COUNT
) -> tuple[CounterfactualScenario, ...]:
    """Generate heterogeneous, deterministic synthetic inputs only."""
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")
    random = Random(seed)
    scenarios: list[CounterfactualScenario] = []
    action_patterns = ("provider", "model", "retrieval", "language", "detail", "style")
    networks = ("poor", "good", "moderate", "good")
    for index in range(scenario_count):
        pattern = action_patterns[index % len(action_patterns)]
        baseline_qoe = round(random.uniform(58.0, 92.0), 2)
        provider = "huggingface"
        retrieval_quality = round(random.uniform(0.78, 0.98), 2)
        preferred_language = "english"
        response_language = "english"
        detail_level = "medium"
        response_style = "concise"
        if pattern == "provider":
            baseline_qoe = round(random.uniform(35.0, 48.0), 2)
            provider = "local"
        elif pattern == "model":
            baseline_qoe = round(random.uniform(35.0, 48.0), 2)
        elif pattern == "retrieval":
            retrieval_quality = round(random.uniform(0.2, 0.6), 2)
        elif pattern == "language":
            preferred_language = "hindi"
        elif pattern == "detail":
            detail_level = "low"
        elif pattern == "style":
            response_style = "narrative"
        scenarios.append(
            CounterfactualScenario(
                scenario_id=f"synthetic-{index + 1}",
                eq=round(random.uniform(40.0, 95.0), 2),
                aq=round(random.uniform(40.0, 95.0), 2),
                nq=round(random.uniform(40.0, 95.0), 2),
                ux=round(random.uniform(40.0, 95.0), 2),
                latency_ms=random.randint(250, 2400),
                cpu_percent=round(random.uniform(15.0, 95.0), 2),
                memory_percent=round(random.uniform(15.0, 95.0), 2),
                energy_score=round(random.uniform(35.0, 95.0), 2),
                network_condition=networks[index % len(networks)],
                retrieval_quality=retrieval_quality,
                preferred_language=preferred_language,
                provider=provider,
                response_language=response_language,
                detail_level=detail_level,
                response_style=response_style,
                baseline_qoe=baseline_qoe,
                baseline_sre=round(random.uniform(35.0, 95.0), 2),
                baseline_resource_efficiency=round(random.uniform(35.0, 95.0), 2),
                baseline_learning_gain=round(random.uniform(0.05, 0.85), 3),
                baseline_success=random.random() > 0.12,
            )
        )
    return tuple(scenarios)


def run_counterfactual_experiment(
    *, seed: int = DEFAULT_SEED, scenario_count: int = DEFAULT_SCENARIO_COUNT
) -> CounterfactualExperiment:
    """Pair fixed baseline outcomes with adaptive counterfactual outcomes."""
    scenarios = generate_counterfactual_scenarios(seed=seed, scenario_count=scenario_count)
    service = AdaptiveIntelligenceService(repository=object())
    records: list[CounterfactualRecord] = []
    for scenario in scenarios:
        configuration: dict[str, object] = {
            "provider": scenario.provider,
            "model_id": "synthetic-primary-model",
            "cloud_model_id": "synthetic-light-model",
            "local_model_id": "synthetic-local-model",
            "knowledge_retrieval_top_k": 3,
            "response_language": scenario.response_language,
            "detail_level": scenario.detail_level,
            "response_style": scenario.response_style,
        }
        decision = service.select_action(
            learner_id=None,
            current_qoe_score=scenario.baseline_qoe,
            current_runtime_configuration=configuration,
            latency_ms=scenario.latency_ms,
            retrieval_quality_is_adequate=scenario.retrieval_quality >= 0.75,
            personalization_context={
                "preferred_language": scenario.preferred_language,
                "detail_level": "high" if scenario.detail_level == "low" else None,
                "response_style": "step_by_step" if scenario.response_style == "narrative" else None,
            },
        )
        baseline = SyntheticOutcome(
            qoe=scenario.baseline_qoe,
            sre=scenario.baseline_sre,
            latency_ms=scenario.latency_ms,
            resource_efficiency=scenario.baseline_resource_efficiency,
            learning_gain=scenario.baseline_learning_gain,
            success=scenario.baseline_success,
        )
        adaptive_configuration = decision.suggested_runtime_configuration if decision else configuration
        records.append(
            CounterfactualRecord(
                scenario=scenario,
                baseline=baseline,
                adaptive=_apply_synthetic_action_effect(scenario, baseline, decision.action if decision else None),
                selected_action=decision.action if decision else None,
                adaptive_configuration=adaptive_configuration,
            )
        )
    return CounterfactualExperiment(
        scenarios=scenarios,
        records=tuple(records),
        baseline=_aggregate("baseline", [record.baseline for record in records], 0),
        adaptive=_aggregate("adaptive", [record.adaptive for record in records], sum(record.selected_action is not None for record in records)),
    )


def _apply_synthetic_action_effect(
    scenario: CounterfactualScenario, baseline: SyntheticOutcome, action: str | None
) -> SyntheticOutcome:
    """Bounded synthetic assumptions; neutral or adverse effects are intentional."""
    qoe, sre, latency, efficiency, gain, success = (
        baseline.qoe,
        baseline.sre,
        baseline.latency_ms,
        baseline.resource_efficiency,
        baseline.learning_gain,
        baseline.success,
    )
    if action == "CHANGE_PROVIDER":
        if scenario.network_condition in {"good", "moderate"}:
            qoe, latency, sre = qoe + 5, latency + 100, sre - 3
        else:
            qoe, latency, efficiency = qoe - 4, latency + 220, efficiency - 4
    elif action == "CHANGE_MODEL":
        qoe, latency, sre, efficiency = qoe - 1, latency - 180, sre + 5, efficiency + 4
    elif action == "CHANGE_RETRIEVAL_K":
        if scenario.retrieval_quality < 0.75:
            qoe, gain, latency, sre = qoe + 6, gain + 0.06, latency + 90, sre - 2
        else:
            latency, efficiency = latency + 40, efficiency - 1
    elif action == "CHANGE_RESPONSE_DETAIL":
        qoe, gain, latency, efficiency = qoe + 3, gain + 0.09, latency + 75, efficiency - 1
    elif action == "CHANGE_RESPONSE_STYLE":
        qoe, gain, latency = qoe + 2, gain + 0.05, latency + 20
    elif action == "CHANGE_LANGUAGE" and scenario.preferred_language != scenario.response_language:
        qoe, gain, success = qoe + 4, gain + 0.12, True
    return SyntheticOutcome(
        qoe=_bound(qoe),
        sre=_bound(sre),
        latency_ms=max(1, int(latency)),
        resource_efficiency=_bound(efficiency),
        learning_gain=round(min(1.0, max(0.0, gain)), 3),
        success=success,
    )


def _aggregate(mode: str, outcomes: list[SyntheticOutcome], adaptation_count: int) -> CounterfactualAggregate:
    return CounterfactualAggregate(
        mode=mode,
        scenario_count=len(outcomes),
        mean_qoe=round(mean(outcome.qoe for outcome in outcomes), 2),
        mean_sre=round(mean(outcome.sre for outcome in outcomes), 2),
        mean_latency_ms=round(mean(outcome.latency_ms for outcome in outcomes), 2),
        mean_resource_efficiency=round(mean(outcome.resource_efficiency for outcome in outcomes), 2),
        mean_learning_gain=round(mean(outcome.learning_gain for outcome in outcomes), 3),
        success_rate=round(mean(outcome.success for outcome in outcomes), 4),
        adaptation_count=adaptation_count,
    )


def _bound(value: float) -> float:
    return round(min(100.0, max(0.0, value)), 2)
