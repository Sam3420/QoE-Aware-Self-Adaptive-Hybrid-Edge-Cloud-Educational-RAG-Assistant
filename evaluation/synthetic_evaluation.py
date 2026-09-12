"""Deterministic architecture validation with synthetic evaluation inputs.

This module does not collect telemetry or implement the research QoE/SRE equations.
Its synthetic values are only for exercising and comparing the prototype's static
and experience-adaptive execution paths before real experimental data is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
from statistics import mean

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.models import InteractionCreate, LearnerProfileCreate, LearningSessionCreate
from app.services.adaptive_intelligence_service import AdaptiveIntelligenceService
from app.services.experience_service import ExperienceService
from app.services.learning_trace_service import LearningTraceService
from app.services.monitoring_service import MonitoringService


DEFAULT_SEED = 2026


@dataclass(frozen=True)
class SyntheticScenario:
    """Synthetic evaluation inputs, not values collected from the live system."""

    scenario_id: str
    eq: float
    aq: float
    nq: float
    ux: float
    sre_cpu: float
    sre_memory: float
    sre_energy: float
    sre_network_efficiency: float
    latency_ms: int
    cpu_percent: float
    memory_percent: float
    network_condition: str
    learning_gain: float
    resource_efficiency: float
    synthetic_qoe: float
    synthetic_sre: float

    def inputs(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRecord:
    scenario_id: str
    mode: str
    observed_qoe: float
    synthetic_qoe: float
    synthetic_sre: float
    latency_ms: int
    learning_gain: float
    resource_efficiency: float
    adaptive_action: str | None
    runtime_configuration_version: int


@dataclass(frozen=True)
class EvaluationAggregate:
    mode: str
    scenario_count: int
    mean_observed_qoe: float
    mean_synthetic_qoe: float
    mean_synthetic_sre: float
    mean_latency_ms: float
    mean_learning_gain: float
    mean_resource_efficiency: float
    adaptation_count: int


@dataclass(frozen=True)
class EvaluationRun:
    mode: str
    scenarios: tuple[SyntheticScenario, ...]
    records: tuple[EvaluationRecord, ...]
    aggregate: EvaluationAggregate


@dataclass(frozen=True)
class SyntheticEvaluationResult:
    """A fair baseline-vs-adaptive architecture comparison over identical inputs."""

    baseline: EvaluationRun
    adaptive: EvaluationRun


def generate_synthetic_scenarios(
    *,
    seed: int = DEFAULT_SEED,
    scenario_count: int = 8,
) -> tuple[SyntheticScenario, ...]:
    """Create reproducible synthetic inputs for future QoE/SRE experimentation."""
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")

    random = Random(seed)
    network_conditions = ("poor", "moderate", "good")
    scenarios: list[SyntheticScenario] = []
    for index in range(scenario_count):
        eq = round(random.uniform(45.0, 95.0), 2)
        aq = round(random.uniform(45.0, 95.0), 2)
        nq = round(random.uniform(40.0, 95.0), 2)
        ux = round(random.uniform(45.0, 95.0), 2)
        sre_cpu = round(random.uniform(35.0, 90.0), 2)
        sre_memory = round(random.uniform(35.0, 90.0), 2)
        sre_energy = round(random.uniform(35.0, 90.0), 2)
        sre_network_efficiency = round(random.uniform(35.0, 90.0), 2)
        scenarios.append(
            SyntheticScenario(
                scenario_id=f"synthetic-{index + 1}",
                eq=eq,
                aq=aq,
                nq=nq,
                ux=ux,
                sre_cpu=sre_cpu,
                sre_memory=sre_memory,
                sre_energy=sre_energy,
                sre_network_efficiency=sre_network_efficiency,
                latency_ms=random.randint(100, 2200),
                cpu_percent=round(random.uniform(10.0, 95.0), 2),
                memory_percent=round(random.uniform(10.0, 95.0), 2),
                network_condition=network_conditions[index % len(network_conditions)],
                learning_gain=round(random.uniform(0.05, 0.95), 3),
                resource_efficiency=round(random.uniform(35.0, 95.0), 2),
                synthetic_qoe=round(random.uniform(40.0, 95.0), 2),
                synthetic_sre=round(random.uniform(35.0, 95.0), 2),
            )
        )
    return tuple(scenarios)


def run_synthetic_evaluation(
    *,
    seed: int = DEFAULT_SEED,
    scenario_count: int = 8,
) -> SyntheticEvaluationResult:
    """Run static and adaptive modes over the exact same synthetic scenarios."""
    scenarios = generate_synthetic_scenarios(seed=seed, scenario_count=scenario_count)
    return SyntheticEvaluationResult(
        baseline=_run_mode(mode="baseline", scenarios=scenarios),
        adaptive=_run_mode(mode="adaptive", scenarios=scenarios),
    )


def _run_mode(*, mode: str, scenarios: tuple[SyntheticScenario, ...]) -> EvaluationRun:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with session_factory() as session:
        repository = LearningTraceRepository(session)
        trace_service = LearningTraceService(repository)
        learner = trace_service.create_learner(LearnerProfileCreate(display_name="Synthetic Learner"))
        learning_session = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
        runtime_configuration = trace_service.create_runtime_configuration_snapshot(
            name=f"synthetic-{mode}",
            description=f"{mode} synthetic evaluation configuration",
            configuration_data={
                "provider": "huggingface",
                "model_id": "synthetic-model",
                "knowledge_retrieval_top_k": 3,
            },
        )
        monitoring_service = MonitoringService(repository)
        experience_service = ExperienceService(repository)
        adaptive_service = AdaptiveIntelligenceService(repository)
        records: list[EvaluationRecord] = []

        for scenario in scenarios:
            current_runtime = repository.get_latest_runtime_configuration_by_name(
                runtime_configuration.name,
            ) or runtime_configuration
            interaction = trace_service.record_interaction(
                InteractionCreate(
                    session_id=learning_session.id,
                    runtime_configuration_id=current_runtime.id,
                    user_input=f"Synthetic question: {scenario.scenario_id}",
                    assistant_output="Synthetic educational response. " * 8,
                    response_latency_ms=scenario.latency_ms,
                    model_provider="huggingface",
                    model_name="synthetic-model",
                    status="completed",
                )
            )
            monitoring_service.record_interaction_observations(interaction)
            observed_qoe = monitoring_service.evaluate_qoe(interaction)
            experience = experience_service.create_experience_for_interaction(interaction.id)
            experience.state_snapshot = {
                **experience.state_snapshot,
                "synthetic_evaluation_inputs": scenario.inputs(),
            }
            session.flush()

            decision = None
            if mode == "adaptive":
                decision = adaptive_service.select_action(
                    learner_id=learner.id,
                    current_qoe_score=observed_qoe.score,
                    current_runtime_configuration=current_runtime.configuration_data,
                    latency_ms=scenario.latency_ms,
                    retrieval_quality_is_adequate=True,
                )
                if decision is not None and decision.suggested_runtime_configuration is not None:
                    repository.create_runtime_configuration_version(
                        name=current_runtime.name,
                        description=f"Synthetic adaptive update from {decision.action}",
                        configuration_data=decision.suggested_runtime_configuration,
                    )

            records.append(
                EvaluationRecord(
                    scenario_id=scenario.scenario_id,
                    mode=mode,
                    observed_qoe=observed_qoe.score,
                    synthetic_qoe=scenario.synthetic_qoe,
                    synthetic_sre=scenario.synthetic_sre,
                    latency_ms=scenario.latency_ms,
                    learning_gain=scenario.learning_gain,
                    resource_efficiency=scenario.resource_efficiency,
                    adaptive_action=decision.action if decision is not None else None,
                    runtime_configuration_version=current_runtime.version,
                )
            )

        aggregate = _aggregate(mode=mode, records=records)
        session.commit()

    return EvaluationRun(
        mode=mode,
        scenarios=scenarios,
        records=tuple(records),
        aggregate=aggregate,
    )


def _aggregate(*, mode: str, records: list[EvaluationRecord]) -> EvaluationAggregate:
    return EvaluationAggregate(
        mode=mode,
        scenario_count=len(records),
        mean_observed_qoe=round(mean(record.observed_qoe for record in records), 2),
        mean_synthetic_qoe=round(mean(record.synthetic_qoe for record in records), 2),
        mean_synthetic_sre=round(mean(record.synthetic_sre for record in records), 2),
        mean_latency_ms=round(mean(record.latency_ms for record in records), 2),
        mean_learning_gain=round(mean(record.learning_gain for record in records), 3),
        mean_resource_efficiency=round(mean(record.resource_efficiency for record in records), 2),
        adaptation_count=sum(record.adaptive_action is not None for record in records),
    )
