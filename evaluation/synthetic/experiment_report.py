"""Baseline-vs-adaptive reports for synthetic architecture evaluation only."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.synthetic.counterfactual_simulator import (
    DEFAULT_SCENARIO_COUNT,
    DEFAULT_SEED,
    CounterfactualAggregate,
    CounterfactualExperiment,
    CounterfactualScenario,
    run_counterfactual_experiment,
)


SYNTHETIC_DISCLAIMER = (
    "Synthetic counterfactual assumptions for architecture/evaluation validation only; "
    "not real telemetry or evidence of real-world adaptive-system superiority."
)


@dataclass(frozen=True)
class ScenarioComparison:
    scenario_id: str
    baseline_qoe: float
    adaptive_qoe: float
    baseline_sre: float
    adaptive_sre: float
    baseline_resource_efficiency: float
    adaptive_resource_efficiency: float
    baseline_latency_ms: int
    adaptive_latency_ms: int
    baseline_learning_gain: float
    adaptive_learning_gain: float
    baseline_success: bool
    adaptive_success: bool
    adaptive_action_selected: bool
    selected_action: str | None
    qoe_delta: float
    sre_delta: float
    latency_delta_ms: int
    resource_efficiency_delta: float
    learning_gain_delta: float

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateComparison:
    baseline: CounterfactualAggregate
    adaptive: CounterfactualAggregate
    qoe_delta: float
    sre_delta: float
    resource_efficiency_delta: float
    latency_delta_ms: float
    learning_gain_delta: float
    qoe_improvement_percent: float | None
    sre_improvement_percent: float | None
    resource_efficiency_improvement_percent: float | None
    latency_reduction_percent: float | None
    learning_gain_improvement_percent: float | None

    def as_row(self) -> dict[str, object]:
        return {
            "qoe_delta_adaptive_minus_baseline": self.qoe_delta,
            "sre_delta_adaptive_minus_baseline": self.sre_delta,
            "resource_efficiency_delta_adaptive_minus_baseline": self.resource_efficiency_delta,
            "latency_delta_ms_adaptive_minus_baseline": self.latency_delta_ms,
            "learning_gain_delta_adaptive_minus_baseline": self.learning_gain_delta,
            "qoe_improvement_percent": self.qoe_improvement_percent,
            "sre_improvement_percent": self.sre_improvement_percent,
            "resource_efficiency_improvement_percent": self.resource_efficiency_improvement_percent,
            "latency_reduction_percent": self.latency_reduction_percent,
            "learning_gain_improvement_percent": self.learning_gain_improvement_percent,
            "baseline_adaptation_count": self.baseline.adaptation_count,
            "adaptive_adaptation_count": self.adaptive.adaptation_count,
        }


@dataclass(frozen=True)
class SyntheticExperimentReport:
    """Paired synthetic results suitable for console display or CSV export."""

    disclaimer: str
    scenarios: tuple[SyntheticScenario, ...]
    per_scenario: tuple[ScenarioComparison, ...]
    aggregate: AggregateComparison

    def aggregate_rows(self) -> list[dict[str, object]]:
        return [
            _aggregate_row(self.aggregate.baseline),
            _aggregate_row(self.aggregate.adaptive),
            {"approach": "adaptive_minus_baseline", **self.aggregate.as_row()},
        ]


@dataclass(frozen=True)
class CsvExportPaths:
    per_scenario: Path
    aggregate: Path


def build_synthetic_experiment_report(
    *,
    seed: int = DEFAULT_SEED,
    scenario_count: int = 8,
) -> SyntheticExperimentReport:
    """Build a fair paired report over one deterministic scenario collection."""
    result = run_synthetic_evaluation(seed=seed, scenario_count=scenario_count)
    return _report_from_result(result)


def export_report_csv(report: SyntheticExperimentReport, directory: Path) -> CsvExportPaths:
    """Export paired rows and aggregate comparison without plotting dependencies."""
    directory.mkdir(parents=True, exist_ok=True)
    per_scenario_path = directory / "synthetic_per_scenario.csv"
    aggregate_path = directory / "synthetic_aggregate.csv"
    _write_csv(per_scenario_path, [comparison.as_row() for comparison in report.per_scenario])
    _write_csv(aggregate_path, report.aggregate_rows())
    return CsvExportPaths(per_scenario=per_scenario_path, aggregate=aggregate_path)


def _report_from_result(result: SyntheticEvaluationResult) -> SyntheticExperimentReport:
    baseline_by_id = {record.scenario_id: record for record in result.baseline.records}
    adaptive_by_id = {record.scenario_id: record for record in result.adaptive.records}
    comparisons = tuple(
        ScenarioComparison(
            scenario_id=scenario.scenario_id,
            baseline_qoe=baseline_by_id[scenario.scenario_id].observed_qoe,
            adaptive_qoe=adaptive_by_id[scenario.scenario_id].observed_qoe,
            baseline_synthetic_sre=baseline_by_id[scenario.scenario_id].synthetic_sre,
            adaptive_synthetic_sre=adaptive_by_id[scenario.scenario_id].synthetic_sre,
            baseline_resource_efficiency=baseline_by_id[scenario.scenario_id].resource_efficiency,
            adaptive_resource_efficiency=adaptive_by_id[scenario.scenario_id].resource_efficiency,
            baseline_latency_ms=baseline_by_id[scenario.scenario_id].latency_ms,
            adaptive_latency_ms=adaptive_by_id[scenario.scenario_id].latency_ms,
            baseline_learning_gain=baseline_by_id[scenario.scenario_id].learning_gain,
            adaptive_learning_gain=adaptive_by_id[scenario.scenario_id].learning_gain,
            adaptive_action_selected=adaptive_by_id[scenario.scenario_id].adaptive_action is not None,
            selected_action=adaptive_by_id[scenario.scenario_id].adaptive_action,
        )
        for scenario in result.baseline.scenarios
    )
    return SyntheticExperimentReport(
        disclaimer=SYNTHETIC_DISCLAIMER,
        scenarios=result.baseline.scenarios,
        per_scenario=comparisons,
        aggregate=_compare_aggregates(result.baseline.aggregate, result.adaptive.aggregate),
    )


def _compare_aggregates(
    baseline: EvaluationAggregate,
    adaptive: EvaluationAggregate,
) -> AggregateComparison:
    return AggregateComparison(
        baseline=baseline,
        adaptive=adaptive,
        qoe_delta=round(adaptive.mean_observed_qoe - baseline.mean_observed_qoe, 2),
        synthetic_sre_delta=round(adaptive.mean_synthetic_sre - baseline.mean_synthetic_sre, 2),
        resource_efficiency_delta=round(
            adaptive.mean_resource_efficiency - baseline.mean_resource_efficiency,
            2,
        ),
        latency_delta_ms=round(adaptive.mean_latency_ms - baseline.mean_latency_ms, 2),
        learning_gain_delta=round(adaptive.mean_learning_gain - baseline.mean_learning_gain, 3),
        qoe_improvement_percent=_improvement_percent(
            baseline.mean_observed_qoe,
            adaptive.mean_observed_qoe,
        ),
        synthetic_sre_improvement_percent=_improvement_percent(
            baseline.mean_synthetic_sre,
            adaptive.mean_synthetic_sre,
        ),
        resource_efficiency_improvement_percent=_improvement_percent(
            baseline.mean_resource_efficiency,
            adaptive.mean_resource_efficiency,
        ),
        latency_reduction_percent=_improvement_percent(
            baseline.mean_latency_ms,
            adaptive.mean_latency_ms,
            lower_is_better=True,
        ),
        learning_gain_improvement_percent=_improvement_percent(
            baseline.mean_learning_gain,
            adaptive.mean_learning_gain,
        ),
    )


def _improvement_percent(
    baseline: float,
    adaptive: float,
    *,
    lower_is_better: bool = False,
) -> float | None:
    if baseline == 0:
        return None
    difference = baseline - adaptive if lower_is_better else adaptive - baseline
    return round((difference / baseline) * 100.0, 2)


def _aggregate_row(aggregate: EvaluationAggregate) -> dict[str, object]:
    return {
        "approach": aggregate.mode,
        "scenario_count": aggregate.scenario_count,
        "mean_qoe": aggregate.mean_observed_qoe,
        "mean_synthetic_sre": aggregate.mean_synthetic_sre,
        "mean_resource_efficiency": aggregate.mean_resource_efficiency,
        "mean_latency_ms": aggregate.mean_latency_ms,
        "mean_learning_gain": aggregate.mean_learning_gain,
        "adaptation_count": aggregate.adaptation_count,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Export one deterministic synthetic report and its plots."""
    parser = argparse.ArgumentParser(description="Export synthetic baseline-vs-adaptive results.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scenario-count", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output",
    )
    arguments = parser.parse_args()
    report = build_synthetic_experiment_report(
        seed=arguments.seed,
        scenario_count=arguments.scenario_count,
    )
    csv_paths = export_report_csv(report, arguments.output_dir)
    from evaluation.synthetic.experiment_plots import generate_synthetic_plots

    plot_paths = generate_synthetic_plots(report, arguments.output_dir / "plots")
    print(report.disclaimer)
    print(f"Wrote {csv_paths.per_scenario}")
    print(f"Wrote {csv_paths.aggregate}")
    for path in plot_paths.__dict__.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
