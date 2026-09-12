import csv

from evaluation.synthetic.experiment_report import (
    SYNTHETIC_DISCLAIMER,
    build_synthetic_experiment_report,
    export_report_csv,
)


def test_report_is_deterministic_and_uses_identical_scenario_inputs():
    first = build_synthetic_experiment_report(seed=2026, scenario_count=4)
    second = build_synthetic_experiment_report(seed=2026, scenario_count=4)

    assert first.scenarios == second.scenarios
    assert first.per_scenario == second.per_scenario
    assert first.disclaimer == SYNTHETIC_DISCLAIMER


def test_report_contains_paired_per_scenario_results_and_adaptive_actions():
    report = build_synthetic_experiment_report(seed=3, scenario_count=4)

    assert len(report.per_scenario) == len(report.scenarios) == 4
    assert [row.scenario_id for row in report.per_scenario] == [
        scenario.scenario_id for scenario in report.scenarios
    ]
    assert any(row.adaptive_action_selected for row in report.per_scenario)
    assert all(row.selected_action is None or row.adaptive_action_selected for row in report.per_scenario)


def test_report_aggregates_are_comparable_and_count_adaptations():
    report = build_synthetic_experiment_report(seed=5, scenario_count=5)
    aggregate = report.aggregate

    assert aggregate.baseline.scenario_count == aggregate.adaptive.scenario_count == 5
    assert aggregate.baseline.adaptation_count == 0
    assert aggregate.adaptive.adaptation_count == sum(
        row.adaptive_action_selected for row in report.per_scenario
    )
    assert aggregate.qoe_delta == 0.0
    assert aggregate.synthetic_sre_delta == 0.0
    assert aggregate.latency_delta_ms == 0.0


def test_report_exports_per_scenario_and_aggregate_csv(tmp_path):
    report = build_synthetic_experiment_report(seed=7, scenario_count=3)
    paths = export_report_csv(report, tmp_path)

    with paths.per_scenario.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with paths.aggregate.open(encoding="utf-8", newline="") as handle:
        aggregate_rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert "baseline_qoe" in rows[0]
    assert "adaptive_action_selected" in rows[0]
    assert [row["approach"] for row in aggregate_rows] == [
        "baseline",
        "adaptive",
        "adaptive_minus_baseline",
    ]
