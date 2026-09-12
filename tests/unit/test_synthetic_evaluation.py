from evaluation.synthetic_evaluation import (
    generate_synthetic_scenarios,
    run_synthetic_evaluation,
)


def test_synthetic_scenarios_are_deterministic_and_include_future_metric_inputs():
    first = generate_synthetic_scenarios(seed=17, scenario_count=3)
    second = generate_synthetic_scenarios(seed=17, scenario_count=3)

    assert first == second
    assert len(first) == 3
    inputs = first[0].inputs()
    assert {"eq", "aq", "nq", "ux", "sre_cpu", "sre_memory", "sre_energy", "sre_network_efficiency", "latency_ms", "cpu_percent", "memory_percent", "network_condition", "learning_gain", "resource_efficiency"} <= set(inputs)


def test_baseline_and_adaptive_modes_receive_the_same_scenarios():
    result = run_synthetic_evaluation(seed=9, scenario_count=4)

    assert result.baseline.scenarios == result.adaptive.scenarios
    assert [record.scenario_id for record in result.baseline.records] == [
        record.scenario_id for record in result.adaptive.records
    ]


def test_adaptive_mode_creates_runtime_configuration_updates():
    result = run_synthetic_evaluation(seed=3, scenario_count=4)

    assert result.baseline.aggregate.adaptation_count == 0
    assert result.adaptive.aggregate.adaptation_count > 0
    assert any(record.runtime_configuration_version > 1 for record in result.adaptive.records)


def test_synthetic_evaluation_produces_comparable_aggregates():
    result = run_synthetic_evaluation(seed=5, scenario_count=5)
    baseline = result.baseline.aggregate
    adaptive = result.adaptive.aggregate

    assert baseline.scenario_count == adaptive.scenario_count == 5
    assert baseline.mean_synthetic_qoe == adaptive.mean_synthetic_qoe
    assert baseline.mean_synthetic_sre == adaptive.mean_synthetic_sre
    assert baseline.mean_latency_ms == adaptive.mean_latency_ms
    assert baseline.mean_learning_gain == adaptive.mean_learning_gain
    assert baseline.mean_resource_efficiency == adaptive.mean_resource_efficiency
