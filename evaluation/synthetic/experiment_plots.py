"""Publication-style charts for synthetic baseline-vs-adaptive results only."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from evaluation.synthetic.experiment_report import SyntheticExperimentReport


@dataclass(frozen=True)
class PlotPaths:
    qoe_comparison: Path
    sre_resource_efficiency_comparison: Path
    latency_comparison: Path
    learning_gain_comparison: Path
    overall_normalized_comparison: Path
    adaptation_actions_distribution: Path


def generate_synthetic_plots(
    report: SyntheticExperimentReport,
    output_dir: Path,
) -> PlotPaths:
    """Generate six PNGs from one existing synthetic experiment report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = report.aggregate.baseline
    adaptive = report.aggregate.adaptive
    paths = PlotPaths(
        qoe_comparison=output_dir / "synthetic_qoe_comparison.png",
        sre_resource_efficiency_comparison=output_dir / "synthetic_sre_resource_efficiency_comparison.png",
        latency_comparison=output_dir / "synthetic_latency_comparison.png",
        learning_gain_comparison=output_dir / "synthetic_learning_gain_comparison.png",
        overall_normalized_comparison=output_dir / "synthetic_overall_normalized_comparison.png",
        adaptation_actions_distribution=output_dir / "synthetic_adaptation_actions_distribution.png",
    )

    _two_bar_chart(
        plt,
        paths.qoe_comparison,
        title="Synthetic Evaluation — QoE Comparison",
        ylabel="Mean observed QoE",
        baseline_value=baseline.mean_observed_qoe,
        adaptive_value=adaptive.mean_observed_qoe,
        disclaimer=report.disclaimer,
    )
    _grouped_bar_chart(
        plt,
        paths.sre_resource_efficiency_comparison,
        title="Synthetic Evaluation — SRE / Resource Efficiency",
        labels=["Synthetic SRE", "Resource efficiency"],
        baseline_values=[baseline.mean_synthetic_sre, baseline.mean_resource_efficiency],
        adaptive_values=[adaptive.mean_synthetic_sre, adaptive.mean_resource_efficiency],
        ylabel="Mean synthetic value",
        disclaimer=report.disclaimer,
    )
    _two_bar_chart(
        plt,
        paths.latency_comparison,
        title="Synthetic Evaluation — Average Latency (Lower Is Better)",
        ylabel="Mean latency (ms)",
        baseline_value=baseline.mean_latency_ms,
        adaptive_value=adaptive.mean_latency_ms,
        disclaimer=report.disclaimer,
    )
    _two_bar_chart(
        plt,
        paths.learning_gain_comparison,
        title="Synthetic Evaluation — Learning Gain Comparison",
        ylabel="Mean synthetic learning gain",
        baseline_value=baseline.mean_learning_gain,
        adaptive_value=adaptive.mean_learning_gain,
        disclaimer=report.disclaimer,
    )
    latency_reference = min(baseline.mean_latency_ms, adaptive.mean_latency_ms)
    _grouped_bar_chart(
        plt,
        paths.overall_normalized_comparison,
        title="Synthetic Evaluation — Overall Normalized Comparison",
        labels=["QoE", "Synthetic SRE", "Resource efficiency", "Latency efficiency", "Learning gain"],
        baseline_values=[
            baseline.mean_observed_qoe / 100.0,
            baseline.mean_synthetic_sre / 100.0,
            baseline.mean_resource_efficiency / 100.0,
            latency_reference / baseline.mean_latency_ms,
            baseline.mean_learning_gain,
        ],
        adaptive_values=[
            adaptive.mean_observed_qoe / 100.0,
            adaptive.mean_synthetic_sre / 100.0,
            adaptive.mean_resource_efficiency / 100.0,
            latency_reference / adaptive.mean_latency_ms,
            adaptive.mean_learning_gain,
        ],
        ylabel="Normalized value (higher is favorable)",
        disclaimer=report.disclaimer,
    )
    action_counts = Counter(
        comparison.selected_action
        for comparison in report.per_scenario
        if comparison.selected_action is not None
    )
    _action_distribution_chart(plt, paths.adaptation_actions_distribution, action_counts, report.disclaimer)
    return paths


def _two_bar_chart(
    plt,
    path: Path,
    *,
    title: str,
    ylabel: str,
    baseline_value: float,
    adaptive_value: float,
    disclaimer: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(["Baseline", "Adaptive"], [baseline_value, adaptive_value], color=["#4C78A8", "#F58518"])
    axis.set_title(title, weight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    _label_bars(axis, bars)
    _save(figure, path, disclaimer)


def _grouped_bar_chart(
    plt,
    path: Path,
    *,
    title: str,
    labels: list[str],
    baseline_values: list[float],
    adaptive_values: list[float],
    ylabel: str,
    disclaimer: str,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    positions = list(range(len(labels)))
    width = 0.36
    baseline_bars = axis.bar(
        [position - width / 2 for position in positions],
        baseline_values,
        width=width,
        label="Baseline",
        color="#4C78A8",
    )
    adaptive_bars = axis.bar(
        [position + width / 2 for position in positions],
        adaptive_values,
        width=width,
        label="Adaptive",
        color="#F58518",
    )
    axis.set_title(title, weight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, labels)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _label_bars(axis, baseline_bars)
    _label_bars(axis, adaptive_bars)
    _save(figure, path, disclaimer)


def _action_distribution_chart(plt, path: Path, action_counts: Counter[str], disclaimer: str) -> None:
    labels = list(sorted(action_counts)) or ["No adaptive action"]
    values = [action_counts[label] for label in labels] if action_counts else [0]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(labels, values, color="#54A24B")
    axis.set_title("Synthetic Evaluation — Adaptive Action Distribution", weight="bold")
    axis.set_ylabel("Selection count")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    _label_bars(axis, bars)
    _save(figure, path, disclaimer)


def _label_bars(axis, bars) -> None:
    for bar in bars:
        axis.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def _save(figure, path: Path, disclaimer: str) -> None:
    figure.text(0.5, 0.01, disclaimer, ha="center", va="bottom", fontsize=7, wrap=True)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(path, dpi=180)
    figure.clf()
