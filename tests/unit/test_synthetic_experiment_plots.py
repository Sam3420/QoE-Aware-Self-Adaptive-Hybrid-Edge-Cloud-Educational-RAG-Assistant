from evaluation.synthetic.experiment_plots import generate_synthetic_plots
from evaluation.synthetic.experiment_report import build_synthetic_experiment_report


def test_synthetic_plot_generation_creates_all_expected_pngs(tmp_path):
    report = build_synthetic_experiment_report(seed=2026, scenario_count=4)

    paths = generate_synthetic_plots(report, tmp_path)

    generated_paths = tuple(paths.__dict__.values())
    assert len(generated_paths) == 6
    assert all(path.exists() and path.suffix == ".png" for path in generated_paths)
