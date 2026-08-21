from pathlib import Path
import os

import pandas as pd

from roughness.reporting import save_diagnostic_plots, summarize_metrics


def test_reporting_uses_writable_matplotlib_cache():
    cache = Path(os.environ["MPLCONFIGDIR"])

    assert cache.is_dir()
    assert cache.parent.name == ".cache"


def test_reporting_summarizes_metrics_and_writes_plots(tmp_path: Path):
    fold_metrics = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "fold": [0, 1, 0, 1],
            "mae": [0.2, 0.2, 0.1, 0.1],
            "rmse": [0.3, 0.3, 0.2, 0.2],
            "r2": [0.5, 0.5, 0.8, 0.8],
            "weighted_mae": [0.2, 0.2, 0.1, 0.1],
            "weighted_rmse": [0.3, 0.3, 0.2, 0.2],
            "weighted_r2": [0.5, 0.5, 0.8, 0.8],
            "fit_seconds": [0.01, 0.01, 0.02, 0.02],
            "predict_seconds": [0.001, 0.001, 0.002, 0.002],
        }
    )
    predictions = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "y_true": [1.0, 2.0, 1.0, 2.0],
            "y_pred": [1.2, 1.8, 1.1, 1.9],
        }
    )

    summary = summarize_metrics(fold_metrics)
    best = save_diagnostic_plots(predictions, summary, tmp_path)

    assert best == "b"
    assert (tmp_path / "prediction_scatter.png").is_file()
    assert (tmp_path / "residual_plot.png").is_file()
    assert "weighted_mae_mean" in summary.columns
